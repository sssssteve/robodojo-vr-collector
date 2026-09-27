"""Direct RoboDojo/Isaac Lab adapter. No policy server and no evaluator side effects."""
from copy import deepcopy
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf

from recording import CAMERAS


class RoboDojo:
    def __init__(self, root, app, task, layout, seed, device, phase_callback=None):
        from env.observation_manager.obs_manager import ObsManager
        from task.RoboDojo.task_registry import load_task_class
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        self.root, self.task, self.seed = Path(root), task, seed
        self.phase_callback = phase_callback or (lambda phase, **fields: None)
        self.layout_path = Path(layout).resolve()
        self.layout = json.loads(self.layout_path.read_text())
        base = load_yaml(str(self.root / "env_cfg/arx_x5.yml"))
        sections = {key: load_yaml(str(self.root / "env_cfg" / key / (value + ".yml")))
                    for key, value in base["config"].items()}
        sections["task_env"] = load_yaml(str(self.root / f"task/RoboDojo/config/{task}.yml"))
        sections["eval_cfg"] = base
        cfg = OmegaConf.create(sections)
        cfg = process_randomization(cfg)
        cfg, _ = process_config(cfg, task_name=task)
        cfg.sim.device = device
        cfg.sim.scene.num_envs = 1
        cfg.sim.seed = [seed]
        cfg.sim.decimation = 1
        self.particle_task = any(self.layout.get(kind) for kind in ("Garment", "Fluid"))
        cfg.sim.dt = .004 if self.particle_task else .008
        # The collector renders explicitly after enough physics steps for 0.04 s.
        # Keep Kit's internal render loop from rate-limiting that call to 25 Hz.
        cfg.sim.render_interval = 1
        cfg.camera.default_frequency = 25
        cfg.camera.cam_top = {"camera": {"type": "third_view", "mesh": "pinhole",
                                         "pos": [0.0, -0.05, 2.4], "ori": [0.0, 0.0, 0.0]}}
        _, task_class = load_task_class(task)
        self.phase_callback("initialize_env_enter", task=task)
        self.env = task_class(cfg, app)
        self.phase_callback("initialize_env_exit", task=task)
        self.cfg = cfg
        self.log = logging.getLogger("robodojo_pico")
        self.conveyor_surface = None
        self.conveyor_node_enabled = None
        self.top_frame = None
        if task in {"match_and_pick_from_conveyor", "pick_from_conveyor_by_image"}:
            from pxr import PhysxSchema, UsdShade

            reload_env_scene = self.env.scene_manager.reload_env_scene

            def reload_with_conveyor(env_id):
                reload_env_scene(env_id)
                self.conveyor_surface = None
                self.conveyor_node_enabled = None
                for obj in self.env.scene_manager.get_objects([env_id], object_type="dynamic").values():
                    if getattr(obj, "category_name", None) != "conveyor":
                        continue
                    belt = obj.stage.GetPrimAtPath(obj.usd_prim_path + "/ConveyorTrack/Belt")
                    if belt.IsValid():
                        material = obj.stage.GetPrimAtPath(
                            obj.usd_prim_path + "/ConveyorTrack/Physics_materials/ConveyorBelt_Mat")
                        UsdShade.MaterialBindingAPI.Apply(belt).Bind(
                            UsdShade.Material(material), materialPurpose="physics")
                        self.conveyor_surface = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt)
                        self.conveyor_surface.GetSurfaceVelocityEnabledAttr().Set(False)
                        graph = obj.stage.GetPrimAtPath(obj.usd_prim_path + "/ConveyorTrack/ConveyorBeltGraph")
                        velocity = graph.GetAttribute("graph:variable:Velocity")
                        velocity.Set(abs(velocity.Get()))
                        node = obj.stage.GetPrimAtPath(str(graph.GetPath()) + "/ConveyorNode")
                        self.conveyor_node_enabled = node.GetAttribute("inputs:enabled")
                        self.conveyor_node_enabled.Set(False)
                if self.conveyor_surface is None:
                    raise RuntimeError("Conveyor belt surface is missing")

            self.env.scene_manager.reload_env_scene = reload_with_conveyor
        manager = self.env.scene_manager.layout_manager
        get_pose = manager.get_instance_pose

        def numpy_pose(*args, **kwargs):
            # Pinned Dojo returns CUDA tensors for some objects, but its reward
            # parser calls NumPy directly. Normalize this instance's boundary.
            return tuple(v.detach().cpu().numpy() if hasattr(v, "detach") else v
                         for v in get_pose(*args, **kwargs))

        manager.get_instance_pose = numpy_pose
        manager.set_saved_layout(0, deepcopy(self.layout))
        self.log.info("Initializing task and cameras")
        self.phase_callback("initialize_reset_enter", task=task)
        self.env.reset(seed=[seed])
        self.phase_callback("initialize_reset_exit", task=task)
        for obj in self.env.scene_manager.get_objects([0], object_type="geometry").values():
            if hasattr(obj.env_origin, "detach"):
                obj.env_origin = obj.env_origin.detach().cpu()
        robots = [robot for robot in self.env.robot_manager.robot_list if robot.type == "target"]
        if len(robots) != 2 or any(robot.ee_type != "gripper" for robot in robots):
            raise ValueError("This collector requires two target gripper arms")
        self.robots = {r.arm_name.split("_")[0]: r for r in robots}
        self.support_robots = [robot for robot in self.env.robot_manager.robot_list
                               if robot.type != "target"]
        if set(self.robots) != {"left", "right"}:
            raise ValueError("Expected left and right target arms")
        obs_config = deepcopy(base["observation"])
        obs_config["collect_freq"] = 25
        obs_config["vision"].update(intrinsic_matrix=True, extrinsic_matrix=True)
        self.obs = ObsManager(obs_config, 1, self.env.dt, task, {}, seeds_per_env=[seed])
        self.obs.initialize(self.env)
        self.substeps = round(1 / (25 * self.env.dt))
        if not np.isclose(self.substeps * self.env.dt, .04):
            raise ValueError("Physics dt must divide the 25 Hz recording interval")
        self.ticks = 0
        self.ik_failures = 0
        self.support_motion_steps = 0
        self.last_ik_status = {side: "not_requested" for side in self.robots}
        self.last_step_profile = {"solve_ms": 0., "physics_ms": 0., "reward_ms": 0.,
                                  "substeps": self.substeps}
        self._initialize_episode()

    def reset(self):
        # Reuse this fixed layout's live objects and camera products. Full Dojo
        # reset regenerates GPU object names without recreating those objects.
        self.phase_callback("reset_seed_enter", sim_tick=self.ticks)
        self.env.update_seed([self.seed])
        self.env.robot_manager.reset()
        self.phase_callback("reset_physics_enter", sim_tick=self.ticks)
        for _ in range(300):
            self.env.sim_step(render=False)
        self.phase_callback("reset_physics_exit", sim_tick=self.ticks)
        self.env.reward_manager.reset()
        self._initialize_episode()
        self.phase_callback("reset_exit", sim_tick=self.ticks)

    def _initialize_episode(self):
        # Dojo uses env.success as the valid-layout mask, not the task outcome.
        self.env.success = [True]
        self.env.end_flag = [False]
        self.phase_callback("initialize_episode_enter", sim_tick=self.ticks)
        self.env.scene_manager.apply_saved_poses(env_idx_list=[0])
        stable, unstable = self.env.scene_manager.layout_manager.check_layout_stability(self.env)
        if not stable or unstable:
            raise RuntimeError(f"Scene layout is unstable: {unstable}")
        self.env.robot_manager.set_origin_endpose()
        self.env.robot_manager.set_robot_init_state()
        self.env.reward_manager.init_state()
        self.env.run_reward()
        self.obs.reset()
        self.ticks = 0
        self.ik_failures = 0
        self.last_ik_status = {side: "not_requested" for side in self.robots}
        # Populate the renderer after reset without advancing simulation time.
        self.phase_callback("initialize_render_enter", sim_tick=self.ticks)
        for _ in range(3):
            self.env.render()
        self.phase_callback("initialize_render_exit", sim_tick=self.ticks)
        self.hold_joints = {s: self.joints(s) for s in self.robots}
        self.home_joints = {s: joints.copy() for s, joints in self.hold_joints.items()}
        self.hold_grippers = {s: 1. for s in self.robots}
        self.log.info("Task ready; %s render products", len(self.env.capture_manager.tiled_cameras))
        self.phase_callback("initialize_episode_exit", sim_tick=self.ticks)

    def joints(self, side):
        return np.asarray(self.env.robot_manager.get_joint(self.robots[side], [0])[0]).copy()

    def set_task_motion_enabled(self, enabled):
        if self.conveyor_surface is not None:
            if self.conveyor_node_enabled.Get() != enabled:
                self.conveyor_node_enabled.Set(enabled)
            attr = self.conveyor_surface.GetSurfaceVelocityEnabledAttr()
            if attr.Get() != enabled:
                attr.Set(enabled)

    def poses(self):
        return {s: self.env.robot_manager.get_real_endpose(r, [0])[0].copy()
                for s, r in self.robots.items()}

    def observation(self):
        # Rendering happens after scene.update; no physics step between images and state.
        import omni.graph.core as og
        node = og.get_node_by_path("/Render/PostProcess/SDGPipeline/PostProcessDispatcher")

        def stamp():
            denominator = node.get_attribute("outputs:referenceTimeDenominator").get()
            numerator = node.get_attribute("outputs:referenceTimeNumerator").get()
            return float(numerator) / float(denominator) if denominator else None

        self.phase_callback("render_enter", sim_tick=self.ticks)
        started = time.perf_counter()
        before = stamp()
        render_calls = 0
        for _ in range(16):
            self.env.render()
            render_calls += 1
            delivered = stamp()
            if delivered is not None and delivered != before:
                break
        else:
            raise RuntimeError("Camera dispatcher did not deliver a fresh frame; recording stopped")
        rendered = time.perf_counter()
        self.phase_callback("render_exit", sim_tick=self.ticks, render_calls=render_calls)
        self.phase_callback("capture_enter", sim_tick=self.ticks)
        result = self.obs.get_obs([0])[0]
        captured = time.perf_counter()
        self.phase_callback("capture_exit", sim_tick=self.ticks)
        result["render_stamp"] = delivered
        if not all(c in result["vision"] for c in CAMERAS):
            raise RuntimeError("Missing required camera")
        self.top_frame = result["vision"]["cam_top"]["color"]
        result["vision"] = {c: result["vision"][c] for c in CAMERAS}
        state = {}
        for side, robot in self.robots.items():
            state[f"{side}_arm_joint_states"] = self.joints(side)
            actual = self.env.robot_manager.get_end_effector_real_val(robot, [0])[0][0]
            lo, hi = robot.gripper_scale
            opening = (actual - lo) / (hi - lo)
            if robot.gripper_move["sign"] != 1:
                opening = 1 - opening
            state[f"{side}_ee_joint_states"] = np.array([np.clip(opening, 0, 1)])
            state[f"{side}_ee_poses"] = self.env.robot_manager.get_real_endpose(robot, [0])[0].copy()
        result["state"] = state
        result = deepcopy(result)
        finished = time.perf_counter()
        self.last_observation_profile = {
            "render_ms": round((rendered - started) * 1000, 1),
            "capture_ms": round((captured - rendered) * 1000, 1),
            "state_copy_ms": round((finished - captured) * 1000, 1),
            "render_calls": render_calls,
        }
        return result

    def home_step(self, max_joint_step=.05, tolerance=.025, task_motion_enabled=True):
        """Move both arms toward their episode-start joints through one recorded control step."""
        for side in self.robots:
            current = self.joints(side)
            delta = self.home_joints[side] - current
            self.hold_joints[side] = current + np.clip(delta, -max_joint_step, max_joint_step)
            self.hold_grippers[side] = 1.
        command = self.step({}, {}, task_motion_enabled=task_motion_enabled)
        error = max(float(np.max(np.abs(self.home_joints[side] - self.joints(side))))
                    for side in self.robots)
        done = error <= tolerance
        self.last_ik_status = {side: "home_complete" if done else "homing"
                               for side in self.robots}
        return command, done, error

    def step(self, targets, grippers, task_motion_enabled=True):
        from env.robot_manager.control_manager import MetaControl

        self.phase_callback("ik_enter", sim_tick=self.ticks)
        started = time.perf_counter()
        current = {s: self.joints(s) for s in self.robots}
        self.last_ik_status = {side: "not_requested" for side in self.robots}
        for side, pose in targets.items():
            result = self.env.robot_manager.solve_ik(pose, 0, self.robots[side])
            if result["status"] == "Success":
                joints = np.asarray(result["joint_value"])
                if np.isfinite(joints).all() and np.max(np.abs(joints - current[side])) <= .35:
                    self.hold_joints[side] = joints.copy()
                    self.last_ik_status[side] = "accepted"
                else:
                    self.ik_failures += 1
                    self.last_ik_status[side] = "rejected_joint_jump_or_nonfinite"
            else:
                self.ik_failures += 1
                self.last_ik_status[side] = f"rejected_{result['status']}"
        self.hold_grippers.update(grippers)
        solved = time.perf_counter()
        self.phase_callback("ik_exit", sim_tick=self.ticks, ik_status=self.last_ik_status)
        command = {}
        for side in self.robots:
            command[f"{side}_arm_joint_states"] = self.hold_joints[side].copy()
            command[f"{side}_ee_joint_states"] = np.array([self.hold_grippers[side]])
        support_hold = {}
        for robot in self.support_robots:
            support_hold[self.env.robot_manager.process_name(robot.arm_name)] = {
                "position": self.env.robot_manager.get_joint(robot, [0])[0]}
            gripper = self.env.robot_manager.get_end_effector_real_val(robot, [0])[0]
            support_hold[self.env.robot_manager.process_name(robot.gripper_name)] = {
                "position": [np.asarray(gripper).reshape(-1)[0]]}
        if task_motion_enabled and getattr(self.env, "interact", False):
            if hasattr(self.env, "query_support_arm_traj"):
                self.env.query_support_arm_traj(0)
        self.phase_callback("physics_enter", sim_tick=self.ticks, physics_calls=self.substeps)
        for step in range(self.substeps):
            control = {name: dict(value) for name, value in support_hold.items()}
            support_actions = getattr(self.env, "support_arm_action", None)
            if task_motion_enabled and support_actions and support_actions[0]:
                support_step = support_actions[0].pop(0)
                control.update(support_step)
                support_hold.update(support_step)
                self.support_motion_steps += 1
            alpha = min((step + 1) / max(1, self.substeps * .8), 1.)
            for side, robot in self.robots.items():
                control[self.env.robot_manager.process_name(robot.arm_name)] = {
                    "position": (1 - alpha) * current[side] + alpha * self.hold_joints[side]}
                lo, hi = robot.gripper_scale
                opening = self.hold_grippers[side]
                value = lo + (opening if robot.gripper_move["sign"] == 1 else 1 - opening) * (hi - lo)
                control[self.env.robot_manager.process_name(robot.gripper_name)] = {
                    "position": [value, value * robot.gripper_move["mimic"][1] + robot.gripper_move["mimic"][2]]}
            meta = MetaControl(control)
            self.env.robot_manager.control_manager.update_prev_control(0, meta)
            self.env.robot_manager.control_robot([meta])
            self.env.sim_step(render=False)
        simulated = time.perf_counter()
        self.phase_callback("physics_exit", sim_tick=self.ticks, physics_calls=self.substeps)
        self.phase_callback("reward_enter", sim_tick=self.ticks)
        self.env.reward_manager.step([0])
        if task_motion_enabled and getattr(self.env, "interact", False):
            if hasattr(self.env, "query_support_arm_traj"):
                self.env.query_support_arm_traj(0)
            if hasattr(self.env, "check_support_arm_stable"):
                self.env.check_support_arm_stable(0)
        rewarded = time.perf_counter()
        self.phase_callback("reward_exit", sim_tick=self.ticks)
        self.ticks += 1
        self.last_step_profile = {
            "solve_ms": round((solved - started) * 1000, 1),
            "physics_ms": round((simulated - solved) * 1000, 1),
            "reward_ms": round((rewarded - simulated) * 1000, 1),
            "substeps": self.substeps,
        }
        return command

    def success(self):
        return bool(self.env.reward_manager.get_reward()[0] >= 1.)

    @staticmethod
    def _array(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value).squeeze().copy()

    def state_snapshot(self):
        """Capture one takeover state; particle tasks stay explicitly non-restorable."""
        snapshot = {"recoverable": True, "reason": "", "robots": {}, "objects": {}}
        for side, robot in self.robots.items():
            key = self.env.robot_manager.robot_key[self.env.robot_manager.robot_list.index(robot)]
            snapshot["robots"][side] = {
                "joint_position": self._array(key.data.joint_pos[0]),
                "joint_velocity": self._array(key.data.joint_vel[0]),
            }
        for name, obj in self.env.scene_manager.get_objects([0], object_type="rigid").items():
            position, orientation = obj.get_world_pose()
            snapshot["objects"][name] = {
                "type": "rigid",
                "position": self._array(position),
                "orientation": self._array(orientation),
                "linear_velocity": self._array(obj.get_linear_velocity()),
                "angular_velocity": self._array(obj.get_angular_velocity()),
            }
        for name, obj in self.env.scene_manager.get_objects([0], object_type="articulation").items():
            position, orientation = obj.get_world_pose()
            snapshot["objects"][name] = {
                "type": "articulation",
                "position": self._array(position),
                "orientation": self._array(orientation),
                "joint_position": self._array(obj.get_joint_positions()),
                "joint_velocity": self._array(obj.get_joint_velocities()),
            }
        unsupported = []
        for kind in ("garment", "fluid"):
            for name in self.env.scene_manager.get_objects([0], object_type=kind):
                unsupported.append(f"{kind}:{name}")
        if unsupported:
            snapshot["recoverable"] = False
            snapshot["reason"] = "衣物或液体的粒子速度状态尚未通过恢复验证：" + ", ".join(unsupported)
        return snapshot

    def restore_snapshot(self, snapshot):
        """Restore a state written by state_snapshot and verify immediate pose error."""
        if not snapshot or not snapshot.get("recoverable"):
            raise ValueError(snapshot.get("reason", "Episode has no recoverable state") if snapshot else
                             "Episode has no recoverable state")
        import torch

        self.reset()
        for side, state in snapshot["robots"].items():
            robot = self.robots[side]
            key = self.env.robot_manager.robot_key[self.env.robot_manager.robot_list.index(robot)]
            position = torch.as_tensor(state["joint_position"], dtype=torch.float32,
                                       device=key.device).unsqueeze(0)
            velocity = torch.as_tensor(state["joint_velocity"], dtype=torch.float32,
                                       device=key.device).unsqueeze(0)
            env_ids = torch.tensor([0], dtype=torch.long, device=key.device)
            key.write_joint_state_to_sim(position, velocity, env_ids=env_ids)
        objects = self.env.scene_manager.get_objects([0])
        aliases = {}
        manager = self.env.scene_manager.layout_manager
        for object_type in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid"):
            for record in manager.get_layout_records(0, object_type):
                label, instance = record.get("label"), record.get("inst_name")
                if label and instance:
                    match = next((key for key in objects if key.endswith(f"_{instance}")), None)
                    if match is not None:
                        aliases[label] = match
        errors = []
        for name, state in snapshot["objects"].items():
            obj = objects.get(name)
            if obj is None:
                obj = objects.get(aliases.get(name))
            if obj is None:
                raise ValueError(f"Recovery object missing: {name}")
            kind = state["type"]
            obj.set_world_pose(position=state["position"], orientation=state["orientation"])
            if kind == "rigid":
                velocity = np.r_[state["linear_velocity"], state["angular_velocity"]]
                obj._rigid_prim_view.set_velocities(
                    torch.as_tensor(velocity, dtype=torch.float32).unsqueeze(0))
            elif kind == "articulation":
                obj.set_joint_positions(torch.as_tensor(state["joint_position"], dtype=torch.float32))
                obj.set_joint_velocities(torch.as_tensor(state["joint_velocity"], dtype=torch.float32))
            position, _ = obj.get_world_pose()
            errors.append(float(np.max(np.abs(self._array(position) - state["position"]))))
        for side, state in snapshot["robots"].items():
            robot = self.robots[side]
            key = self.env.robot_manager.robot_key[self.env.robot_manager.robot_list.index(robot)]
            errors.append(float(np.max(np.abs(self._array(key.data.joint_pos[0]) - state["joint_position"]))))
        max_error = max(errors, default=0.)
        if max_error > 1e-3:
            raise RuntimeError(f"Recovery restore error exceeds 1e-3: {max_error}")
        self.hold_joints = {side: self.joints(side) for side in self.robots}
        for side, robot in self.robots.items():
            actual = self.env.robot_manager.get_end_effector_real_val(robot, [0])[0][0]
            lo, hi = robot.gripper_scale
            opening = (actual - lo) / (hi - lo)
            if robot.gripper_move["sign"] != 1:
                opening = 1 - opening
            self.hold_grippers[side] = float(np.clip(opening, 0, 1))
        self.env.reward_manager.step([0])
        for _ in range(3):
            self.env.render()
        return {"max_restore_error": max_error, "objects": len(snapshot["objects"]),
                "semantic_aliases": len(aliases)}

    def metadata(self):
        commit = subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        return {"task": self.task, "seed": self.seed, "robodojo_commit": commit,
                "layout_path": str(self.layout_path), "layout": self.layout,
                "layout_sha256": hashlib.sha256(self.layout_path.read_bytes()).hexdigest(),
                "layout_is_official_eval": "Eval_Layout" in self.layout_path.parts,
                "embodiment": "dual_x5", "control_hz": 25, "pose_order": "xyz_wxyz",
                "pose_frame": "environment_world", "gripper": "0_closed_1_open",
                "physics_hz": round(1 / self.env.dt),
                "sim_config": OmegaConf.to_container(self.cfg, resolve=True)}

    def close(self):
        # RoboDojo BaseEnv.close() stops the timeline. In Isaac Lab 2.3.2 the
        # standalone stop callback otherwise renders in a loop waiting for play,
        # which cannot be issued while this same call stack is blocked.
        from isaaclab.sim import SimulationContext
        context = SimulationContext.instance()
        if context is not None:
            context._disable_app_control_on_stop_handle = True
        self.env.close()
