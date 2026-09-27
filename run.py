#!/usr/bin/env python3
"""Run with the Python interpreter belonging to the RoboDojo Isaac Lab install."""
import argparse
import ast
from collections import deque
import json
import logging
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time

import numpy as np

from prompts_zh import PROMPTS_ZH, chinese_prompt
from watchdog import ProgressFile

TASK_SWITCH_EXIT = 75
SUPPORT_MOTION_GATED_TASKS = {
    "imitate_sorting_sequence", "make_kong", "play_tic_tac_toe",
}
CONVEYOR_TASKS = {"match_and_pick_from_conveyor", "pick_from_conveyor_by_image"}
RECORDING_GATED_TASKS = SUPPORT_MOTION_GATED_TASKS | CONVEYOR_TASKS


def available_tasks(root):
    root = Path(root)
    layouts = {path.name[:-7] for path in
               (root / "Assets/Eval_Layout/RoboDojo/arx_x5/0").glob("*_0.json")}
    configs = {path.stem for path in (root / "task/RoboDojo/config").glob("*.yml")}
    modules = {path.stem for path in (root / "task/RoboDojo/tasks").glob("*.py")}
    return sorted(layouts & configs & modules)


def task_requires_fluid(root, task):
    for layout_path in task_layouts(root, task).values():
        try:
            if json.loads(layout_path.read_text()).get("Fluid"):
                return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return False


def task_runtime_device(root, task):
    return "cpu" if task_requires_fluid(root, task) else "cuda:0"


def task_layouts(root, task):
    paths = (Path(root) / "Assets/Eval_Layout/RoboDojo/arx_x5/0").glob(f"{task}_*.json")
    result = {}
    for path in paths:
        suffix = path.stem[len(task) + 1:]
        if suffix.isdigit():
            result[int(suffix)] = path
    return result


def load_scene_state(path, default_seed=0):
    state = {"enabled": False, "seed": int(default_seed)}
    if path is None or not Path(path).is_file():
        return state
    try:
        saved = json.loads(Path(path).read_text())
        if isinstance(saved.get("enabled"), bool) and type(saved.get("seed")) is int:
            return {"enabled": saved["enabled"], "seed": saved["seed"]}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return state


def save_scene_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state) + "\n")


def select_task_layout(root, task, explicit_layout, state):
    if explicit_layout is not None:
        return Path(explicit_layout).resolve(), int(state["seed"]), 1
    layouts = task_layouts(root, task)
    if not layouts:
        raise ValueError(f"No official layouts found for task: {task}")
    seed = int(state["seed"]) if state["enabled"] else 0
    if seed not in layouts:
        seed = min(layouts)
    return layouts[seed].resolve(), seed, len(layouts)


def next_scene_seed(root, task, current):
    seeds = sorted(task_layouts(root, task))
    choices = [seed for seed in seeds if seed != current] or seeds
    if not choices:
        raise ValueError(f"No official layouts found for task: {task}")
    return secrets.choice(choices)


def task_identity(task):
    is_random_variant = task.endswith("_random")
    return {
        "task_family": task[:-len("_random")] if is_random_variant else task,
        "task_variant": "random" if is_random_variant else "standard",
    }


def task_catalog(root, tasks):
    """Read operator-facing task text from the pinned RoboDojo task sources."""
    root = Path(root)

    def render_string(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(render_string(value) if isinstance(value, ast.Constant) else "{…}"
                           for value in node.values)
        return ""

    result = {}
    for task in tasks:
        source = root / "task/RoboDojo/tasks" / f"{task}.py"
        text = source.read_text()
        tree = ast.parse(text)
        method = next((node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "gen_instruction"), None)
        candidates = []
        if method is not None:
            for node in ast.walk(method):
                if isinstance(node, (ast.Constant, ast.JoinedStr)):
                    value = render_string(node).strip()
                    if len(value) >= 12:
                        candidates.append(value)
        instruction = max(candidates, key=len, default=task.replace("_", " "))
        requires_origin = "all_robot_back_to_origin" in text
        success = "完成上述任务目标，并使 RoboDojo 官方 run_reward 的全部判据通过。"
        if requires_origin:
            success += " 最后将双臂回到初始位。"
        result[task] = {
            "title": task.replace("_", " "),
            "instruction": instruction,
            "prompt_zh": chinese_prompt(task, instruction),
            "success": success,
            "requires_robot_origin": requires_origin,
            "source": str(source.relative_to(root)),
        }
    return result


def scene_objects(layout):
    rows = []
    for kind in ("Rigid", "Articulation", "Garment", "Fluid", "Geometry"):
        for category, instances in layout.get(kind, {}).items():
            labels = [item.get("label", category) for item in instances]
            if labels == ["camera_stand"]:
                continue
            rows.append(f"{kind}: {', '.join(labels)}")
    return rows


def preview_mosaic(vision):
    """Head view in the center, gripper-following wrist views on both sides."""
    head = np.asarray(vision["cam_head"]["color"], dtype=np.uint8)
    left = np.asarray(vision["cam_left_wrist"]["color"], dtype=np.uint8)[::2, ::2]
    right = np.asarray(vision["cam_right_wrist"]["color"], dtype=np.uint8)[::2, ::2]
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[120:600, 320:960] = head
    canvas[240:480, :320] = left
    canvas[240:480, 960:] = right
    return canvas


def persistent_token(path):
    if path is None:
        return secrets.token_urlsafe(24)
    path = Path(path).resolve()
    if path.is_file():
        token = path.read_text().strip()
        if token:
            return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    path.write_text(token + "\n")
    path.chmod(0o600)
    return token


def run_input_only(args):
    """Exercise the real browser/WebXR/USB transport without importing Isaac."""
    from recording import jpeg
    from server import Bridge, Server

    progress_path = args.progress_file or Path(__file__).with_name("runtime_progress.json")
    progress = ProgressFile(progress_path, args.epoch_file)
    env_epoch, _ = progress.start_epoch("input-only")
    bridge = Bridge(env_epoch, progress.path)
    token = persistent_token(args.token_file)
    if args.adb_serial:
        subprocess.run(["adb", "-s", args.adb_serial, "reverse",
                        f"tcp:{args.port}", f"tcp:{args.port}"], check=True)
    server = Server(bridge, args.host, args.port, token, args.cert, args.key,
                    args.output.resolve(), args.allow_http_lan)
    server.start()
    bridge.publish_image(jpeg(np.zeros((720, 1280, 3), dtype=np.uint8), quality=60))
    print(f"INPUT_ONLY_READY http://127.0.0.1:{args.port}/#token={token}", flush=True)
    last_log = 0.
    try:
        while True:
            packet, commands, input_status = bridge.snapshot()
            if packet is not None:
                bridge.mark_applied(packet)
            for command in commands:
                if command["name"] == "pause":
                    continue
                operation_id = command.get("operation_id")
                if operation_id:
                    bridge.update_operation(operation_id, "failed", "input_only",
                                            "input-only 模式不执行环境操作")
            status = {
                "phase": "input_only", "task": "input-only", "env_epoch": env_epoch,
                "sim_tick": 0, "input": input_status,
                "message": "仅诊断 WebXR/USB 输入；未启动 Isaac、IK、渲染或录制。",
            }
            bridge.publish_status(status)
            progress.update("input_only", input=input_status)
            if time.monotonic() - last_log >= 2:
                print("INPUT", json.dumps(input_status, ensure_ascii=False), flush=True)
                last_log = time.monotonic()
            time.sleep(.02)
    except KeyboardInterrupt:
        return 0
    finally:
        progress.update("shutdown_input_only")
        server.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robodojo-root", type=Path)
    parser.add_argument("--task", default="stack_blocks")
    parser.add_argument("--layout", type=Path, help="Saved scene JSON; default: official layout 0, explicitly tagged")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("pico_data"))
    parser.add_argument("--max-wall-gap", type=float, default=.2,
                        help="Reject an episode when valid samples contain a larger wall-clock gap")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--cert", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--allow-http-lan", action="store_true",
                        help="Allow token-protected HTTP/WebSockets on a trusted private LAN")
    parser.add_argument("--scale", type=float, default=1.)
    parser.add_argument("--token-file", type=Path, help="Persist the browser token across task restarts")
    parser.add_argument("--task-state-file", type=Path, help="Write selected task and exit 75 for a supervisor restart")
    parser.add_argument("--device-state-file", type=Path,
                        help="Persist cpu/cuda device selection across supervised task restarts")
    parser.add_argument("--scene-state-file", type=Path,
                        help="Persist the random-scene toggle and selected official layout seed")
    parser.add_argument("--smoke-steps", type=int, default=0,
                        help="Explicit synthetic input, finite simulator integration check; never human data")
    parser.add_argument("--inspect", type=Path, help="Validate saved HDF5; does not start Isaac Sim")
    parser.add_argument("--replay", type=Path, help="Replay recorded commands and compare measured joints")
    parser.add_argument("--progress-file", type=Path,
                        help="Atomic main-loop progress read by the process-out watchdog")
    parser.add_argument("--epoch-file", type=Path,
                        help="Persistent environment generation counter")
    parser.add_argument("--input-only", action="store_true",
                        help="Run WebXR/USB input diagnostics without Isaac, IK, cameras or recording")
    parser.add_argument("--adb-serial",
                        help="Explicit adb serial used for input-only USB reverse")
    known, _ = parser.parse_known_args()
    if known.input_only:
        args = parser.parse_args()
        return run_input_only(args)
    if known.inspect:
        from recording import inspect_episode
        print(json.dumps(inspect_episode(known.inspect), ensure_ascii=False, indent=2))
        return
    if known.robodojo_root is None:
        parser.error("--robodojo-root is required unless --input-only is used")
    root = known.robodojo_root.resolve()
    if not (root / "env/environment/task_env.py").is_file():
        parser.error("--robodojo-root must point to the RoboDojo checkout")
    # Pin the project utils namespace before Kit imports cv2.utils.
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(root / "third_party/curobo"))
    import utils.load_file  # noqa: F401
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        parser.error("--scale must be in (0, 1]")
    if args.max_wall_gap <= 0:
        parser.error("--max-wall-gap must be positive")
    args.output = args.output.resolve()
    tasks = available_tasks(root)
    missing_prompts = sorted(set(tasks) - set(PROMPTS_ZH))
    if missing_prompts:
        parser.error(f"Missing Chinese prompts for official tasks: {', '.join(missing_prompts)}")
    if args.task not in tasks:
        if args.task_state_file is None or not tasks:
            parser.error(f"Task is not supported by the dual-X5 VR collector: {args.task}")
        fallback = "stack_blocks" if "stack_blocks" in tasks else tasks[0]
        logging.warning("Task %s is not supported by the dual-X5 VR collector; using %s",
                        args.task, fallback)
        args.task = fallback
        args.task_state_file.parent.mkdir(parents=True, exist_ok=True)
        args.task_state_file.write_text(args.task + "\n")
    required_device = task_runtime_device(root, args.task)
    if str(args.device) != required_device:
        if args.device_state_file is None:
            parser.error(f"Task {args.task} requires --device {required_device}")
        args.device_state_file.parent.mkdir(parents=True, exist_ok=True)
        args.device_state_file.write_text(required_device + "\n")
        if args.task_state_file is not None:
            args.task_state_file.parent.mkdir(parents=True, exist_ok=True)
            args.task_state_file.write_text(args.task + "\n")
        return TASK_SWITCH_EXIT
    if args.replay:
        args.replay = args.replay.resolve()
    scene_state = load_scene_state(args.scene_state_file, args.seed)
    layout, args.seed, layout_count = select_task_layout(root, args.task, args.layout, scene_state)
    if not layout.is_file():
        parser.error(f"Layout missing: {layout}")
    if bool(args.cert) != bool(args.key):
        parser.error("Provide both --cert and --key")
    cert = args.cert.resolve() if args.cert else None
    key = args.key.resolve() if args.key else None
    from server import Bridge, Server
    progress_path = args.progress_file or Path(__file__).with_name("runtime_progress.json")
    progress = ProgressFile(progress_path, args.epoch_file)
    env_epoch, inherited_operation = progress.start_epoch(args.task)
    bridge = Bridge(env_epoch, progress.path)
    catalog = task_catalog(root, tasks)
    bridge.catalog = catalog
    bridge.status.update({
        "phase": "starting", "task": args.task, "tasks": tasks,
        "task_info": catalog[args.task], "message": "正在加载任务和三相机画面…",
        "episode_counts": {}, "episodes": [],
    })
    token = persistent_token(args.token_file)
    server = Server(bridge, args.host, args.port, token, cert, key, args.output,
                    args.allow_http_lan)
    server.start()
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("robodojo_pico")
    log.setLevel(logging.INFO)
    log.info("Open %s://%s:%s/#token=%s", "https" if cert else "http", args.host, args.port, token)
    os.chdir(root)
    args.enable_cameras = True
    args.kit_args = (args.kit_args or "") + " --enable isaacsim.sensors.camera --enable isaacsim.replicator.behavior --enable isaacsim.asset.gen.conveyor"
    # The dispatcher timestamp check in RoboDojo.observation rejects stale frames.
    # Global waitIdle serializes the whole RTX pipeline and roughly halves throughput.
    args.kit_args += " --/app/updateOrder/checkForHydraRenderComplete=1000"
    app = None
    backend = None
    recorder = None
    retiring_recorder = None
    preview_encoder = None
    observer_encoder = None
    restart_task = None
    try:
        if args.smoke_steps:
            import faulthandler
            faulthandler.dump_traceback_later(180, repeat=True)
        app = AppLauncher(args).app
        from control import Controller
        from preview import PreviewEncoder
        from recording import (AsyncRecorder, RecorderBackpressure, delete_episode, episode_counts,
                               episode_path, inspect_episode, list_episodes, read_snapshot)
        from robodojo import RoboDojo
        preview_encoder = PreviewEncoder(bridge.publish_image)
        observer_encoder = PreviewEncoder(bridge.publish_observer)

        def create_runtime(task, selected_layout, seed):
            candidate = None
            try:
                candidate = RoboDojo(root, app, task, selected_layout, seed, args.device,
                                     phase_callback=progress.update)
                log.info("Reading first synchronized observation")
                progress.update("first_observation_enter", task=task)
                first_observation = candidate.observation()
                progress.update("first_observation_exit", task=task)
                if tuple(first_observation.get("vision", {})) != (
                        "cam_head", "cam_left_wrist", "cam_right_wrist"):
                    raise RuntimeError("Runtime is not ready: synchronized three-camera observation missing")
                if first_observation.get("render_stamp") is None:
                    raise RuntimeError("Runtime is not ready: camera render stamp missing")
                if not all(np.isfinite(value).all() for name, value in
                           first_observation["state"].items() if "poses" not in name):
                    raise RuntimeError("Runtime is not ready: robot state is nonfinite")
            except BaseException:
                if candidate is not None:
                    candidate.close()
                raise
            runtime_controller = Controller(args.scale)
            runtime_metadata = candidate.metadata()
            instruction = re.sub(r"\bof(?=[A-Za-z])", "of ",
                                 str(first_observation["instruction"]))
            runtime_task_info = dict(catalog[task])
            runtime_task_info["instruction"] = instruction
            runtime_task_info["prompt_zh"] = chinese_prompt(task, instruction)
            runtime_metadata["task_description"] = runtime_task_info
            runtime_metadata["embodiment"] = "arx_x5"
            runtime_metadata["max_wall_gap_s"] = args.max_wall_gap
            runtime_metadata["collection_purpose"] = ("integration_check" if args.smoke_steps
                                                       else "human_demonstration")
            runtime_metadata["translation_scale"] = args.scale
            runtime_metadata["random_scene_enabled"] = scene_state["enabled"]
            runtime_metadata.update(task_identity(task))
            runtime_metadata["collector_sha256"] = source_hashes()
            runtime_episodes = list_episodes(args.output, task=task)
            runtime_last_command = {key: value for key, value in first_observation["state"].items()
                                    if "poses" not in key}
            return (candidate, runtime_controller, first_observation, runtime_metadata,
                    runtime_task_info, runtime_episodes, runtime_last_command)

        try:
            (backend, controller, observation, metadata, task_info, episodes,
             last_command) = create_runtime(args.task, layout, args.seed)
            counts_by_task = episode_counts(args.output)
        except RuntimeError as error:
            if (scene_state["enabled"] and args.scene_state_file is not None
                    and "Scene layout is unstable" in str(error)):
                failed_seed = args.seed
                scene_state["seed"] = next_scene_seed(root, args.task, failed_seed)
                save_scene_state(args.scene_state_file, scene_state)
                restart_task = args.task
                log.warning("Official layout seed %s is unstable; retrying seed %s",
                            failed_seed, scene_state["seed"])
                return TASK_SWITCH_EXIT
            raise
        if args.replay:
            replay(backend, args.replay.resolve(), args.output)
            return
        message = "已就绪。先松开双手侧握键，再按住侧握键接管机械臂。"
        if inherited_operation:
            inherited_operation.update(status="complete", phase="ready_after_restart")
            progress.update("ready", operation=inherited_operation)
        started = time.monotonic()
        if args.smoke_steps:
            recorder = AsyncRecorder(args.output, metadata, observation)
        frames = 0
        applied_seq = -1
        homing = False
        homing_steps = 0
        home_error = 0.
        cycle_samples = deque(maxlen=45000)
        last_diagnostic_log = 0.
        previous_tick_start = None
        while app.is_running():
            tick_start = time.monotonic()
            runtime_reload = None
            period_ms = None if previous_tick_start is None else (tick_start - previous_tick_start) * 1000
            previous_tick_start = tick_start
            packet, commands, input_status = bridge.snapshot()
            if args.smoke_steps:
                packet = smoke_packet(frames)
            for command in commands:
                name = command["name"] if isinstance(command, dict) else command
                value = command.get("value") if isinstance(command, dict) else None
                operation_id = command.get("operation_id") if isinstance(command, dict) else None
                if operation_id:
                    bridge.update_operation(operation_id, "running", "validating")
                physical_record = name == "physical_record_toggle"
                if physical_record:
                    if input_status.get("control_mode") != "leader":
                        message = "实体输入模式已退出，录制命令未执行。"
                        continue
                    name = "record" if recorder is None else "save"
                if name == "record" and recorder is None:
                    if physical_record and (input_status.get("active_source") != "leader-can" or
                                            not all(bridge.leader.status()["ready"].values())):
                        message = "实体双主臂尚未同时就绪，无法开始录制。"
                    elif retiring_recorder is not None and retiring_recorder.thread.is_alive():
                        message = "上一次录制写入正在受控结束，请稍后再开始。"
                    else:
                        retiring_recorder = None
                        recording_metadata = dict(metadata, recording_start_tick=backend.ticks)
                        if physical_record:
                            recording_metadata["require_dual_controller_input"] = True
                        if input_status.get("active_source") == "desktop-debug":
                            recording_metadata["collection_purpose"] = "integration_check"
                        recorder = AsyncRecorder(args.output, recording_metadata, observation)
                        message = "正在录制普通示范。"
                elif name == "record_recovery" and recorder is None:
                    if input_status.get("active_source") == "leader-can":
                        message = "实体模式请用键盘空格开始普通录制。"
                    elif retiring_recorder is not None and retiring_recorder.thread.is_alive():
                        message = "上一次录制写入正在受控结束，请稍后再开始。"
                    else:
                        retiring_recorder = None
                        recovery_metadata = dict(metadata, recording_start_tick=backend.ticks,
                                                 collection_purpose="failure_recovery",
                                                 recovery_mode="live_failure_takeover",
                                                 recovery_source="current_live_state")
                        if input_status.get("active_source") == "desktop-debug":
                            recovery_metadata["collection_purpose"] = "integration_check"
                        recorder = AsyncRecorder(args.output, recovery_metadata, observation)
                        message = "正在从当前错误状态录制恢复轨迹。"
                elif name == "save" and recorder is not None:
                    if recorder.frames < 2:
                        message = "有效动作少于两帧，请稍后再保存。"
                        continue
                    progress.update("record_finish_enter", sim_tick=backend.ticks)
                    saved = recorder.finish(backend.success(), "operator_save", safe_snapshot(backend, log))
                    recorder = None
                    progress.update("record_finish_exit", sim_tick=backend.ticks)
                    if "rejected" in saved.relative_to(args.output).parts:
                        message = f"质量检查未通过，已隔离：{saved.name}"
                    else:
                        message = f"已验证并保存：{saved.name}"
                    log.info(message)
                    controller.stop("recording_saved")
                    episodes = list_episodes(args.output, task=args.task)
                    counts_by_task = episode_counts(args.output)
                elif name == "discard" and recorder is not None:
                    progress.update("record_discard_enter", sim_tick=backend.ticks)
                    recorder.discard()
                    recorder = None
                    progress.update("record_discard_exit", sim_tick=backend.ticks)
                    controller.stop("recording_discarded")
                    message = "已丢弃当前未保存的录制。"
                elif name == "delete_episode" and recorder is None:
                    try:
                        deleted = delete_episode(args.output, value, task=args.task)
                        episodes = list_episodes(args.output, task=args.task)
                        counts_by_task = episode_counts(args.output)
                        message = f"已删除：{deleted.name}"
                    except (ValueError, FileNotFoundError) as error:
                        log.warning("Delete refused: %s", error)
                        message = "无法删除所选轨迹，请确认轨迹仍然存在。"
                elif name == "takeover" and recorder is None:
                    try:
                        source = episode_path(args.output, value, task=args.task)
                        candidates = {item["name"]: item for item in
                                      list_episodes(args.output, task=args.task)}
                        info = candidates.get(source.name)
                        if info is None or info["success"]:
                            raise ValueError("只能接手当前任务中的失败轨迹")
                        snapshot = read_snapshot(source)
                        report = backend.restore_snapshot(snapshot)
                        observation = backend.observation()
                        controller.stop("recovery_takeover")
                        recovery_metadata = dict(metadata, recording_start_tick=backend.ticks,
                                                 collection_purpose="failure_recovery",
                                                 recovery_mode="validated_snapshot_takeover",
                                                 recovery_source_episode=source.name,
                                                 restore_validation=report)
                        if input_status.get("active_source") == "desktop-debug":
                            recovery_metadata["collection_purpose"] = "integration_check"
                        recorder = AsyncRecorder(args.output, recovery_metadata, observation)
                        message = f"已从失败末状态接手，正在录制恢复轨迹：{source.name}"
                    except Exception as error:
                        log.exception("Recovery takeover failed")
                        controller.stop("recovery_failed")
                        backend.reset()
                        observation = backend.observation()
                        message = "无法从末状态接手；状态快照不完整或还原校验未通过。"
                elif name == "select_task":
                    if recorder is not None:
                        message = "正在录制，切换任务前请先保存或丢弃。"
                        bridge.update_operation(operation_id, "failed", "refused_recording", message)
                    elif value not in tasks:
                        message = f"未知任务：{value}"
                        bridge.update_operation(operation_id, "failed", "validation", message)
                    elif args.task_state_file is None or args.device_state_file is None:
                        message = "当前启动方式未启用任务切换。"
                        bridge.update_operation(operation_id, "failed", "validation", message)
                    elif value == args.task:
                        message = f"已经是当前任务：{value}"
                        bridge.update_operation(operation_id, "complete", "already_current")
                    else:
                        controller.stop("lifecycle_restart", paused=True)
                        target_seed = (next_scene_seed(root, value, scene_state["seed"])
                                       if scene_state["enabled"] else 0)
                        target_layout, target_seed, target_count = select_task_layout(
                            root, value, None,
                            {"enabled": scene_state["enabled"], "seed": target_seed})
                        target_device = task_runtime_device(root, value)
                        if target_device != str(args.device):
                            scene_state.update(seed=target_seed)
                            save_scene_state(args.scene_state_file, scene_state)
                            args.task_state_file.parent.mkdir(parents=True, exist_ok=True)
                            args.task_state_file.write_text(value + "\n")
                            args.device_state_file.parent.mkdir(parents=True, exist_ok=True)
                            args.device_state_file.write_text(target_device + "\n")
                            restart_task = value
                            message = (f"任务 {value} 需要 {target_device} 物理，"
                                       "正在受控重启 Isaac。")
                            operation = {"operation_id": operation_id, "name": name,
                                         "status": "running", "phase": "device_restart",
                                         "target_task": value, "target_device": target_device,
                                         "target_seed": target_seed}
                            progress.update("restart_requested", sim_tick=backend.ticks,
                                            restart_requested=True, operation=operation,
                                            task=value, target_device=target_device)
                            break
                        message = f"正在当前 Isaac 进程内切换任务：{value}"
                        operation = {"operation_id": operation_id, "name": name, "status": "running",
                                     "phase": "task_reload", "target_task": value,
                                     "target_seed": target_seed}
                        progress.update("reset_task_requested", sim_tick=backend.ticks,
                                        operation=operation, task=value)
                        runtime_reload = (value, target_layout, target_seed, target_count,
                                          scene_state["enabled"], operation, "task")
                        break
                elif name == "reset":
                    if recorder is not None:
                        message = "正在录制，重置场景前请先保存或丢弃。"
                        bridge.update_operation(operation_id, "failed", "refused_recording", message)
                    else:
                        homing = False
                        controller.stop("environment_reset", paused=True)
                        operation = {"operation_id": operation_id, "name": name,
                                     "status": "running", "phase": "soft_reset"}
                        progress.update("reset_enter", sim_tick=backend.ticks, operation=operation)
                        bridge.publish_status(dict(bridge.status, phase="resetting",
                                                   message="正在重置当前固定场景。",
                                                   env_epoch=env_epoch, operation=operation))
                        backend.reset()
                        observation = backend.observation()
                        bridge.update_operation(operation_id, "complete", "ready")
                        operation.update(status="complete", phase="ready")
                        progress.update("reset_exit", sim_tick=backend.ticks, operation=operation)
                        message = "场景已重置，请松开双手侧握键后重新接管。"
                elif name == "next_scene":
                    if recorder is not None:
                        message = "正在录制，更换官方布局前请先保存或丢弃。"
                        bridge.update_operation(operation_id, "failed", "refused_recording", message)
                    elif args.scene_state_file is None or not scene_state["enabled"]:
                        message = "请先开启官方布局轮换，再更换布局。"
                        bridge.update_operation(operation_id, "failed", "validation", message)
                    else:
                        controller.stop("lifecycle_restart", paused=True)
                        target_seed = next_scene_seed(root, args.task, backend.seed)
                        target_layout, target_seed, target_count = select_task_layout(
                            root, args.task, None, {"enabled": True, "seed": target_seed})
                        message = f"正在进程内更换官方布局 seed={target_seed}"
                        operation = {"operation_id": operation_id, "name": name, "status": "running",
                                     "phase": "layout_reload", "target_task": args.task,
                                     "target_seed": target_seed}
                        progress.update("reset_layout_requested", sim_tick=backend.ticks,
                                        operation=operation)
                        runtime_reload = (args.task, target_layout, target_seed, target_count,
                                          True, operation, "layout")
                        break
                elif name == "set_random_scene":
                    if recorder is not None:
                        message = "正在录制，修改布局轮换开关前请先保存或丢弃。"
                        bridge.update_operation(operation_id, "failed", "refused_recording", message)
                    elif args.scene_state_file is None:
                        message = "当前启动方式未启用布局轮换开关。"
                        bridge.update_operation(operation_id, "failed", "validation", message)
                    elif value == scene_state["enabled"]:
                        message = f"官方布局轮换已经{'开启' if value else '关闭'}。"
                        bridge.update_operation(operation_id, "complete", "already_current")
                    else:
                        controller.stop("lifecycle_restart", paused=True)
                        target_seed = (next_scene_seed(root, args.task, backend.seed) if value else 0)
                        target_layout, target_seed, target_count = select_task_layout(
                            root, args.task, None, {"enabled": value, "seed": target_seed})
                        message = f"布局轮换已{'开启' if value else '关闭'}，正在进程内更换布局。"
                        operation = {"operation_id": operation_id, "name": name, "status": "running",
                                     "phase": "layout_reload", "target_task": args.task,
                                     "target_seed": target_seed}
                        progress.update("reset_layout_requested", sim_tick=backend.ticks,
                                        operation=operation)
                        runtime_reload = (args.task, target_layout, target_seed, target_count,
                                          value, operation, "layout")
                        break
                elif name == "home":
                    controller.stop("arm_homing", paused=True)
                    homing = True
                    homing_steps = 0
                    message = "双臂正在复位到初始关节位；复位过程会正常写入当前录制。"
                elif name == "pause":
                    controller.pause()
                    message = "已解除遥操作；如果正在录制，录制仍会继续。"
            if restart_task is not None:
                bridge.publish_status(dict(
                    bridge.status, phase="restarting", task=restart_task,
                    message=message, input_fresh=False, env_epoch=env_epoch,
                    teleop_allowed=False, release_required=True,
                    hold_reason="lifecycle_restart"))
                break
            if runtime_reload is not None:
                (target_task, target_layout, target_seed, target_count,
                 target_enabled, operation, reload_kind) = runtime_reload
                bridge.publish_status(dict(
                    bridge.status, phase="resetting", task=target_task,
                    message=message, input_fresh=False, env_epoch=env_epoch,
                    teleop_allowed=False, release_required=True,
                    hold_reason=f"{reload_kind}_reload", operation=operation))
                try:
                    progress.update(f"reset_{reload_kind}_close_enter", sim_tick=backend.ticks,
                                    operation=operation)
                    backend.close()
                    backend = None
                    progress.update(f"reset_{reload_kind}_rebuild_enter", sim_tick=0,
                                    operation=operation,
                                    target_seed=target_seed)
                    (backend, controller, observation, metadata, task_info, episodes,
                     last_command) = create_runtime(target_task, target_layout, target_seed)
                except BaseException:
                    log.exception("In-process %s reload failed; requesting supervisor restart",
                                  reload_kind)
                    scene_state.update(enabled=target_enabled, seed=target_seed)
                    save_scene_state(args.scene_state_file, scene_state)
                    args.task_state_file.parent.mkdir(parents=True, exist_ok=True)
                    args.task_state_file.write_text(target_task + "\n")
                    restart_task = target_task
                    operation.update(phase="restart_fallback")
                    progress.update("restart_requested", sim_tick=0, restart_requested=True,
                                    operation=operation, task=target_task)
                    break
                args.task = target_task
                layout = target_layout
                layout_count = target_count
                scene_state.update(enabled=target_enabled, seed=target_seed)
                save_scene_state(args.scene_state_file, scene_state)
                args.task_state_file.parent.mkdir(parents=True, exist_ok=True)
                args.task_state_file.write_text(target_task + "\n")
                metadata["random_scene_enabled"] = target_enabled
                args.seed = target_seed
                counts_by_task = episode_counts(args.output)
                bridge.update_operation(operation["operation_id"], "complete", "ready")
                operation.update(status="complete", phase="ready")
                progress.update("ready", sim_tick=backend.ticks, operation=operation,
                                restart_requested=False, target_seed=target_seed)
                noun = "任务" if reload_kind == "task" else "布局"
                message = (f"已在当前 Isaac 进程内切换到{noun} {target_task} / {target_layout.name}；"
                           "请松开双手侧握键后重新接管。")
                frames = 0
                started = time.monotonic()
                previous_tick_start = None
                continue
            control_start = time.monotonic()
            transient_gap = bool(input_status["connected"] and not input_status["timed_out"])
            if packet is not None:
                bridge.mark_applied(packet)
                input_status["applied_seq"] = packet["seq"]
                applied_seq = packet["seq"]
            task_motion_waiting = args.task in RECORDING_GATED_TASKS and recorder is None
            task_motion_enabled = not task_motion_waiting
            backend.set_task_motion_enabled(task_motion_enabled)
            if homing:
                last_command, home_complete, home_error = backend.home_step(
                    task_motion_enabled=task_motion_enabled)
                homing_steps += 1
                if home_complete:
                    homing = False
                    controller.stop("arm_home_complete", paused=True)
                    message = "双臂复位完成；请先松开双手侧握键，再重新接管。"
                elif homing_steps >= 125:
                    homing = False
                    controller.stop("arm_home_timeout", paused=True)
                    message = f"双臂复位超时，最大关节误差 {home_error:.3f} rad；请检查碰撞后重试。"
            else:
                if (input_status.get("control_mode") == "leader" and packet is not None and
                        all(bridge.leader.status()["ready"].values()) and
                        (controller.teleop_paused or controller.require_release)):
                    controller.reanchor()
                targets, grippers = controller.targets(
                    packet, backend.poses(), transient_gap=transient_gap)
                last_command = backend.step(
                    targets, grippers, task_motion_enabled=task_motion_enabled)
            control_ms = (time.monotonic() - control_start) * 1000
            next_observation = backend.observation()
            record_enqueue_ms = 0.
            if recorder is not None:
                record_start = time.monotonic()
                try:
                    recorder.append(observation, next_observation, last_command, packet,
                                    (backend.ticks - 1 - recorder.recorder.metadata.get(
                                        "recording_start_tick", 0)) * .04, tick_start)
                except RecorderBackpressure as error:
                    log.error("Recording stopped by explicit backpressure: %s", error)
                    retiring_recorder, recorder = recorder, None
                    retiring_recorder.close_incomplete_async()
                    controller.stop("recording_backpressure", paused=True)
                    message = "录制写入跟不上，已停止录制并保留 incomplete 文件；请检查磁盘。"
                record_enqueue_ms = (time.monotonic() - record_start) * 1000
            observation = next_observation
            frames += 1
            recorder_status = recorder.diagnostics() if recorder is not None else {
                "queue_depth": 0, "queue_capacity": 16, "backpressure": False,
                "backpressure_count": 0, "writer_alive": False, "writer_error": None}
            status = {"phase": "recording" if recorder else "ready", "task": args.task,
                      "frames": recorder.frames if recorder else 0, "message": message,
                      "env_epoch": env_epoch, "sim_tick": backend.ticks,
                      "input_fresh": packet is not None, "input": input_status,
                      "arm_homing": homing, "home_error": round(home_error, 4),
                      **controller.diagnostics(),
                      "ik_failures": backend.ik_failures, "simulation_seconds": backend.ticks * .04,
                      "physics_hz": round(1 / backend.env.dt),
                      "wall_hz": round(frames / max(time.monotonic() - started, .001), 2),
                      "tasks": tasks, "episodes": episodes, "episode_counts": counts_by_task,
                      "task_info": task_info,
                      **task_identity(args.task),
                      "layout_name": backend.layout_path.name,
                      "layout_sha256": metadata["layout_sha256"],
                      "scene_seed": backend.seed, "random_scene": scene_state["enabled"],
                      "layout_count": layout_count,
                      "scene_objects": scene_objects(backend.layout), "control_scale": args.scale,
                      "task_motion": ("waiting_for_recording" if task_motion_waiting else
                                      "recording_active" if args.task in RECORDING_GATED_TASKS else
                                      "not_gated"),
                      "conveyor_surface_enabled": (backend.conveyor_surface.GetSurfaceVelocityEnabledAttr().Get()
                                                   if backend.conveyor_surface is not None else None),
                      "control_ms": round(control_ms, 1),
                      "record_enqueue_ms": round(record_enqueue_ms, 1),
                      "recording_writer": recorder_status,
                      "ik_status": backend.last_ik_status,
                      "support_motion_steps": backend.support_motion_steps,
                      "lifecycle": dict(progress.state),
                      "observation_profile": backend.last_observation_profile,
                      "step_profile": backend.last_step_profile}
            preview_start = time.monotonic()
            preview_encoder.submit(preview_mosaic(observation["vision"]))
            observer_encoder.submit(np.ascontiguousarray(backend.top_frame[::2, ::2]))
            status["preview_enqueue_ms"] = round((time.monotonic() - preview_start) * 1000, 1)
            status["preview"] = preview_encoder.diagnostics()
            status["observer_preview"] = observer_encoder.diagnostics()
            compute_ms = (time.monotonic() - tick_start) * 1000
            cycle_ms = period_ms if period_ms is not None else compute_ms
            cycle_samples.append(cycle_ms)
            values = np.asarray(cycle_samples)
            status["cycle_ms"] = {
                "current": round(cycle_ms, 1), "compute": round(compute_ms, 1),
                "p50": round(float(np.percentile(values, 50)), 1),
                "p95": round(float(np.percentile(values, 95)), 1),
                "p99": round(float(np.percentile(values, 99)), 1),
                "max": round(float(np.max(values)), 1), "samples": len(values),
            }
            status["published_at_ms"] = int(time.time() * 1000)
            bridge.publish_status(status)
            progress.update(status["phase"], sim_tick=backend.ticks, task=args.task,
                            operation=progress.state.get("operation"))
            if time.monotonic() - last_diagnostic_log >= 2:
                log.info("runtime %s", json.dumps({
                    "tick": backend.ticks, "cycle_ms": status["cycle_ms"],
                    "input": input_status, "control": controller.diagnostics(),
                    "step": backend.last_step_profile, "observation": backend.last_observation_profile,
                    "writer": recorder_status, "preview": status["preview"],
                }, ensure_ascii=False))
                last_diagnostic_log = time.monotonic()
            if args.smoke_steps and frames >= args.smoke_steps:
                saved = recorder.finish(backend.success(), "synthetic_smoke_completed",
                                        safe_snapshot(backend, log))
                recorder = None
                report = inspect_episode(saved)
                report["runtime"] = status
                backend.reset()
                backend.observation()
                report["render_products_after_reset"] = len(backend.env.capture_manager.tiled_cameras)
                if report["render_products_after_reset"] != 3:
                    raise RuntimeError("Soft reset changed the number of camera render products")
                report["soft_reset_verified"] = True
                args.output.mkdir(parents=True, exist_ok=True)
                (args.output / "smoke_report.json").write_text(json.dumps(report, indent=2))
                log.info("SIMULATION_SMOKE_COMPLETE %s", saved)
                break
            time.sleep(max(0., .04 - (time.monotonic() - tick_start)))
    except BaseException:
        progress.update("failed", error="collector_exception", sim_tick=0 if backend is None else backend.ticks)
        log.exception("Collector stopped")
        raise
    finally:
        if args.smoke_steps:
            faulthandler.cancel_dump_traceback_later()
        if recorder is not None:
            progress.update("shutdown_recording", sim_tick=0 if backend is None else backend.ticks)
            recorder.close_incomplete()
        if app is not None:
            from isaaclab.sim import SimulationContext
            context = SimulationContext.instance()
            if context is not None:
                context._disable_app_control_on_stop_handle = True
        if backend is not None:
            progress.update("shutdown_backend_enter", sim_tick=backend.ticks)
            backend.close()
            progress.update("shutdown_backend_exit", sim_tick=backend.ticks)
        if app is not None:
            progress.update("shutdown_app_enter")
            app.close(wait_for_replicator=False)
            progress.update("shutdown_app_exit")
        if preview_encoder is not None:
            preview_encoder.close()
        if observer_encoder is not None:
            observer_encoder.close()
        server.close()
    if restart_task is not None:
        return TASK_SWITCH_EXIT
    return 0


def safe_snapshot(backend, log):
    try:
        return backend.state_snapshot()
    except Exception as error:
        log.exception("State snapshot failed; episode remains video-only")
        return {"recoverable": False, "reason": f"snapshot failed: {error}",
                "robots": {}, "objects": {}}


def source_hashes():
    import hashlib
    root = Path(__file__).parent
    files = list(root.glob("*.py")) + list((root / "web").glob("*"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def smoke_packet(frame):
    # Explicit test-only controller poses; exercise release, clutch and 1 cm lift.
    hands = {}
    for side in ("left", "right"):
        lift = min(max(frame - 3, 0), 10) * .001
        hands[side] = {"pose": [0, lift, 0, 0, 0, 0, 1],
                       "trigger": .1 if frame > 12 else 0., "squeeze": 1. if frame > 1 else 0.}
    return {"type": "pose", "seq": frame, "hands": hands}


def replay(backend, path, output):
    """Same-scene command replay check; does not claim exact object/contact determinism."""
    import h5py
    import numpy as np
    with h5py.File(path, "r") as f:
        meta = json.loads(f["metadata/json"][()].decode())
        if meta["layout_sha256"] != backend.metadata()["layout_sha256"] or meta["task"] != backend.task:
            raise ValueError("Replay requires the recorded task and layout")
        if not f.attrs["complete"]:
            raise ValueError("Cannot replay an incomplete episode")
        for side in backend.robots:
            initial_error = np.max(np.abs(backend.joints(side) - f[f"state/{side}_arm_joint_states"][0]))
            if initial_error > .05:
                raise ValueError("Recording started away from reset pose; reset-based command replay is not applicable")
        errors = []
        for i in range(len(f["action_valid"])):
            for side in backend.robots:
                backend.hold_joints[side] = f[f"command/{side}_arm_joint_states"][i]
                backend.hold_grippers[side] = float(f[f"command/{side}_ee_joint_states"][i, 0])
            backend.step({}, {})
            for side in backend.robots:
                errors.append(float(np.max(np.abs(backend.joints(side) - f[f"action/{side}_arm_joint_states"][i]))))
        report = {"episode": str(path), "steps": len(f["action_valid"]),
                  "max_joint_error_rad": max(errors), "scope": "joint_command_replay",
                  "object_contact_replay_verified": False}
        output.mkdir(parents=True, exist_ok=True)
        (output / "replay_report.json").write_text(json.dumps(report, indent=2))
        if report["max_joint_error_rad"] > .05:
            raise RuntimeError(f"Replay joint error exceeds 0.05 rad: {report}")
        logging.getLogger("robodojo_pico").info("REPLAY_COMPLETE %s", report)


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception:
        exit_code = 1
    raise SystemExit(exit_code)
