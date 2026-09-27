"""WebXR input validation and clutch-relative bimanual end-effector targets."""
from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation

SIDES = ("left", "right")
TRACKING_GRACE_S = .15
# WebXR: +x right, +y up, -z forward. RoboDojo: +x right, +y forward, +z up.
XR_TO_WORLD = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def validate_packet(packet):
    if not isinstance(packet, dict) or packet.get("type") != "pose":
        raise ValueError("Expected a pose packet")
    if type(packet.get("seq")) is not int or packet["seq"] < 0:
        raise ValueError("seq must be a nonnegative integer")
    result = {"type": "pose", "seq": packet["seq"], "hands": {}}
    for name in ("sample_time_ms", "client_buffered_amount"):
        value = packet.get(name)
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            result[name] = value
    dropped = packet.get("client_dropped_poses")
    if dropped is not None:
        if type(dropped) is not int or dropped < 0:
            raise ValueError("client_dropped_poses must be a nonnegative integer")
        result["client_dropped_poses"] = dropped
    epoch = packet.get("env_epoch")
    if epoch is not None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("env_epoch must be a nonnegative integer")
        result["env_epoch"] = epoch
    for side in SIDES:
        hand = packet.get("hands", {}).get(side)
        if hand is None:
            continue
        pose = np.asarray(hand.get("pose"), dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Hand pose must contain seven finite xyz/xyzw values")
        norm = np.linalg.norm(pose[3:])
        if not .5 < norm < 1.5:
            raise ValueError("Invalid quaternion")
        pose[3:] /= norm
        trigger = float(hand.get("trigger", 0))
        squeeze = float(hand.get("squeeze", 0))
        if not (0 <= trigger <= 1 and 0 <= squeeze <= 1):
            raise ValueError("Trigger and squeeze must be in [0, 1]")
        result["hands"][side] = {"pose": pose.tolist(), "trigger": trigger, "squeeze": squeeze}
    return result


@dataclass
class Clutch:
    scale: float = .6
    max_translation: float = .02
    max_rotation: float = .15
    reference: np.ndarray | None = None
    robot_reference: np.ndarray | None = None
    target: np.ndarray | None = None
    released: bool = True

    def release(self):
        self.reference = None
        self.robot_reference = None
        self.target = None
        self.released = True

    def update(self, hand, current_pose):
        """Return xyz/wxyz in the environment world frame; None means hold."""
        if hand is None or hand["squeeze"] < .5:
            self.release()
            return None
        pose = np.asarray(hand["pose"], dtype=float)
        current_pose = np.asarray(current_pose, dtype=float)
        if self.released:
            self.reference = pose.copy()
            self.robot_reference = current_pose.copy()
            self.target = current_pose.copy()
            self.released = False
            return self.target.copy()
        displacement = XR_TO_WORLD @ (pose[:3] - self.reference[:3]) * self.scale
        target_pos = self.robot_reference[:3] + displacement
        # Limit displacement from measured robot pose, not from unexecuted targets.
        step = target_pos - current_pose[:3]
        step *= min(1., self.max_translation / max(np.linalg.norm(step), 1e-12))
        delta = Rotation.from_quat(pose[3:]) * Rotation.from_quat(self.reference[3:]).inv()
        delta_world = Rotation.from_matrix(XR_TO_WORLD @ delta.as_matrix() @ XR_TO_WORLD.T)
        anchor_rot = Rotation.from_quat(self.robot_reference[[4, 5, 6, 3]])
        desired_rot = delta_world * anchor_rot
        measured_rot = Rotation.from_quat(current_pose[[4, 5, 6, 3]])
        rot_step = (desired_rot * measured_rot.inv()).as_rotvec()
        rot_step *= min(1., self.max_rotation / max(np.linalg.norm(rot_step), 1e-12))
        q = (Rotation.from_rotvec(rot_step) * measured_rot).as_quat()
        self.target = np.r_[current_pose[:3] + step, q[3], q[:3]]
        return self.target.copy()


class Controller:
    def __init__(self, scale=.6):
        self.clutches = {side: Clutch(scale=scale) for side in SIDES}
        self.require_release = True
        self.teleop_paused = False
        self.hold_reason = "safety_release_required"
        self.teleop_allowed = False
        self.missing_since = {side: None for side in SIDES}
        self.side_release_required = {side: False for side in SIDES}

    def stop(self, reason="safety_release_required", paused=False):
        for clutch in self.clutches.values():
            clutch.release()
        self.missing_since = {side: None for side in SIDES}
        self.side_release_required = {side: False for side in SIDES}
        self.require_release = True
        self.teleop_paused = bool(paused)
        self.teleop_allowed = False
        self.hold_reason = reason

    def pause(self):
        self.stop("teleop_paused", paused=True)

    def reanchor(self):
        for clutch in self.clutches.values():
            clutch.release()
        self.require_release = False
        self.teleop_paused = False

    def targets(self, packet, poses, transient_gap=False):
        if packet is None:
            if not transient_gap:
                self.stop("input_timeout")
            else:
                self.teleop_allowed = False
                self.hold_reason = "input_gap"
            return {}, {}
        hands = packet["hands"]
        if self.teleop_paused:
            if all(side in hands and hands[side]["squeeze"] < .5 for side in SIDES):
                self.teleop_paused = False
                self.require_release = False
            self.teleop_allowed = False
            self.hold_reason = "teleop_paused" if self.teleop_paused else "safety_release_required"
            return {}, {}
        # A fresh connection/recenter/reset must see BOTH grips released first.
        if self.require_release:
            if all(side in hands and hands[side]["squeeze"] < .5 for side in SIDES):
                self.require_release = False
            self.teleop_allowed = False
            self.hold_reason = "safety_release_required"
            return {}, {}
        targets, grippers = {}, {}
        now = time.monotonic()
        for side in SIDES:
            hand = hands.get(side)
            if hand is None:
                if self.missing_since[side] is None:
                    self.missing_since[side] = now
                if now - self.missing_since[side] >= TRACKING_GRACE_S:
                    self.clutches[side].release()
                    self.side_release_required[side] = True
                continue
            if self.missing_since[side] is not None:
                if now - self.missing_since[side] >= TRACKING_GRACE_S:
                    self.clutches[side].release()
                    self.side_release_required[side] = True
                self.missing_since[side] = None
            if self.side_release_required[side]:
                if hand["squeeze"] < .5:
                    self.side_release_required[side] = False
                continue
            target = self.clutches[side].update(hand, poses[side])
            if target is not None:
                targets[side] = target
                grippers[side] = 1. - hand["trigger"]
        self.teleop_allowed = bool(targets)
        self.hold_reason = ("active" if targets else
                            "tracking_release_required" if any(self.side_release_required.values()) else
                            "tracking_gap" if any(side not in hands for side in SIDES) else
                            "grips_released")
        return targets, grippers

    def diagnostics(self):
        return {
            "teleop_allowed": self.teleop_allowed,
            "teleop_paused": self.teleop_paused,
            "simulation_paused": False,
            "release_required": self.require_release,
            "side_release_required": dict(self.side_release_required),
            "hold_reason": self.hold_reason,
        }
