"""Receive-only Piper leader arm to relative RoboDojo pose input."""
import math
import select
import socket
import struct
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from control import XR_TO_WORLD, validate_packet


FRAME = struct.Struct("=IB3x8s")
JOINT_IDS = (0x155, 0x156, 0x157)
# Piper base: x forward, y left, z up. Dojo world: y forward, -x left, z up.
PIPER_TO_WORLD = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])


def leader_pose(joints_mdeg):
    """Piper SDK DH model, converted to a virtual controller pose (metres/xyzw)."""
    a = (0., 0., 285.03, -21.98, 0., 0.)
    alpha = (0., -math.pi / 2, 0., math.pi / 2, -math.pi / 2, math.pi / 2)
    offset = (0., math.radians(-172.22), math.radians(-102.78), 0., 0., 0.)
    d = (123., 0., 0., 250.75, 0., 91.)
    transform = np.eye(4)
    for index, raw in enumerate(joints_mdeg):
        angle = math.radians(raw * .001) + offset[index]
        ca, sa, ct, st = math.cos(alpha[index]), math.sin(alpha[index]), math.cos(angle), math.sin(angle)
        link = np.array([[ct, -st, 0., a[index]],
                         [st * ca, ct * ca, -sa, -sa * d[index]],
                         [st * sa, ct * sa, ca, ca * d[index]],
                         [0., 0., 0., 1.]])
        transform = transform @ link
    basis = XR_TO_WORLD.T @ PIPER_TO_WORLD
    position = basis @ transform[:3, 3] * .001
    quaternion = Rotation.from_matrix(basis @ transform[:3, :3]).as_quat()
    return np.r_[position, quaternion].tolist()


class LeaderCan:
    def __init__(self, bridge, interfaces=None):
        self.bridge = bridge
        self.interfaces = interfaces or {"left": "can0", "right": "can1"}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.state = "stopped"
        self.error = None
        self.frames = 0
        self.last_joint_at = {side: None for side in self.interfaces}
        self.ready = {side: False for side in self.interfaces}
        self.sensor_fresh = {side: False for side in self.interfaces}

    def status(self):
        with self.lock:
            ages = {side: None if stamp is None else round((time.monotonic() - stamp) * 1000)
                    for side, stamp in self.last_joint_at.items()}
            return {"state": self.state, "error": self.error, "frames": self.frames,
                    "joint_age_ms": ages, "ready": dict(self.ready),
                    "sensor_fresh": dict(self.sensor_fresh),
                    "interfaces": dict(self.interfaces)}

    def start(self):
        with self.bridge.lock:
            if self.bridge.status.get("phase") == "recording":
                return False, "请先结束当前录制"
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                return True, None
            sockets = {}
            try:
                for side, interface in self.interfaces.items():
                    can = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                    sockets[side] = can
                    filters = b"".join(struct.pack("=II", can_id, 0x7ff)
                                       for can_id in (*JOINT_IDS, 0x159))
                    can.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
                    can.bind((interface,))
            except OSError as exc:
                for can in sockets.values():
                    can.close()
                self.state, self.error = "error", str(exc)
                return False, str(exc)
            self.stop_event.clear()
            self.state, self.error = "waiting_for_joints", None
            self.frames = 0
            self.last_joint_at = {side: None for side in self.interfaces}
            self.ready = {side: False for side in self.interfaces}
            self.sensor_fresh = {side: False for side in self.interfaces}
            self.thread = threading.Thread(target=self._run, args=(sockets,), daemon=True)
            self.thread.start()
            return True, None

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1)

    def _run(self, sockets):
        fragments = {side: {} for side in sockets}
        gripper_um = {side: None for side in sockets}
        seq = 0
        release_until = {side: None for side in sockets}
        next_send = 0.
        epoch = self.bridge.env_epoch
        try:
            self.bridge.register_input(self, "leader-can", takeover=True)
            while not self.stop_event.is_set():
                if self.bridge.env_epoch != epoch:
                    break  # A new scene needs an explicit new reference/start.
                readable, _, _ = select.select(list(sockets.values()), [], [], .04)
                for can in readable:
                    side = next(side for side, candidate in sockets.items() if candidate is can)
                    raw = can.recv(FRAME.size)
                    if len(raw) == FRAME.size:
                        can_id, length, data = FRAME.unpack(raw)
                        if not can_id & 0xe0000000 and length == 8:
                            now = time.monotonic()
                            if can_id in JOINT_IDS:
                                values = struct.unpack(">ii", data)
                                if all(abs(value) <= 220000 for value in values):
                                    fragments[side][can_id] = (values, now)
                                    with self.lock:
                                        self.last_joint_at[side] = now
                            elif can_id == 0x159:
                                value = struct.unpack(">i", data[:4])[0]
                                if 0 <= value <= 80000:
                                    gripper_um[side] = value
                now = time.monotonic()
                ready = {side: len(parts) == 3 for side, parts in fragments.items()}
                fresh = {side: ready[side] and now - max(item[1] for item in parts.values()) < 10
                         for side, parts in fragments.items()}
                with self.lock:
                    self.ready = ready
                    self.sensor_fresh = fresh
                if not any(ready.values()):
                    with self.lock:
                        self.state = "waiting_for_joints"
                    continue
                if now < next_send:
                    continue
                next_send = now + .04
                hands = {}
                for side in sockets:
                    if ready[side]:
                        if release_until[side] is None:
                            release_until[side] = now + .5
                        joints = [value for can_id in JOINT_IDS for value in fragments[side][can_id][0]]
                        hands[side] = {"pose": leader_pose(joints),
                                       "trigger": 0. if gripper_um[side] is None else
                                       1. - min(gripper_um[side] / 70000., 1.),
                                       "squeeze": 1. if now >= release_until[side] else 0.}
                    else:
                        hands[side] = {"pose": [0, 0, 0, 0, 0, 0, 1],
                                       "trigger": 0., "squeeze": 0.}
                packet = validate_packet({"type": "pose", "seq": seq, "env_epoch": epoch,
                                          "hands": hands})
                packet["leader_fresh"] = fresh
                if not self.bridge.accept_packet(self, packet):
                    with self.lock:
                        self.state = "preempted"
                    break
                seq += 1
                with self.lock:
                    self.state, self.frames = "active", seq
        except (OSError, ValueError) as exc:
            with self.lock:
                self.state, self.error = "error", str(exc)
        finally:
            self.bridge.unregister_input(self)
            for can in sockets.values():
                can.close()
            with self.lock:
                if self.state not in ("error", "preempted"):
                    self.state = "stopped"
