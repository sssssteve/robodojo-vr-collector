import sys
import asyncio
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import aiohttp
from aiohttp import web

sys.path.insert(0, str(Path(__file__).parents[1]))
from control import Clutch, Controller, XR_TO_WORLD
from leader_can import FRAME, LeaderCan, PIPER_TO_WORLD, leader_pose
from server import Bridge, create_app


def test_zero_pose_matches_piper_sdk_fk():
    hand = leader_pose([0] * 6)
    xyz_piper = PIPER_TO_WORLD.T @ XR_TO_WORLD @ hand[:3]
    np.testing.assert_allclose(xyz_piper, [.056127512, 0, .213266268], atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(hand[3:]), 1, atol=1e-12)


def test_leader_takeover_blocks_training_record():
    bridge = Bridge()
    assert bridge.select_mode("leader")[0]
    bridge.register_input("leader", "leader-can", takeover=True)
    assert not bridge.submit_command("record")["accepted"]
    assert not bridge.submit_command("record_recovery")["accepted"]
    bridge.unregister_input("leader")
    assert bridge.select_mode("vr")[0]
    assert bridge.submit_command("record")["accepted"]


def test_relative_left_motion_uses_cartesian_mapping():
    before = np.asarray(leader_pose([0] * 6))
    after = np.asarray(leader_pose([1000, 0, 0, 0, 0, 0]))
    clutch = Clutch(scale=.6)
    robot = np.array([.1, .2, .8, 1., 0., 0., 0.])
    assert np.array_equal(clutch.update({"pose": before, "squeeze": 1}, robot), robot)
    target = clutch.update({"pose": after, "squeeze": 1}, robot)
    expected = .6 * XR_TO_WORLD @ (after[:3] - before[:3])
    np.testing.assert_allclose(target[:3] - robot[:3], expected, atol=1e-8)


def test_synthetic_can_frames_release_then_drive_both_sides():
    bridge = Bridge()
    assert bridge.select_mode("leader")[0]
    leader = LeaderCan(bridge)
    left_rx, left_tx = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    right_rx, right_tx = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    worker = threading.Thread(target=leader._run,
                              args=({"left": left_rx, "right": right_rx},), daemon=True)
    worker.start()
    try:
        for sender in (left_tx, right_tx):
            for can_id in (0x155, 0x156, 0x157):
                sender.send(FRAME.pack(can_id, 8, struct.pack(">ii", 0, 0)))
        release_packet = active_packet = None
        until = time.monotonic() + 2
        while time.monotonic() < until and active_packet is None:
            packet, _, _ = bridge.snapshot()
            if packet is not None:
                left, right = packet["hands"]["left"], packet["hands"]["right"]
                if left["squeeze"] == right["squeeze"] == 0:
                    release_packet = packet
                if left["squeeze"] == right["squeeze"] == 1:
                    active_packet = packet
            time.sleep(.015)
        assert release_packet is not None and active_packet is not None
        assert leader.status()["ready"] == {"left": True, "right": True}
        assert bridge.owner_source == "leader-can"
        controller = Controller()
        poses = {side: np.array([0., 0., 1., 1., 0., 0., 0.])
                 for side in ("left", "right")}
        assert controller.targets(release_packet, poses) == ({}, {})
        assert set(controller.targets(active_packet, poses)[0]) == {"left", "right"}
    finally:
        leader.stop_event.set()
        worker.join(timeout=1)
        left_tx.close()
        right_tx.close()
    assert not worker.is_alive()


def test_explicit_mode_prevents_cross_source_takeover():
    bridge = Bridge()
    vr, leader = object(), object()
    bridge.register_input(vr, "webxr")
    bridge.register_input(leader, "leader-can")
    packet = {"seq": 0, "env_epoch": 0, "hands": {"left": {}, "right": {}}}
    assert not bridge.accept_packet(vr, packet)
    assert bridge.select_mode("leader")[0]
    assert bridge.accept_packet(leader, packet)
    assert not bridge.accept_packet(vr, packet)
    assert not bridge.command_source_allowed(vr)
    assert bridge.command_source_allowed(leader)
    bridge.publish_status({"phase": "recording"})
    assert bridge.select_mode("vr") == (False, "请先保存或丢弃当前录制，再切换输入模式")


def test_mode_endpoint_requires_auth_and_reports_selection():
    async def scenario():
        bridge = Bridge()
        runner = web.AppRunner(create_app(bridge, "test-token"))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        try:
            async with aiohttp.ClientSession() as client:
                async with client.post(base + "/mode", json={"mode": "vr"}) as response:
                    assert response.status == 401
                async with client.post(base + "/mode?token=test-token", json={"mode": "vr"}) as response:
                    assert response.status == 202
                    assert (await response.json())["mode"] == "vr"
                async with client.get(base + "/status?token=test-token") as response:
                    assert (await response.json())["control_mode"] == "vr"
        finally:
            await runner.cleanup()
    asyncio.run(scenario())
