import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import aiohttp
from aiohttp import web
import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parents[1]))
from control import Clutch, Controller, validate_packet
from export_official_hdf5 import export_episode
from preview import PreviewEncoder
from recording import (AsyncRecorder, CAMERAS, Recorder, RecorderBackpressure,
                       delete_episode, episode_counts, inspect_episode, list_episodes,
                       read_snapshot)
from robodojo import RoboDojo
from server import Bridge, create_app, parse_command
from run import (CONVEYOR_TASKS, RECORDING_GATED_TASKS,
                 SUPPORT_MOTION_GATED_TASKS, available_tasks, load_scene_state, preview_mosaic,
                 select_task_layout, task_catalog,
                 task_identity, task_runtime_device)
from prompts_zh import PROMPTS_ZH, chinese_prompt
from watchdog import ProgressFile, read_progress, supervise


def packet(squeeze=0., y=0., trigger=0., seq=0):
    return {"type": "pose", "seq": seq, "env_epoch": 0, "hands": {
        side: {"pose": [0., y, 0., 0., 0., 0., 1.], "trigger": trigger, "squeeze": squeeze}
        for side in ("left", "right")}}


def test_clutch_motion_rotation_and_reengagement():
    clutch = Clutch(scale=1., max_translation=.05)
    origin = np.array([.2, -.1, .9, 1., 0., 0., 0.])
    hand = packet(1.)["hands"]["left"]
    np.testing.assert_allclose(clutch.update(hand, origin), origin)
    hand["pose"][1] = .01
    out = clutch.update(hand, origin)
    np.testing.assert_allclose(out[:3], origin[:3] + [0, 0, .01])
    # Positive WebXR yaw maps to positive world yaw (Y-up -> Z-up).
    hand["pose"][3:] = Rotation.from_rotvec([0, .1, 0]).as_quat().tolist()
    rotated = clutch.update(hand, origin)
    np.testing.assert_allclose(Rotation.from_quat(rotated[[4, 5, 6, 3]]).as_rotvec(), [0, 0, .1], atol=1e-8)
    hand["squeeze"] = 0
    assert clutch.update(hand, origin) is None
    hand["pose"][1] = 5
    hand["squeeze"] = 1
    np.testing.assert_allclose(clutch.update(hand, out), out)


def test_limits_use_measured_state_not_unexecuted_target():
    clutch = Clutch(scale=1.)
    origin = np.array([0, 0, 1, 1, 0, 0, 0.])
    hand = packet(1.)["hands"]["left"]
    clutch.update(hand, origin)
    hand["pose"][1] = 10
    for _ in range(100):
        out = clutch.update(hand, origin)
        assert np.linalg.norm(out[:3] - origin[:3]) <= .02000001


def test_missing_connection_requires_release_before_motion():
    controller = Controller()
    poses = {s: [0, 0, 1, 1, 0, 0, 0] for s in ("left", "right")}
    assert controller.targets(packet(1.), poses) == ({}, {})
    controller.targets(packet(0.), poses)
    assert len(controller.targets(packet(1.), poses)[0]) == 2
    assert controller.targets(None, poses) == ({}, {})
    assert controller.targets(packet(1.), poses) == ({}, {})
    controller.targets(packet(0.), poses)
    targets, grippers = controller.targets(packet(1., trigger=.7), poses)
    assert len(targets) == 2
    assert grippers["left"] == pytest.approx(.3)
    lost = packet(1.)
    del lost['hands']['right']
    targets, _ = controller.targets(lost, poses)
    assert set(targets) == {"left"}
    assert not controller.require_release


def test_single_hand_tracking_loss_has_grace_and_requires_side_release(monkeypatch):
    import control
    now = [0.]
    monkeypatch.setattr(control.time, "monotonic", lambda: now[0])
    controller = Controller()
    poses = {side: [0, 0, 1, 1, 0, 0, 0] for side in ("left", "right")}
    controller.targets(packet(0.), poses)
    assert set(controller.targets(packet(1.), poses)[0]) == {"left", "right"}
    lost = packet(1.)
    del lost["hands"]["right"]
    assert set(controller.targets(lost, poses)[0]) == {"left"}
    assert not controller.diagnostics()["side_release_required"]["right"]
    now[0] = .2
    assert set(controller.targets(lost, poses)[0]) == {"left"}
    assert controller.diagnostics()["side_release_required"]["right"]
    assert set(controller.targets(packet(1.), poses)[0]) == {"left"}
    controller.targets(packet(0.), poses)
    assert set(controller.targets(packet(1.), poses)[0]) == {"left", "right"}


def test_transient_input_gap_holds_without_requiring_grip_release():
    controller = Controller()
    poses = {s: [0, 0, 1, 1, 0, 0, 0] for s in ("left", "right")}
    controller.targets(packet(0.), poses)
    assert len(controller.targets(packet(1.), poses)[0]) == 2
    assert controller.targets(None, poses, transient_gap=True) == ({}, {})
    assert not controller.require_release
    assert len(controller.targets(packet(1., y=.01), poses)[0]) == 2


@pytest.mark.parametrize("field,value", [("pose", [0, 0, 0, 0, 0, 0, 0]),
                                         ("pose", [0, 0, 0, 0, 0, 0, float('nan')]),
                                         ("pose", [0, 0]), ("trigger", 2), ("squeeze", float('nan'))])
def test_invalid_controller_data_rejected(field, value):
    data = packet()
    data["hands"]["left"][field] = value
    with pytest.raises(ValueError):
        validate_packet(data)


def observation(value):
    return {"instruction": np.str_("Stack the blocks"), "render_stamp": float(value), "state": {
        "left_arm_joint_states": np.full(6, value), "right_arm_joint_states": np.full(6, -value),
        "left_ee_joint_states": np.array([1.]), "right_ee_joint_states": np.array([1.]),
        "left_ee_poses": np.array([value, 0, 0, 1, 0, 0, 0.]),
        "right_ee_poses": np.array([-value, 0, 0, 1, 0, 0, 0.])},
        "vision": {camera: {"color": np.full((480, 640, 3), value * 10, dtype=np.uint8),
                            "intrinsic_matrix": np.eye(3), "extrinsic_matrix": np.eye(4)} for camera in CAMERAS}}


def test_preview_mosaic_keeps_head_and_places_wrist_views_at_the_sides():
    obs = observation(1)
    image = preview_mosaic(obs["vision"])
    assert image.shape == (720, 1280, 3)
    np.testing.assert_array_equal(image[120:600, 320:960], obs["vision"]["cam_head"]["color"])
    np.testing.assert_array_equal(image[240:480, :320], obs["vision"]["cam_left_wrist"]["color"][::2, ::2])
    np.testing.assert_array_equal(image[240:480, 960:], obs["vision"]["cam_right_wrist"]["color"][::2, ::2])


def test_observer_stream_is_separate_and_cleared_on_scene_change():
    async def scenario():
        bridge = Bridge()
        runner = web.AppRunner(create_app(bridge, "test-token"))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(f"http://127.0.0.1:{port}/observer?token=test-token") as ws:
                    bridge.publish_observer(b"observer-jpeg")
                    assert (await ws.receive(timeout=2)).data == b"observer-jpeg"
                    assert bridge.image is None
                    bridge.reset_epoch(1)
                    assert bridge.observer_image is None
        finally:
            await runner.cleanup()
    asyncio.run(scenario())


def test_random_scene_state_selects_an_official_seeded_layout(tmp_path):
    layouts = tmp_path / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    layouts.mkdir(parents=True)
    for seed in (0, 2, 10):
        (layouts / f"example_{seed}.json").write_text("{}")
    state_file = tmp_path / "scene.json"
    state_file.write_text('{"enabled": true, "seed": 10}\n')
    state = load_scene_state(state_file)
    layout, seed, count = select_task_layout(tmp_path, "example", None, state)
    assert layout.name == "example_10.json" and seed == 10 and count == 3
    fixed, seed, _ = select_task_layout(tmp_path, "example", None, {"enabled": False, "seed": 10})
    assert fixed.name == "example_0.json" and seed == 0


def test_fluid_tasks_select_cpu_and_solid_tasks_select_cuda(tmp_path):
    layouts = tmp_path / "Assets/Eval_Layout/RoboDojo/arx_x5/0"
    layouts.mkdir(parents=True)
    (layouts / "solid_0.json").write_text('{"Rigid": {}}')
    (layouts / "liquid_0.json").write_text('{"Fluid": {"wine": [{}]}}')
    assert task_runtime_device(tmp_path, "solid") == "cuda:0"
    assert task_runtime_device(tmp_path, "liquid") == "cpu"


def test_random_task_variant_keeps_its_own_identity_and_data_directory(tmp_path):
    assert task_identity("stack_bowls") == {
        "task_family": "stack_bowls", "task_variant": "standard"}
    assert task_identity("stack_bowls_random") == {
        "task_family": "stack_bowls", "task_variant": "random"}
    standard = Recorder(tmp_path, {
        "task": "stack_bowls", "collection_purpose": "human_demonstration"}, observation(0))
    random_variant = Recorder(tmp_path, {
        "task": "stack_bowls_random", "collection_purpose": "human_demonstration"}, observation(0))
    assert standard.data_directory != random_variant.data_directory
    assert standard.data_directory.relative_to(tmp_path).parts[:2] == ("RoboDojo", "stack_bowls")
    assert random_variant.data_directory.relative_to(tmp_path).parts[:2] == (
        "RoboDojo", "stack_bowls_random")
    standard.discard()
    random_variant.discard()


def test_episode_roundtrip_and_alignment(tmp_path):
    obs = [observation(i) for i in range(3)]
    rec = Recorder(tmp_path, {"input_source": "test_fixture"}, obs[0])
    command = {"left_arm_joint_states": np.ones(6)}
    rec.append(obs[0], obs[1], command, packet(), 0., 100.)
    rec.append(obs[1], obs[2], command, packet(seq=1), .04, 100.05)
    saved = rec.finish(False, "operator_save")
    report = inspect_episode(saved)
    assert report["frames"] == 3 and not report["success"]
    assert "input_source" not in report["source"]
    with h5py.File(saved) as f:
        assert f.attrs["label_source"] == "robodojo_task_reward"
        np.testing.assert_array_equal(f["action_valid"][:], [True, True, False])
        np.testing.assert_array_equal(f['command/left_arm_joint_states'][0], 1)
        np.testing.assert_array_equal(f['action/left_arm_joint_states'][1], 2)
        official_parts = ("left_arm_joint_states", "left_ee_joint_states",
                          "right_arm_joint_states", "right_ee_joint_states")
        state = np.concatenate([f[f"state/{name}"][:] for name in official_parts], axis=1)
        action = np.concatenate([f[f"action/{name}"][:] for name in official_parts], axis=1)
        assert state.shape == action.shape == (3, 14)
        assert all(len(f[f"vision/{camera}/colors"]) == 3 for camera in CAMERAS)
        assert all(f[f"vision/{camera}/colors"].dtype.kind == "S" for camera in CAMERAS)
        assert all(f[f"state/{name}"].dtype == np.float64 for name in official_parts)
        assert f.attrs["quality_pass"]


def test_official_layout_reports_wall_gaps_without_rejecting_episode(tmp_path):
    metadata = {"task": "stack_blocks", "embodiment": "arx_x5",
                "input_source": "meta_quest_webxr", "max_wall_gap_s": .2}
    command = {"left_arm_joint_states": np.ones(6)}
    good = Recorder(tmp_path, metadata, observation(0))
    good.append(observation(0), observation(1), command, packet(), 0., 100.)
    good.append(observation(1), observation(2), command, packet(seq=1), .04, 100.05)
    good_path = good.finish(True, "operator_save")
    assert good_path.relative_to(tmp_path).parts[:4] == ("RoboDojo", "stack_blocks", "arx_x5", "data")
    assert good_path.name == "episode_0000000.hdf5"

    delayed = Recorder(tmp_path, metadata, observation(0))
    delayed.append(observation(0), observation(1), command, packet(), 0., 100.)
    delayed.append(observation(1), observation(2), command, packet(seq=1), .04, 100.5)
    delayed_path = delayed.finish(False, "operator_save")
    assert delayed_path.relative_to(tmp_path).parts[:4] == (
        "RoboDojo", "stack_blocks", "arx_x5", "data")
    with h5py.File(delayed_path) as file:
        assert file.attrs["quality_pass"]
        assert file.attrs["timing_wall_gap_count"] == 1
    assert [item["name"] for item in list_episodes(tmp_path, task="stack_blocks")] == [
        delayed_path.name, good_path.name]


def test_missing_controller_packet_is_reported_without_rejecting_episode(tmp_path):
    metadata = {"task": "stack_blocks", "embodiment": "arx_x5"}
    command = {"left_arm_joint_states": np.ones(6)}
    recorder = Recorder(tmp_path, metadata, observation(0))
    recorder.append(observation(0), observation(1), command, packet(), 0., 100.)
    recorder.append(observation(1), observation(2), command, None, .04, 100.05)

    path = recorder.finish(True, "operator_save")

    assert path.relative_to(tmp_path).parts[:4] == (
        "RoboDojo", "stack_blocks", "arx_x5", "data")
    with h5py.File(path) as file:
        assert file.attrs["quality_pass"]
        assert file.attrs["controller_input_missing_transition_count"] == 1
        assert file.attrs["controller_input_valid_transition_count"] == 2


def test_physical_recording_rejects_missing_dual_controller_input(tmp_path):
    metadata = {"task": "stack_blocks", "embodiment": "arx_x5",
                "require_dual_controller_input": True}
    command = {"left_arm_joint_states": np.ones(6)}
    recorder = Recorder(tmp_path, metadata, observation(0))
    recorder.append(observation(0), observation(1), command, packet(), 0., 100.)
    recorder.append(observation(1), observation(2), command, None, .04, 100.05)

    path = recorder.finish(True, "operator_save")

    assert path.relative_to(tmp_path).parts[:2] == ("rejected", "stack_blocks")
    with h5py.File(path) as file:
        assert not file.attrs["quality_pass"]
        assert "missing controller input" in file.attrs["quality_reason"]


def test_recorder_skips_an_orphaned_partial_episode_id(tmp_path):
    staging = tmp_path / "staging/stack_blocks/arx_x5"
    staging.mkdir(parents=True)
    (staging / "episode_0000003.partial.hdf5").touch()

    recorder = Recorder(tmp_path, {"task": "stack_blocks"}, observation(0))

    assert recorder.episode_name == "episode_0000004.hdf5"
    recorder.discard()


def test_export_official_hdf5_uses_command_and_omits_extra_fields(tmp_path):
    names = ("left_arm_joint_states", "left_ee_joint_states",
             "right_arm_joint_states", "right_ee_joint_states")
    command = {name: np.full_like(observation(0)["state"][name], 9.) for name in names}
    rec = Recorder(tmp_path, {"task": "stack_blocks"}, observation(0))
    rec.append(observation(0), observation(1), command, packet(), 0., 100.)
    rec.append(observation(1), observation(2), command, packet(seq=1), .04, 100.04)
    source = rec.finish(True, "operator_save")
    destination = tmp_path / "official.hdf5"
    export_episode(source, destination)
    with h5py.File(destination) as file:
        assert set(file) == {"data_format_version", "instruction", "additional_info",
                             "state", "action", "vision"}
        assert set(file["action"]) == set(names)
        assert len(file["state/left_arm_joint_states"]) == 2
        assert len(file["vision/cam_head/colors"]) == 2
        assert np.all(file["action/left_arm_joint_states"][:] == 9.)


def test_export_official_hdf5_rejects_failed_demonstration(tmp_path):
    names = ("left_arm_joint_states", "left_ee_joint_states",
             "right_arm_joint_states", "right_ee_joint_states")
    command = {name: np.ones_like(observation(0)["state"][name]) for name in names}
    rec = Recorder(tmp_path, {"task": "stack_blocks"}, observation(0))
    rec.append(observation(0), observation(1), command, packet(), 0., 100.)
    rec.append(observation(1), observation(2), command, packet(seq=1), .04, 100.04)
    source = rec.finish(False, "operator_save")
    with pytest.raises(ValueError, match="positive demonstration"):
        export_episode(source, tmp_path / "not_exported.hdf5")


def test_debug_episode_is_kept_out_of_training_directory(tmp_path):
    metadata = {"task": "stack_blocks", "embodiment": "arx_x5",
                "collection_purpose": "integration_check"}
    command = {"left_arm_joint_states": np.ones(6)}
    rec = Recorder(tmp_path, metadata, observation(0))
    rec.append(observation(0), observation(1), command, packet(), 0., 100.)
    rec.append(observation(1), observation(2), command, packet(seq=1), .04, 100.04)
    path = rec.finish(False, "operator_save")
    assert path.relative_to(tmp_path).parts[:2] == ("validation", "stack_blocks")
    with h5py.File(path) as file:
        assert file.attrs["quality_pass"] and not file.attrs["training_eligible"]


def test_episode_counts_include_only_saved_collection_data(tmp_path):
    command = {"left_arm_joint_states": np.ones(6)}
    for task, purpose in (("stack_blocks", "human_demonstration"),
                          ("stack_blocks", "failure_recovery"),
                          ("build_tower", "integration_check")):
        rec = Recorder(tmp_path, {"task": task, "collection_purpose": purpose}, observation(0))
        rec.append(observation(0), observation(1), command, packet(), 0., 100.)
        rec.append(observation(1), observation(2), command, packet(seq=1), .04, 100.04)
        rec.finish(False, "operator_save")
    assert episode_counts(tmp_path) == {"stack_blocks": 2}


def test_async_recorder_preserves_frame_order(tmp_path):
    rec = AsyncRecorder(tmp_path, {"input_source": "async_test"}, observation(0))
    command = {"left_arm_joint_states": np.ones(6)}
    rec.append(observation(0), observation(1), command, packet(), 0., 100.)
    rec.append(observation(1), observation(2), command, packet(seq=1), .04, 100.04)
    path = rec.finish(False, "test")
    assert rec.frames == 2
    with h5py.File(path) as file:
        np.testing.assert_array_equal(file["state/left_arm_joint_states"][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(file["action_valid"][:], [True, True, False])


def test_missing_camera_fails_and_aborted_file_is_not_complete(tmp_path):
    obs = observation(0)
    rec = Recorder(tmp_path, {}, obs)
    del obs["vision"]["cam_left_wrist"]
    with pytest.raises(ValueError, match="all three"):
        rec.append(obs, obs, {}, None, 0., 0.)
    rec.close_incomplete()
    assert rec.path.name.endswith('.partial.hdf5')
    with h5py.File(rec.path) as f:
        assert not f.attrs['complete']


def test_recovery_snapshot_listing_and_exact_delete(tmp_path):
    rec = Recorder(tmp_path, {"task": "stack_blocks", "collection_purpose": "human_demonstration"},
                   observation(0))
    command = {"left_arm_joint_states": np.ones(6)}
    rec.append(observation(0), observation(1), command, packet(), 0., 100.)
    rec.append(observation(1), observation(2), command, packet(seq=1), .04, 100.04)
    snapshot = {"recoverable": True, "reason": "", "robots": {
        "left": {"joint_position": np.arange(8), "joint_velocity": np.zeros(8)}},
        "objects": {"env0_rigid_block": {"type": "rigid", "position": np.ones(3)}}}
    path = rec.finish(False, "operator_save", snapshot)
    restored = read_snapshot(path)
    assert restored["recoverable"] and restored["objects"]["env0_rigid_block"]["type"] == "rigid"
    episodes = list_episodes(tmp_path, task="stack_blocks")
    assert episodes[0]["name"] == path.name and episodes[0]["recoverable"] and not episodes[0]["success"]
    assert delete_episode(tmp_path, path.name) == path and not path.exists()
    with pytest.raises(ValueError):
        delete_episode(tmp_path, "../outside.hdf5")


def test_transport_auth_multiple_idle_clients_stale_input_and_independent_video():
    async def scenario():
        bridge = Bridge()
        bridge.select_mode("vr")
        app = create_app(bridge, "test-token")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        try:
            async with aiohttp.ClientSession() as client:
                async with client.get(base + '/status') as response:
                    assert response.status == 401
                async with client.get(base + '/') as response:
                    page = await response.text()
                    assert response.status == 200 and 'Pico' in page
                    assert 'id="debug"' in page and '桌面模拟 VR 输入' in page
                    assert 'id="spectator-link"' in page and '打开 PC 监看页面' in page
                    assert 'id="lifecycle-bar"' in page and '空间重建进度' in page
                async with client.get(base + '/client.js') as response:
                    assert response.status == 200 and 'javascript' in response.content_type
                    client_js = await response.text()
                    assert 'function connect()' in client_js
                    assert 'function desktopDebugFrame(time)' in client_js
                    assert 'if (!menuSyncSupported' in client_js
                    assert 'function renderLifecycleProgress' in client_js
                    assert '/spectator#${' in client_js
                    assert 'window.close()' not in client_js and "location.replace('about:blank')" not in client_js
                    assert "window.addEventListener('blur', releaseDesktopDebugControls)" in client_js
                async with client.get(base + '/spectator') as response:
                    spectator_page = await response.text()
                    assert response.status == 200 and '4090' in spectator_page
                    assert 'id="recording"' in spectator_page
                    assert 'id="menu"' in spectator_page and '当前任务已验收保存' in spectator_page
                    assert 'id="fullscreen-button"' in spectator_page
                    assert 'id="fullscreen-hud"' in spectator_page
                    assert 'id="quality-panel"' in spectator_page
                    assert 'id="top-observer"' in spectator_page
                    assert 'id="leader-start"' in spectator_page
                async with client.get(base + '/spectator.js') as response:
                    assert response.status == 200 and 'javascript' in response.content_type
                    spectator_js = await response.text()
                    assert '正在录制' in spectator_js and 'renderVrMenu(msg.vr_ui)' in spectator_js
                    assert 'requestFullscreen()' in spectator_js
                    assert 'fullscreenRecordingElement' in spectator_js
                    assert 'function renderQuality' in spectator_js
                    assert 'function connectTopObserver' in spectator_js
                async with client.ws_connect(base + '/input?token=test-token') as ws:
                    observer = await client.ws_connect(base + '/input?token=test-token')
                    assert len(bridge.clients) == 2 and bridge.owner is None
                    await observer.send_json({"type": "ping", "client_ms": 122})
                    assert json.loads((await observer.receive()).data) == {
                        "type": "pong", "client_ms": 122}
                    await ws.send_json({"type": "ping", "client_ms": 123})
                    assert json.loads((await ws.receive()).data) == {"type": "pong", "client_ms": 123}
                    await ws.send_json(packet(seq=2))
                    await asyncio.sleep(.03)
                    assert bridge.snapshot()[0]['seq'] == 2
                    assert bridge.snapshot()[0] is None  # latest pose is consumed once
                    await ws.send_json({"type": "ui_state", "open": True,
                                        "title": "RoboDojo 数采菜单",
                                        "items": ["开始普通录制", "保存当前录制"],
                                        "body": [], "selected": 1})
                    await asyncio.sleep(.03)
                    async with client.get(base + '/status?token=test-token') as response:
                        ui = (await response.json())["vr_ui"]
                        assert ui["open"] and ui["selected"] == 1
                    task_items = [f"task-{index}" for index in range(55)]
                    await ws.send_json({"type": "ui_state", "open": True,
                                        "title": "选择 RoboDojo 任务",
                                        "items": task_items, "body": [], "selected": 54})
                    await asyncio.sleep(.03)
                    async with client.get(base + '/status?token=test-token') as response:
                        ui = (await response.json())["vr_ui"]
                        assert ui["items"] == task_items and ui["selected"] == 54
                    await observer.send_json(packet(seq=99))
                    await asyncio.sleep(.03)
                    assert bridge.snapshot()[0] is None  # passive pages cannot override a fresh owner
                    await observer.send_json({"type": "ui_state", "open": True,
                                              "title": "旧页面菜单", "items": ["旧选项"],
                                              "body": [], "selected": 0})
                    await asyncio.sleep(.03)
                    async with client.get(base + '/status?token=test-token') as response:
                        assert (await response.json())["vr_ui"]["title"] == "选择 RoboDojo 任务"
                    await ws.send_json({"type": "command", "command": "select_task", "value": "stack_blocks"})
                    accepted = json.loads((await ws.receive()).data)
                    assert accepted["accepted"] and accepted["command"] == "select_task"
                    assert accepted["operation_id"] == "0-1"
                    await asyncio.sleep(.03)
                    queued = bridge.snapshot()[1]
                    assert queued == [{"name": "select_task", "value": "stack_blocks",
                                       "operation_id": "0-1"}]
                    await ws.send_json({"type": "command", "command": "set_random_scene", "value": True})
                    refused = json.loads((await ws.receive()).data)
                    assert not refused["accepted"] and refused["operation_id"] == "0-1"
                    bridge.update_operation("0-1", "complete", "ready")
                    async with client.post(base + '/command?token=test-token', json={"command": "reset"}) as response:
                        assert response.status == 202
                        reset = await response.json()
                        assert reset["operation_id"] == "0-2"
                    async with client.post(base + '/command?token=test-token', json={"command": "reset"}) as response:
                        assert response.status == 409
                    assert bridge.snapshot()[1] == [{"name": "reset", "operation_id": "0-2"}]
                    bridge.update_operation("0-2", "complete", "ready")
                    await ws.send_json(packet(seq=1))
                    await asyncio.sleep(.03)
                    assert bridge.snapshot()[0] is None
                    await ws.send_json(packet(seq=3))
                    await asyncio.sleep(.03)
                    bridge.received = time.monotonic() - .3
                    assert bridge.snapshot()[0]['seq'] == 3
                    await ws.send_json(packet(seq=4))
                    await asyncio.sleep(.03)
                    bridge.received = time.monotonic() - .8
                    pose, _, diagnostics = bridge.snapshot()
                    assert pose is None and diagnostics["timed_out"]
                    bridge.publish(b'jpeg-test', {'phase':'ready'})
                    async with client.ws_connect(base + '/video?token=test-token') as video:
                        video_status = json.loads((await video.receive()).data)
                        assert video_status['phase'] == 'ready'
                        assert video_status['vr_ui']['title'] == '选择 RoboDojo 任务'
                        assert (await video.receive()).data == b'jpeg-test'
                        # A hung simulator must not masquerade as fresh video.
                        with pytest.raises(asyncio.TimeoutError):
                            await asyncio.wait_for(video.receive(), .05)
                        bridge.publish(b'next-jpeg', {'phase':'ready'})
                        assert json.loads((await video.receive()).data)['phase'] == 'ready'
                        assert (await video.receive()).data == b'next-jpeg'
                    bad = packet(seq=3)
                    bad['hands']['left']['pose'][0] = float('nan')
                    await ws.send_json(bad)
                    assert 'error' in json.loads((await ws.receive()).data)
                    assert bridge.snapshot()[0] is None
                    await observer.close()
                    assert bridge.owner_session_id == 1 and bridge.input_connected
                await asyncio.sleep(.03)
                assert bridge.owner is None and not bridge.input_connected
                assert bridge.snapshot()[0] is None
        finally:
            await runner.cleanup()
    asyncio.run(scenario())


def test_portable_launcher_opens_authorized_quest_page():
    launcher = (Path(__file__).parents[1] / "start_desktop.sh").read_text()
    assert 'source "$PROJECT/config.env"' in launcher
    assert "ROBODOJO_ADB" in launcher and '"$ADB"' in launcher
    assert 'reverse "tcp:$PORT" "tcp:$PORT"' in launcher
    assert "com.oculus.vrshell" in launcher
    assert 'MONITOR_PAGE="http://127.0.0.1:$PORT/spectator#token=$TOKEN"' in launcher


def test_desktop_debug_can_take_over_input_owner():
    async def scenario():
        bridge = Bridge()
        bridge.select_mode("vr")
        app = create_app(bridge, "test-token")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        try:
            async with aiohttp.ClientSession() as client:
                first = await client.ws_connect(base + '/input?token=test-token')
                await first.send_json(packet(seq=1))
                await asyncio.sleep(.03)
                assert bridge.snapshot()[0]["seq"] == 1
                replacement = await client.ws_connect(
                    base + '/input?token=test-token&takeover=desktop-debug')
                assert bridge.session_id == 2 and bridge.input_connected
                assert len(bridge.clients) == 2 and bridge.owner is not None
                assert bridge.owner_session_id == 2
                assert bridge.owner_source == "desktop-debug" and not first.closed
                await first.send_json(packet(seq=2))
                await asyncio.sleep(.03)
                assert bridge.snapshot()[0] is None
                await replacement.send_json({"type": "ping", "client_ms": 456})
                assert json.loads((await replacement.receive()).data) == {
                    "type": "pong", "client_ms": 456}
                await replacement.send_json(packet(seq=0))
                await asyncio.sleep(.03)
                assert bridge.snapshot()[0]["seq"] == 0
                await replacement.close()
                await asyncio.sleep(.03)
                assert bridge.owner is None and bridge.input_connected
                await first.send_json(packet(seq=3))
                await asyncio.sleep(.03)
                assert bridge.snapshot()[0]["seq"] == 3
                await first.close()
                await asyncio.sleep(.03)
                assert not bridge.input_connected and bridge.owner is None
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_task_catalog_uses_official_instruction_and_success_requirement(tmp_path):
    source = tmp_path / "task/RoboDojo/tasks/example.py"
    source.parent.mkdir(parents=True)
    source.write_text('''\nclass Example:\n    def gen_instruction(self, env_idx):\n        color = "blue"\n        return [f"Stack the {color} block, then reset the robot arm."]\n    def run_reward(self):\n        self.reward_manager.check([self.reward_manager.is_stacked(),\n                                   self.reward_manager.all_robot_back_to_origin()])\n''')
    result = task_catalog(tmp_path, ["example"])["example"]
    assert result["instruction"] == "Stack the {…} block, then reset the robot arm."
    assert result["requires_robot_origin"]
    assert "双臂回到初始位" in result["success"]


def test_available_tasks_keeps_scripted_support_arm_tasks(tmp_path):
    for task in ("stack_blocks", "make_kong"):
        (tmp_path / f"Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json").parent.mkdir(
            parents=True, exist_ok=True)
        (tmp_path / f"Assets/Eval_Layout/RoboDojo/arx_x5/0/{task}_0.json").write_text("{}")
        (tmp_path / f"task/RoboDojo/config/{task}.yml").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"task/RoboDojo/config/{task}.yml").write_text("{}")
        (tmp_path / f"task/RoboDojo/tasks/{task}.py").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"task/RoboDojo/tasks/{task}.py").write_text("")
    (tmp_path / "task/RoboDojo/config/_task.yml").write_text(
        "common:\n  robot_config: dual_x5\ntasks:\n"
        "  stack_blocks:\n    eval_nums: 25\n"
        "  make_kong:\n    robot_config: dual_x5_and_franka_competition\n")

    assert available_tasks(tmp_path) == ["make_kong", "stack_blocks"]


def test_automatic_motion_tasks_are_recording_gated():
    assert SUPPORT_MOTION_GATED_TASKS == {
        "imitate_sorting_sequence", "make_kong", "play_tic_tac_toe"}
    assert CONVEYOR_TASKS == {
        "match_and_pick_from_conveyor", "pick_from_conveyor_by_image"}
    assert RECORDING_GATED_TASKS == SUPPORT_MOTION_GATED_TASKS | CONVEYOR_TASKS


def test_conveyor_surface_and_graph_follow_recording_gate():
    class Attribute:
        def __init__(self):
            self.value = False

        def Get(self):
            return self.value

        def Set(self, value):
            self.value = value

    class Surface:
        def __init__(self):
            self.attribute = Attribute()

        def GetSurfaceVelocityEnabledAttr(self):
            return self.attribute

    backend = RoboDojo.__new__(RoboDojo)
    backend.conveyor_surface = Surface()
    backend.conveyor_node_enabled = Attribute()
    backend.set_task_motion_enabled(True)
    assert backend.conveyor_surface.attribute.value is True
    assert backend.conveyor_node_enabled.value is True
    backend.set_task_motion_enabled(False)
    assert backend.conveyor_surface.attribute.value is False
    assert backend.conveyor_node_enabled.value is False


def test_all_tasks_have_chinese_prompts_and_dynamic_values_are_preserved():
    assert len(PROMPTS_ZH) == 54
    assert all(any('\u3400' <= char <= '\u9fff' for char in prompt) for prompt in PROMPTS_ZH.values())
    assert chinese_prompt(
        "stack_blocks_by_language",
        "stack the blocks from bottom to top in the order of blue, yellow, and orange, then reset the robot arm.",
    ) == "从下到上依次堆叠蓝色、黄色、橙色方块，然后将双臂复位。"
    assert "青绿色瓶" in chinese_prompt(
        "pour_by_language",
        "Pour the liquid from the first red bottle into the first black bowl, "
        "from the second turquoise bottle into the second white bowl, and "
        "from the third violet bottle into the third brown bowl. Then reset the robot arm.",
    )


def test_pause_is_teleop_only_and_requires_release_to_resume():
    controller = Controller()
    poses = {side: [0, 0, 1, 1, 0, 0, 0] for side in ("left", "right")}
    controller.pause()
    assert controller.diagnostics()["teleop_paused"]
    assert not controller.diagnostics()["simulation_paused"]
    assert controller.targets(packet(1.), poses) == ({}, {})
    controller.targets(packet(0.), poses)
    assert not controller.teleop_paused
    assert len(controller.targets(packet(1.), poses)[0]) == 2


def test_home_command_and_incremental_joint_targets():
    assert parse_command({"command": "home"}) == ("home", None)
    assert parse_command({"command": "next_scene"}) == ("next_scene", None)
    backend = RoboDojo.__new__(RoboDojo)
    backend.robots = {"left": object(), "right": object()}
    positions = {"left": np.ones(6), "right": -np.ones(6)}
    backend.home_joints = {side: np.zeros(6) for side in backend.robots}
    backend.hold_joints = {side: value.copy() for side, value in positions.items()}
    backend.hold_grippers = {side: 0. for side in backend.robots}
    backend.joints = lambda side: positions[side].copy()

    def step(targets, grippers, task_motion_enabled=True):
        assert targets == {} and grippers == {}
        assert task_motion_enabled
        for side in positions:
            positions[side] = backend.hold_joints[side].copy()
        return {f"{side}_arm_joint_states": positions[side].copy() for side in positions}

    backend.step = step
    _, done, error = backend.home_step(max_joint_step=.05, tolerance=.01)
    assert not done and error == pytest.approx(.95)
    assert all(value == 1. for value in backend.hold_grippers.values())
    for _ in range(20):
        _, done, error = backend.home_step(max_joint_step=.05, tolerance=.01)
        if done:
            break
    assert done and error <= .01


def test_epoch_change_clears_old_input_commands_and_preview():
    bridge = Bridge(env_epoch=3)
    bridge.packet = packet(seq=8)
    bridge.commands.append({"name": "pause"})
    bridge.publish(b"old", {"phase": "ready", "env_epoch": 3})
    bridge.reset_epoch(4)
    pose, commands, diagnostics = bridge.snapshot()
    assert pose is None and commands == [] and bridge.image is None
    assert diagnostics["received_seq"] == -1 and bridge.status["env_epoch"] == 4


def test_lifecycle_requests_are_serialized_and_idempotently_rejected():
    bridge = Bridge(env_epoch=7)
    first = bridge.submit_command("reset")
    duplicate = bridge.submit_command("select_task", "stack_blocks")
    assert first == {"accepted": True, "command": "reset", "operation_id": "7-1"}
    assert not duplicate["accepted"] and duplicate["operation_id"] == "7-1"
    bridge.update_operation("7-1", "complete", "ready")
    second = bridge.submit_command("select_task", "stack_blocks")
    assert second["accepted"] and second["operation_id"] == "7-2"
    bridge.update_operation("7-2", "complete", "ready")
    third = bridge.submit_command("next_scene")
    assert third == {"accepted": True, "command": "next_scene", "operation_id": "7-3"}


def test_writer_backpressure_is_explicit_and_never_blocks_forever(tmp_path):
    import threading
    rec = AsyncRecorder(tmp_path, {"input_source": "slow_writer"}, observation(0),
                        queue_size=1, join_timeout=1)
    original = rec.recorder.append
    entered, release = threading.Event(), threading.Event()

    def slow_append(*args, **kwargs):
        entered.set()
        assert release.wait(1)
        return original(*args, **kwargs)

    rec.recorder.append = slow_append
    command = {"left_arm_joint_states": np.ones(6)}
    rec.append(observation(0), observation(1), command, packet(), 0., 1.)
    assert entered.wait(1)
    rec.append(observation(1), observation(2), command, packet(seq=1), .04, 1.04)
    started = time.monotonic()
    with pytest.raises(RecorderBackpressure):
        rec.append(observation(2), observation(3), command, packet(seq=2), .08, 1.08)
    assert time.monotonic() - started < .1
    assert rec.diagnostics()["backpressure_count"] == 1
    release.set()
    rec.close_incomplete()
    with h5py.File(rec.path) as file:
        assert not file.attrs["complete"]


def test_preview_encoder_drops_intermediate_work_without_blocking(monkeypatch):
    import preview
    import threading
    entered, release = threading.Event(), threading.Event()
    published = []

    def slow_jpeg(image, quality=70):
        entered.set()
        assert release.wait(1)
        return bytes([int(image[0, 0, 0])])

    monkeypatch.setattr(preview, "jpeg", slow_jpeg)
    encoder = PreviewEncoder(published.append)
    encoder.submit(np.full((2, 2, 3), 1, dtype=np.uint8))
    assert entered.wait(1)
    encoder.submit(np.full((2, 2, 3), 2, dtype=np.uint8))
    encoder.submit(np.full((2, 2, 3), 3, dtype=np.uint8))
    assert encoder.diagnostics()["dropped"] == 1
    release.set()
    for _ in range(100):
        if published and published[-1] == b"\x03":
            break
        time.sleep(.01)
    encoder.close()
    assert published[-1] == b"\x03"


def _watchdog_child(tmp_path, source):
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(source))
    return [sys.executable, str(script), str(tmp_path / "progress.json")]


def test_process_out_watchdog_terminates_hang_and_limits_restarts(tmp_path):
    command = _watchdog_child(tmp_path, """
        import json, os, pathlib, sys, time
        path=pathlib.Path(sys.argv[1])
        path.write_text(json.dumps({'pid':os.getpid(),'phase':'reset_enter',
                                    'updated_monotonic':time.monotonic()}))
        while True: time.sleep(1)
    """)
    status = supervise(command, tmp_path / "progress.json", tmp_path / "diagnostics",
                       initial_timeout=.5, runtime_timeout=.1, reset_timeout=.1,
                       shutdown_timeout=.1, terminate_grace=.05, max_restarts=1,
                       poll_interval=.01)
    assert status != 0
    assert len(list((tmp_path / "diagnostics").glob("watchdog_*.json"))) == 2


def test_watchdog_allows_normal_initialization_before_runtime_timeout(tmp_path):
    command = _watchdog_child(tmp_path, """
        import json, os, pathlib, sys, time
        path=pathlib.Path(sys.argv[1])
        time.sleep(.15)
        for _ in range(5):
            path.write_text(json.dumps({'pid':os.getpid(),'phase':'ready',
                                        'updated_monotonic':time.monotonic()}))
            time.sleep(.02)
    """)
    status = supervise(command, tmp_path / "progress.json", tmp_path / "diagnostics",
                       initial_timeout=.5, runtime_timeout=.08, reset_timeout=.1,
                       shutdown_timeout=.1, terminate_grace=.05, max_restarts=0,
                       poll_interval=.01)
    assert status == 0


def test_watchdog_treats_collector_exception_as_failure_even_with_zero_exit(tmp_path):
    command = _watchdog_child(tmp_path, """
        import json, os, pathlib, sys, time
        path=pathlib.Path(sys.argv[1])
        path.write_text(json.dumps({'pid':os.getpid(),'phase':'shutdown_app_enter',
                                    'error':'collector_exception',
                                    'updated_monotonic':time.monotonic()}))
    """)
    status = supervise(command, tmp_path / "progress.json", tmp_path / "diagnostics",
                       initial_timeout=.5, runtime_timeout=.1, reset_timeout=.1,
                       shutdown_timeout=.1, terminate_grace=.05, max_restarts=0,
                       poll_interval=.01)
    assert status != 0


def test_watchdog_substitutes_task_and_device_state(tmp_path):
    script = tmp_path / "child.py"
    output = tmp_path / "resolved.txt"
    progress = tmp_path / "progress.json"
    task_file = tmp_path / "task.txt"
    device_file = tmp_path / "device.txt"
    task_file.write_text("pour_liquid_into_cup\n")
    device_file.write_text("cpu\n")
    script.write_text(textwrap.dedent("""
        import json, os, pathlib, sys, time
        task, device, progress, output = sys.argv[1:]
        pathlib.Path(output).write_text(f"{task}|{device}")
        pathlib.Path(progress).write_text(json.dumps({
            'pid': os.getpid(), 'phase': 'ready',
            'updated_monotonic': time.monotonic()}))
    """))
    status = supervise(
        [sys.executable, str(script), "__TASK__", "__DEVICE__", str(progress), str(output)],
        progress, tmp_path / "diagnostics", max_restarts=0,
        task_file=task_file, device_file=device_file,
    )
    assert status == 0
    assert output.read_text() == "pour_liquid_into_cup|cpu"
