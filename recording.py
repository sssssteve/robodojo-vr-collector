"""Append-only HDF5 episodes with next-observed-state actions and provenance."""
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from queue import Full, Queue
import threading
import time
import uuid

import h5py
import numpy as np
from PIL import Image

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
SNAPSHOT_SCHEMA = "robodojo_state_v1"
JPEG_FIELD_BYTES = 300_000
OFFICIAL_STATE_FIELDS = (
    "left_arm_joint_states", "left_ee_joint_states", "left_ee_poses",
    "right_arm_joint_states", "right_ee_joint_states", "right_ee_poses",
)


class RecorderBackpressure(RuntimeError):
    pass


def jpeg(image, quality=90):
    stream = BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(stream, format="JPEG", quality=quality)
    return stream.getvalue()


def jpeg_payload(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).rstrip(b"\0")
    return np.asarray(value, dtype=np.uint8).tobytes()


class Recorder:
    def __init__(self, directory, metadata, observation):
        self.root = Path(directory)
        metadata = dict(metadata)
        metadata.pop("input_source", None)
        self.metadata = metadata
        task = metadata.get("task")
        embodiment = metadata.get("embodiment", "arx_x5")
        purpose = metadata.get("collection_purpose", "human_demonstration")
        self.training_eligible = purpose in {"human_demonstration", "failure_recovery"}
        if task:
            if self.training_eligible:
                self.data_directory = self.root / "RoboDojo" / task / embodiment / "data"
            else:
                self.data_directory = self.root / "validation" / task / embodiment
            staging = self.root / "staging" / task / embodiment
            ids = []
            for path in self.root.rglob("episode_*.hdf5"):
                try:
                    episode_id = path.name.removeprefix("episode_").removesuffix(".hdf5")
                    ids.append(int(episode_id.removesuffix(".partial")))
                except ValueError:
                    pass
            self.episode_name = f"episode_{max(ids, default=-1) + 1:07d}.hdf5"
            staging.mkdir(parents=True, exist_ok=True)
            self.path = staging / self.episode_name.replace(".hdf5", ".partial.hdf5")
        else:
            # Test fixtures and legacy callers without task metadata retain the
            # old flat layout; production collection always supplies a task.
            self.data_directory = self.root
            self.root.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            self.episode_name = f"episode_{stamp}_{uuid.uuid4().hex[:8]}.hdf5"
            self.path = self.root / self.episode_name.replace(".hdf5", ".partial.hdf5")
        self.max_wall_gap_s = float(metadata.get("max_wall_gap_s", .2))
        self.file = h5py.File(self.path, "x")
        self.frames = 0
        self.file["data_format_version"] = "v1.0"
        self.file["instruction"] = str(observation["instruction"])
        self.file["additional_info/frequency"] = 25
        self.file["metadata/json"] = json.dumps(metadata, ensure_ascii=False)
        self.file.attrs["complete"] = False
        self.file.attrs["action_semantics"] = "next_observed_state; command stores actual joint targets"
        self.file.attrs["camera_order"] = json.dumps(CAMERAS)
        self.file.attrs["quality_pass"] = False
        self.file.attrs["training_eligible"] = self.training_eligible
        self._terminal = None

    def append_array(self, name, value):
        value = np.asarray(value)
        if name not in self.file:
            self.file.create_dataset(name, shape=(0, *value.shape), maxshape=(None, *value.shape), dtype=value.dtype)
        ds = self.file[name]
        ds.resize(self.frames + 1, axis=0)
        ds[self.frames] = value

    def append(self, observation, next_observation, command, packet, sim_time, wall_time,
               action_valid=True):
        if tuple(sorted(observation["vision"])) != tuple(sorted(CAMERAS)):
            raise ValueError("Each sample must contain all three RoboDojo cameras")
        for group, values in [("state", observation["state"]), ("action", next_observation["state"]),
                              ("command", command)]:
            for name, value in values.items():
                array = np.asarray(value, dtype=np.float64)
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite {group}/{name}")
                self.append_array(f"{group}/{name}", array)
        for camera in CAMERAS:
            frame = observation["vision"][camera]
            name = f"vision/{camera}/colors"
            payload = jpeg(frame["color"])
            if len(payload) > JPEG_FIELD_BYTES:
                raise ValueError(f"JPEG for {camera} exceeds {JPEG_FIELD_BYTES} bytes")
            if name not in self.file:
                self.file.create_dataset(name, (0,), maxshape=(None,), dtype=f"S{JPEG_FIELD_BYTES}",
                                         chunks=(1,), compression="lzf")
                self.file[f"vision/{camera}/shape"] = np.asarray(frame["color"].shape)
                self.file[f"vision/{camera}/intrinsic_matrix"] = frame["intrinsic_matrix"]
            ds = self.file[name]
            ds.resize(self.frames + 1, axis=0)
            ds[self.frames] = payload
            self.append_array(f"vision/{camera}/extrinsic_matrix", frame["extrinsic_matrix"])
        self.append_array("timestamps/simulation", float(sim_time))
        self.append_array("timestamps/monotonic", float(wall_time))
        self.append_array("timestamps/render_dispatch", float(observation["render_stamp"]))
        self.append_array("action_valid", bool(action_valid))
        self.append_array("teleop/input_valid", bool(packet is not None or not action_valid))
        name = "teleop/input_json"
        if name not in self.file:
            self.file.create_dataset(name, (0,), maxshape=(None,), dtype=h5py.string_dtype())
        ds = self.file[name]
        ds.resize(self.frames + 1, axis=0)
        ds[self.frames] = json.dumps(packet)
        self.frames += 1
        if action_valid:
            self._terminal = (next_observation, command, float(sim_time) + .04, float(wall_time))
        if self.frames % 25 == 0:
            self.file.flush()

    def finish(self, outcome, reason, final_snapshot=None):
        if self.frames < 2:
            self.discard()
            raise ValueError("Episode too short to save")
        # The terminal state keeps array lengths aligned but is never a training
        # transition. Valid rows still satisfy action[t] == state[t + 1].
        terminal, command, sim_time, wall_time = self._terminal
        self.append(terminal, terminal, command, None, sim_time, wall_time,
                    action_valid=False)
        if final_snapshot is not None:
            write_snapshot(self.file, final_snapshot)
        self.file.attrs["complete"] = True
        self.file.attrs["success"] = bool(outcome)
        self.file.attrs["label_source"] = "robodojo_task_reward"
        self.file.attrs["termination_reason"] = reason
        self.file.flush()
        try:
            report = _inspect_open_file(self.file, require_complete=False,
                                        max_wall_gap_s=self.max_wall_gap_s)
            self.file.attrs["quality_pass"] = True
            self.file.attrs["quality_reason"] = "verified"
        except (AssertionError, KeyError, OSError, ValueError) as error:
            report = _timing_report(self.file, self.max_wall_gap_s)
            self.file.attrs["quality_pass"] = False
            self.file.attrs["quality_reason"] = str(error)
        for name, value in report.get("timing", {}).items():
            self.file.attrs[f"timing_{name}"] = value
        for name, value in report.get("controller_input", {}).items():
            self.file.attrs[f"controller_input_{name}"] = value
        self.file.attrs["training_eligible"] = self.training_eligible and report.get("quality_pass", False)
        self.file.close()
        if report.get("quality_pass", False):
            target_directory = self.data_directory
        else:
            target_directory = (self.root / "rejected" / self.metadata.get("task", "unknown") /
                                self.metadata.get("embodiment", "arx_x5"))
        target_directory.mkdir(parents=True, exist_ok=True)
        target = target_directory / self.episode_name
        self.path.rename(target)
        return target

    def discard(self):
        self.file.close()
        self.path.unlink(missing_ok=True)

    def close_incomplete(self):
        self.file.close()


class AsyncRecorder:
    """Serialize JPEG/HDF5 writes without blocking the simulation loop."""
    def __init__(self, *args, queue_size=16, join_timeout=30, **kwargs):
        self.recorder = Recorder(*args, **kwargs)
        self.path = self.recorder.path
        self.frames = 0
        self.error = None
        self.queue = Queue(maxsize=queue_size)
        self.join_timeout = float(join_timeout)
        self.backpressure_count = 0
        self.closing = False
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def _write(self):
        while True:
            item = self.queue.get()
            if item is None:
                self.closed.set()
                return
            if self.error is None:
                try:
                    args, kwargs = item
                    self.recorder.append(*args, **kwargs)
                except BaseException as error:
                    self.error = error

    def append(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        if self.closing:
            raise RuntimeError("Recorder is closing")
        try:
            self.queue.put_nowait((args, kwargs))
        except Full as error:
            self.backpressure_count += 1
            raise RecorderBackpressure(
                f"Recording writer queue is full ({self.queue.qsize()}/{self.queue.maxsize})") from error
        self.frames += 1

    def _join(self):
        self.closing = True
        try:
            self.queue.put(None, timeout=min(1., self.join_timeout))
        except Full as error:
            raise RecorderBackpressure("Recording writer did not drain for shutdown") from error
        self.thread.join(timeout=self.join_timeout)
        if self.thread.is_alive():
            raise TimeoutError(f"Recording writer did not stop within {self.join_timeout:g}s")
        if self.error is not None:
            self.recorder.close_incomplete()
            raise self.error

    def diagnostics(self):
        return {
            "queue_depth": self.queue.qsize(),
            "queue_capacity": self.queue.maxsize,
            "backpressure": self.queue.full(),
            "backpressure_count": self.backpressure_count,
            "writer_alive": self.thread.is_alive(),
            "writer_error": None if self.error is None else repr(self.error),
        }

    def close_incomplete_async(self):
        def close():
            try:
                self.close_incomplete()
            except BaseException as error:
                self.error = self.error or error
        threading.Thread(target=close, name="recorder-abort", daemon=True).start()

    def finish(self, *args, **kwargs):
        self._join()
        return self.recorder.finish(*args, **kwargs)

    def discard(self):
        self._join()
        return self.recorder.discard()

    def close_incomplete(self):
        self._join()
        return self.recorder.close_incomplete()


def write_snapshot(file, snapshot):
    """Store one final simulator state used for validated recovery takeover."""
    root = file.create_group("recovery_state")
    root.attrs["schema"] = SNAPSHOT_SCHEMA
    root.attrs["recoverable"] = bool(snapshot["recoverable"])
    root.attrs["reason"] = str(snapshot.get("reason", ""))
    for section in ("robots", "objects"):
        section_group = root.create_group(section)
        for name, state in snapshot.get(section, {}).items():
            item = section_group.create_group(name)
            for key, value in state.items():
                if isinstance(value, str):
                    item.attrs[key] = value
                else:
                    item.create_dataset(key, data=np.asarray(value))


def read_snapshot(path):
    with h5py.File(path, "r") as file:
        if "recovery_state" not in file:
            return None
        root = file["recovery_state"]
        result = {
            "schema": root.attrs["schema"],
            "recoverable": bool(root.attrs["recoverable"]),
            "reason": str(root.attrs.get("reason", "")),
            "robots": {},
            "objects": {},
        }
        for section in ("robots", "objects"):
            for name, item in root[section].items():
                state = {key: value[()] for key, value in item.items()}
                state.update({key: str(value) for key, value in item.attrs.items()})
                result[section][name] = state
        return result


def episode_path(directory, name, task=None):
    directory = Path(directory).resolve()
    if name != Path(name).name:
        raise ValueError("Invalid episode name")
    if Path(name).suffix != ".hdf5":
        raise ValueError("Invalid episode name")
    candidates = []
    legacy = directory / name
    if legacy.is_file():
        candidates.append(legacy)
    pattern = f"RoboDojo/{task}/*/data/{name}" if task else f"RoboDojo/*/*/data/{name}"
    candidates.extend(directory.glob(pattern))
    if len(candidates) != 1:
        if not candidates:
            raise FileNotFoundError(directory / name)
        raise ValueError("Episode name is ambiguous; specify its task")
    return candidates[0].resolve()


def list_episodes(directory, task=None, limit=40):
    """Return recent complete episodes without decoding their image payloads."""
    directory = Path(directory)
    result = []
    paths = list(directory.glob("episode_*.hdf5")) + list(directory.glob("RoboDojo/*/*/data/episode_*.hdf5"))
    for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            with h5py.File(path, "r") as file:
                if not bool(file.attrs.get("complete", False)):
                    continue
                if "quality_pass" in file.attrs and not bool(file.attrs["quality_pass"]):
                    continue
                metadata = json.loads(file["metadata/json"][()].decode())
                if task is not None and metadata.get("task") != task:
                    continue
                state = file.get("recovery_state")
                result.append({
                    "name": path.name,
                    "task": metadata.get("task", ""),
                    "purpose": metadata.get("collection_purpose", "human_demonstration"),
                    "success": bool(file.attrs.get("success", False)),
                    "frames": int(len(file["action_valid"])),
                    "recoverable": bool(state is not None and state.attrs.get("recoverable", False)),
                    "recovery_reason": "" if state is None else str(state.attrs.get("reason", "")),
                })
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
        if limit is not None and len(result) >= limit:
            break
    return result


def episode_counts(directory):
    """Count saved, training-eligible human collection episodes by task."""
    counts = {}
    for episode in list_episodes(directory, limit=None):
        if episode["purpose"] not in {"human_demonstration", "failure_recovery"}:
            continue
        task = episode["task"]
        if task:
            counts[task] = counts.get(task, 0) + 1
    return counts


def delete_episode(directory, name, task=None):
    path = episode_path(directory, name, task=task)
    if not path.is_file():
        raise FileNotFoundError(path)
    path.unlink()
    return path


def episode_frame(directory, name, index, camera="cam_head", task=None):
    if camera not in CAMERAS:
        raise ValueError("Unknown camera")
    path = episode_path(directory, name, task=task)
    with h5py.File(path, "r") as file:
        frames = file[f"vision/{camera}/colors"]
        if not 0 <= index < len(frames):
            raise IndexError(index)
        return jpeg_payload(frames[index]), len(frames)


def _timing_report(file, max_wall_gap_s):
    valid = np.asarray(file["action_valid"][:], dtype=bool)
    wall = np.asarray(file["timestamps/monotonic"][:], dtype=np.float64)[valid]
    gaps = np.diff(wall)
    median = float(np.median(gaps)) if len(gaps) else 0.
    return {"timing": {
        "wall_hz_median": 0. if median <= 0 else 1. / median,
        "wall_dt_p95_s": float(np.percentile(gaps, 95)) if len(gaps) else 0.,
        "wall_gap_max_s": float(np.max(gaps)) if len(gaps) else 0.,
        "wall_gap_threshold_s": float(max_wall_gap_s),
        "wall_gap_count": int(np.count_nonzero(gaps > max_wall_gap_s)),
    }}


def _inspect_open_file(file, require_complete=True, max_wall_gap_s=.2):
    if require_complete:
        assert bool(file.attrs["complete"]), "Incomplete episode"
    count = len(file["action_valid"])
    assert count >= 3, "Episode has fewer than two valid transitions"
    action_valid = np.asarray(file["action_valid"][:], dtype=bool)
    assert np.all(action_valid[:-1]) and not action_valid[-1], "Expected one invalid terminal row"
    assert set(file["state"]) == set(OFFICIAL_STATE_FIELDS), "Official RoboDojo state fields differ"
    assert set(file["action"]) == set(OFFICIAL_STATE_FIELDS), "Official RoboDojo action fields differ"
    for name in OFFICIAL_STATE_FIELDS:
        state, action = file[f"state/{name}"][:], file[f"action/{name}"][:]
        assert len(state) == len(action) == count, f"Length mismatch for {name}"
        assert state.dtype == np.float64 and action.dtype == np.float64, f"Wrong dtype for {name}"
        assert np.isfinite(state).all() and np.isfinite(action).all(), f"Nonfinite value in {name}"
        np.testing.assert_array_equal(action[:-1], state[1:])
    np.testing.assert_allclose(np.diff(file["timestamps/simulation"][:]), .04, atol=1e-6)
    assert np.all(np.diff(file["timestamps/render_dispatch"][:]) > 0), "Camera frame was repeated"
    input_valid = np.asarray(file["teleop/input_valid"][:], dtype=bool)
    missing_input_count = int(np.count_nonzero(~input_valid[action_valid]))
    for camera in CAMERAS:
        frames = file[f"vision/{camera}/colors"]
        assert len(frames) == count, f"Frame count mismatch for {camera}"
        assert len(file[f"vision/{camera}/extrinsic_matrix"]) == count
        for frame in frames:
            with Image.open(BytesIO(jpeg_payload(frame))) as image:
                image.load()
                assert image.size == (640, 480), f"Wrong image size for {camera}"
    timing = _timing_report(file, max_wall_gap_s)["timing"]
    raw_metadata = file["metadata/json"][()]
    source = json.loads(raw_metadata.decode() if isinstance(raw_metadata, bytes) else raw_metadata)
    if source.get("require_dual_controller_input"):
        assert missing_input_count == 0, "Physical recording contains missing controller input"
        for index in np.flatnonzero(action_valid):
            raw_packet = file["teleop/input_json"][index]
            packet = json.loads(raw_packet.decode() if isinstance(raw_packet, bytes) else raw_packet)
            assert (isinstance(packet, dict) and
                    set(packet.get("hands", {})) == {"left", "right"}), (
                "Physical recording requires both leader arms on every transition")
    return {"frames": count, "cameras": list(CAMERAS),
            "success": bool(file.attrs["success"]), "alignment": "verified",
            "quality_pass": True, "timing": timing,
            "controller_input": {
                "missing_transition_count": missing_input_count,
                "valid_transition_count": int(np.count_nonzero(action_valid)),
            }, "source": source}


def inspect_episode(path):
    """Read and decode every camera frame; verify temporal/structural alignment."""
    with h5py.File(path, "r") as file:
        threshold = float(file.attrs.get("timing_wall_gap_threshold_s", .2))
        report = _inspect_open_file(file, max_wall_gap_s=threshold)
        assert bool(file.attrs.get("quality_pass", True)), str(file.attrs.get("quality_reason", "rejected"))
        report["path"] = str(path)
        return report
