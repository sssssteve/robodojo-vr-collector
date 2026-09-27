"""A local WebXR page, multi-client input lease, and bounded latest-frame preview."""
import asyncio
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import secrets
import ssl
import threading
import time

from aiohttp import web

from control import validate_packet
from leader_can import LeaderCan
from recording import episode_frame, list_episodes


COMMANDS = {"record", "record_recovery", "save", "discard", "delete_episode",
            "reset", "next_scene", "home", "pause", "select_task", "takeover",
            "set_random_scene"}
LIFECYCLE_COMMANDS = {"reset", "next_scene", "select_task", "set_random_scene"}
INPUT_TIMEOUT = .75
MAX_COMMANDS = 64


def parse_command(obj):
    name = obj.get("command")
    if name not in COMMANDS:
        raise ValueError("未知操作命令")
    value = None
    if name in ("select_task", "takeover", "delete_episode"):
        value = obj.get("value")
        if not isinstance(value, str) or not value or Path(value).name != value:
            raise ValueError("操作参数无效")
    elif name == "set_random_scene":
        value = obj.get("value")
        if not isinstance(value, bool):
            raise ValueError("随机场景开关参数无效")
    return name, value


class Bridge:
    def __init__(self, env_epoch=0, progress_path=None):
        self.lock = threading.Lock()
        self.packet = None
        self.received = 0.
        self.commands = deque()
        self.image = None
        self.image_seq = 0
        self.image_published = 0.
        self.observer_image = None
        self.observer_seq = 0
        self.observer_published = 0.
        self.status = {"phase": "starting", "frames": 0, "env_epoch": env_epoch}
        self.catalog = {}
        self.clients = {}
        self.owner = None
        self.owner_source = None
        self.owner_session_id = None
        self.owner_since = 0.
        self.last_seq = -1
        self.env_epoch = int(env_epoch)
        self.session_id = 0
        self.receive_count = 0
        self.received_seq = -1
        self.applied_seq = -1
        self.seq_gaps = 0
        self.input_connected = False
        self.last_tracking = {"left": False, "right": False}
        self.pending_operation = None
        self.last_operation = None
        self.operation_counter = 0
        self.progress_path = Path(progress_path) if progress_path else None
        self.vr_ui = {"open": False}
        self.vr_ui_session_id = None
        self.leader = None
        self.control_mode = "unselected"

    def select_mode(self, mode):
        if mode not in ("unselected", "vr", "leader"):
            raise ValueError("未知输入模式")
        with self.lock:
            if self.status.get("phase") == "recording":
                return False, "请先保存或丢弃当前录制，再切换输入模式"
            if self.control_mode == mode:
                return True, None
            self.control_mode = mode
            self.owner = None
            self.owner_source = None
            self.owner_session_id = None
            self.packet = None
            self.received = 0.
            self.last_tracking = {"left": False, "right": False}
            self.vr_ui = {"open": False}
            self.vr_ui_session_id = None
            if len(self.commands) < MAX_COMMANDS:
                self.commands.append({"name": "pause"})
            return True, None

    def _source_allowed(self, source):
        return ((self.control_mode == "vr" and source in ("webxr", "desktop-debug")) or
                (self.control_mode == "leader" and source == "leader-can"))

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            age = now - self.received if self.received else None
            timed_out = bool(self.received and age >= INPUT_TIMEOUT)
            packet = deepcopy(self.packet) if self.packet is not None and not timed_out else None
            # A pose is consumed at most once. A short gap holds the current target;
            # it must not integrate the same controller delta again.
            self.packet = None
            commands = list(self.commands)
            self.commands.clear()
            if not self.input_connected:
                input_state = "disconnected"
            elif self.owner is None:
                input_state = "idle"
            elif timed_out:
                input_state = "timeout"
            elif packet is None:
                input_state = "gap"
            elif not all(self.last_tracking.values()):
                input_state = "tracking_lost"
            else:
                input_state = "fresh"
            diagnostics = {
                "control_mode": self.control_mode,
                "session_id": self.session_id,
                "connected": self.input_connected,
                "client_count": len(self.clients),
                "active_session_id": self.owner_session_id,
                "active_source": self.owner_source,
                "received_seq": self.received_seq,
                "applied_seq": self.applied_seq,
                "receive_count": self.receive_count,
                "seq_gaps": self.seq_gaps,
                "effective_age_ms": None if age is None else round(age * 1000, 1),
                "timed_out": timed_out,
                "state": input_state,
                "left_tracking": self.last_tracking["left"],
                "right_tracking": self.last_tracking["right"],
                "client_buffered_amount": None if packet is None else packet.get("client_buffered_amount"),
                "client_dropped_poses": None if packet is None else packet.get("client_dropped_poses"),
                "sample_time_ms": None if packet is None else packet.get("sample_time_ms"),
            }
            return packet, commands, diagnostics

    def register_input(self, ws, source, takeover=False):
        with self.lock:
            self.session_id += 1
            session_id = self.session_id
            self.clients[ws] = {"session_id": session_id, "source": source}
            self.input_connected = True
            if takeover and self._source_allowed(source):
                self._activate_input_locked(ws)
            return session_id

    def _activate_input_locked(self, ws):
        client = self.clients[ws]
        if self.vr_ui_session_id not in (None, client["session_id"]):
            self.vr_ui = {"open": False}
            self.vr_ui_session_id = None
        self.owner = ws
        self.owner_source = client["source"]
        self.owner_session_id = client["session_id"]
        self.owner_since = time.monotonic()
        self.last_seq = -1
        self.packet = None
        self.received = 0.
        self.received_seq = -1
        self.applied_seq = -1
        self.last_tracking = {"left": False, "right": False}
        if len(self.commands) < MAX_COMMANDS:
            self.commands.append({"name": "pause"})

    def accept_packet(self, ws, packet):
        with self.lock:
            if (ws not in self.clients or packet.get("env_epoch") != self.env_epoch or
                    not self._source_allowed(self.clients[ws]["source"])):
                return False
            now = time.monotonic()
            owner_age = now - (self.received or self.owner_since) if self.owner is not None else None
            if self.owner is not ws:
                if self.owner is not None and owner_age < INPUT_TIMEOUT:
                    return False
                self._activate_input_locked(ws)
            if packet["seq"] <= self.last_seq:
                return False
            if self.last_seq >= 0 and packet["seq"] > self.last_seq + 1:
                self.seq_gaps += packet["seq"] - self.last_seq - 1
            self.last_seq = packet["seq"]
            self.received_seq = packet["seq"]
            self.receive_count += 1
            self.last_tracking = {side: side in packet["hands"] for side in ("left", "right")}
            self.packet = packet
            self.received = now
            return True

    def update_vr_ui(self, ws, state):
        with self.lock:
            client = self.clients.get(ws)
            if client is None or client["source"] != "webxr" or self.control_mode != "vr":
                return False
            if self.owner not in (None, ws):
                return False
            is_open = state.get("open")
            if not isinstance(is_open, bool):
                raise ValueError("VR 菜单状态无效")
            ui = {"open": is_open}
            if is_open:
                title, items, body, selected = (state.get("title"), state.get("items"),
                                                state.get("body", []), state.get("selected"))
                if (not isinstance(title, str) or len(title) > 120 or
                        not isinstance(items, list) or not 1 <= len(items) <= 128 or
                        not all(isinstance(item, str) and len(item) <= 240 for item in items) or
                        not isinstance(body, list) or len(body) > 12 or
                        not all(isinstance(line, str) and len(line) <= 500 for line in body) or
                        not isinstance(selected, int) or not 0 <= selected < len(items)):
                    raise ValueError("VR 菜单状态无效")
                ui.update(title=title, items=list(items), body=list(body), selected=selected)
            self.vr_ui = ui
            self.vr_ui_session_id = client["session_id"]
            return True

    def command_source_allowed(self, ws):
        with self.lock:
            client = self.clients.get(ws)
            return bool(client and self._source_allowed(client["source"]) and
                        self.owner in (None, ws))

    def unregister_input(self, ws):
        with self.lock:
            client = self.clients.pop(ws, None)
            if client and self.vr_ui_session_id == client["session_id"]:
                self.vr_ui = {"open": False}
                self.vr_ui_session_id = None
            self.input_connected = bool(self.clients)
            if self.owner is not ws:
                return
            self.owner = None
            self.owner_source = None
            self.owner_session_id = None
            self.owner_since = 0.
            self.last_tracking = {"left": False, "right": False}
            self.packet = None
            self.received = 0.
            if len(self.commands) < MAX_COMMANDS:
                self.commands.append({"name": "pause"})

    def mark_applied(self, packet):
        if packet is None:
            return
        with self.lock:
            self.applied_seq = packet["seq"]

    def submit_command(self, name, value=None, physical=False):
        with self.lock:
            if physical and (self.control_mode != "leader" or name not in
                             ("physical_record_toggle", "discard", "reset", "home",
                              "select_task", "next_scene", "set_random_scene")):
                return {"accepted": False, "command": name, "error": "请先选择实体双主臂模式"}
            if name == "physical_record_toggle" and not physical:
                return {"accepted": False, "command": name, "error": "该命令仅限实体双主臂键盘"}
            if name in ("record", "record_recovery") and self.control_mode == "unselected":
                return {"accepted": False, "command": name,
                        "error": "请先选择 VR 或实体双主臂输入模式"}
            if name in ("record", "record_recovery") and self.control_mode == "leader":
                return {"accepted": False, "command": name,
                        "error": "实体双主臂模式暂仅用于仿真联调，不可录制训练数据"}
            if name in LIFECYCLE_COMMANDS:
                if self.pending_operation is not None:
                    response = {"accepted": False, "command": name,
                                "error": "已有环境操作正在执行",
                                "operation_id": self.pending_operation["operation_id"]}
                    if value is not None:
                        response["value"] = value
                    return response
                self.operation_counter += 1
                operation_id = f"{self.env_epoch}-{self.operation_counter}"
                self.pending_operation = {
                    "operation_id": operation_id, "name": name, "status": "accepted",
                    "phase": "queued", "started_monotonic": time.monotonic(),
                }
            else:
                operation_id = None
            if len(self.commands) >= MAX_COMMANDS:
                if operation_id is not None:
                    self.pending_operation = None
                response = {"accepted": False, "command": name,
                            "error": "命令队列已满，请稍后重试"}
                if value is not None:
                    response["value"] = value
                return response
            command = {"name": name}
            if value is not None:
                command["value"] = value
            if operation_id is not None:
                command["operation_id"] = operation_id
            self.commands.append(command)
            if name in ("pause", "home", "reset", "next_scene", "discard", "save",
                        "select_task", "takeover"):
                self.packet = None
            response = {"accepted": True, "command": name}
            if value is not None:
                response["value"] = value
            if operation_id is not None:
                response["operation_id"] = operation_id
            return response

    def update_operation(self, operation_id, status, phase, error=None):
        with self.lock:
            if not self.pending_operation or self.pending_operation["operation_id"] != operation_id:
                return
            self.pending_operation.update(status=status, phase=phase)
            if error:
                self.pending_operation["error"] = str(error)
            if status in ("complete", "failed"):
                self.last_operation = dict(self.pending_operation)
                self.status["last_operation"] = dict(self.last_operation)
                self.pending_operation = None

    def publish(self, image, status):
        with self.lock:
            self.image, self.status = image, status
            self.image_seq += 1

    def publish_status(self, status):
        with self.lock:
            if self.pending_operation is not None:
                operation = dict(self.pending_operation)
                operation["duration_ms"] = round(
                    (time.monotonic() - operation["started_monotonic"]) * 1000, 1)
                status = dict(status, operation=operation)
            if self.last_operation is not None:
                status = dict(status, last_operation=dict(self.last_operation))
            self.status = status

    def publish_image(self, image):
        with self.lock:
            self.image = image
            self.image_seq += 1
            self.image_published = time.monotonic()

    def publish_observer(self, image):
        with self.lock:
            self.observer_image = image
            self.observer_seq += 1
            self.observer_published = time.monotonic()

    def reset_epoch(self, env_epoch):
        with self.lock:
            self.env_epoch = int(env_epoch)
            if self.control_mode == "leader":
                self.control_mode = "unselected"
                self.owner = None
                self.owner_source = None
                self.owner_session_id = None
            self.packet = None
            self.received = 0.
            self.commands.clear()
            self.image = None
            self.image_published = 0.
            self.image_seq += 1
            self.observer_image = None
            self.observer_published = 0.
            self.observer_seq += 1
            self.last_seq = -1
            self.received_seq = -1
            self.applied_seq = -1
            self.pending_operation = None
            self.last_operation = None
            self.status = {"phase": "starting", "frames": 0, "env_epoch": self.env_epoch}
            self.vr_ui = {"open": False}
            self.vr_ui_session_id = None


def create_app(bridge, token, output=None):
    app = web.Application(client_max_size=32768)
    leader = LeaderCan(bridge)
    bridge.leader = leader
    root = Path(__file__).parent / "web"
    output = Path(output).resolve() if output is not None else None

    def authorized(request):
        return secrets.compare_digest(request.query.get("token", ""), token)

    async def static(request):
        names = {"/client.js": "client.js", "/spectator": "spectator.html",
                 "/spectator.html": "spectator.html", "/spectator.js": "spectator.js"}
        name = names.get(request.path, "index.html")
        response = web.FileResponse(root / name)
        response.headers["Cache-Control"] = "no-store"
        return response

    async def status(request):
        if not authorized(request):
            raise web.HTTPUnauthorized()
        with bridge.lock:
            current = dict(bridge.status)
            current["control_mode"] = bridge.control_mode
            current["leader_can"] = leader.status()
            current["vr_ui"] = deepcopy(bridge.vr_ui)
            current["preview_transport"] = {
                "frame_seq": bridge.image_seq,
                "age_ms": None if not bridge.image_published else
                round((time.monotonic() - bridge.image_published) * 1000, 1),
            }
            current["observer_transport"] = {
                "frame_seq": bridge.observer_seq,
                "age_ms": None if not bridge.observer_published else
                round((time.monotonic() - bridge.observer_published) * 1000, 1),
            }
        if bridge.progress_path is not None:
            try:
                progress = json.loads(bridge.progress_path.read_text())
                current["lifecycle"] = progress
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return web.json_response(current)

    async def episodes(request):
        if not authorized(request) or output is None:
            raise web.HTTPUnauthorized()
        return web.json_response(list_episodes(output, task=request.query.get("task")))

    async def catalog(request):
        if not authorized(request):
            raise web.HTTPUnauthorized()
        return web.json_response(bridge.catalog)

    async def command(request):
        if not authorized(request):
            raise web.HTTPUnauthorized()
        try:
            name, value = parse_command(await request.json())
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="操作参数无效") from None
        response = bridge.submit_command(name, value)
        return web.json_response(response, status=202 if response["accepted"] else 409)

    async def physical_command(request):
        if not authorized(request):
            raise web.HTTPUnauthorized()
        try:
            payload = await request.json()
            name = payload.get("command")
            if name == "physical_record_toggle":
                value = None
            else:
                name, value = parse_command(payload)
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="操作参数无效") from None
        response = bridge.submit_command(name, value, physical=True)
        return web.json_response(response, status=202 if response["accepted"] else 409)

    async def mode_command(request):
        if not authorized(request):
            raise web.HTTPUnauthorized()
        try:
            mode = (await request.json()).get("mode")
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
            raise web.HTTPBadRequest(text="操作参数无效") from None
        if mode not in ("unselected", "vr", "leader"):
            raise web.HTTPBadRequest(text="操作参数无效")
        accepted, error = bridge.select_mode(mode)
        if accepted and mode == "leader":
            accepted, error = leader.start()
            if not accepted:
                bridge.select_mode("unselected")
        elif accepted:
            leader.stop()
        return web.json_response({"accepted": accepted, "error": error, "mode": bridge.control_mode,
                                  "leader_can": leader.status()},
                                 status=202 if accepted else 409)

    async def frame(request):
        if not authorized(request) or output is None:
            raise web.HTTPUnauthorized()
        try:
            payload, count = episode_frame(output, request.match_info["name"],
                                           int(request.query.get("index", 0)),
                                           task=bridge.status.get("task"))
        except (ValueError, IndexError, KeyError):
            raise web.HTTPBadRequest() from None
        except FileNotFoundError:
            raise web.HTTPNotFound() from None
        return web.Response(body=payload, content_type="image/jpeg", headers={"X-Frame-Count": str(count)})

    async def input_socket(request):
        if not authorized(request):
            raise web.HTTPUnauthorized()
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=32768)
        takeover = request.query.get("takeover") == "desktop-debug"
        source = "desktop-debug" if takeover else "webxr"
        try:
            await ws.prepare(request)
            bridge.register_input(ws, source, takeover=takeover)
            async for message in ws:
                if message.type != web.WSMsgType.TEXT:
                    continue
                try:
                    obj = json.loads(message.data)
                    if obj.get("type") == "ping":
                        await ws.send_json({"type": "pong", "client_ms": obj.get("client_ms")})
                    elif obj.get("type") == "command":
                        name, value = parse_command(obj)
                        response = (bridge.submit_command(name, value)
                                    if bridge.command_source_allowed(ws) else
                                    {"accepted": False, "command": name,
                                     "error": "当前页面不是已选控制入口，请先切换模式"})
                        await ws.send_json(response)
                    elif obj.get("type") == "ui_state":
                        bridge.update_vr_ui(ws, obj)
                    else:
                        packet = validate_packet(obj)
                        bridge.accept_packet(ws, packet)
                except (ValueError, TypeError, AttributeError):
                    with bridge.lock:
                        if bridge.owner is ws:
                            bridge.packet = None
                            if len(bridge.commands) < MAX_COMMANDS:
                                bridge.commands.append({"name": "pause"})
                    await ws.send_json({"error": "手柄数据无效，请松开双手侧握键后重试。"})
        finally:
            bridge.unregister_input(ws)
        return ws

    async def video_socket(request):
        if not authorized(request):
            raise web.HTTPUnauthorized()
        # This channel is send-only, so aiohttp never reads the browser's Pong
        # frames.  A server heartbeat would therefore close a healthy stream.
        ws = web.WebSocketResponse(max_msg_size=1024)
        await ws.prepare(request)
        last_sent = -1
        last_status_sent = 0.
        try:
            while not ws.closed:
                with bridge.lock:
                    image, current, sequence = bridge.image, dict(bridge.status), bridge.image_seq
                    current["vr_ui"] = deepcopy(bridge.vr_ui)
                    current["control_mode"] = bridge.control_mode
                    current["leader_can"] = leader.status()
                    current["preview_transport"] = {
                        "frame_seq": sequence,
                        "age_ms": None if not bridge.image_published else
                        round((time.monotonic() - bridge.image_published) * 1000, 1),
                    }
                if image is not None and sequence != last_sent:
                    await ws.send_json(current)
                    await ws.send_bytes(image)
                    last_sent = sequence
                    last_status_sent = time.monotonic()
                elif time.monotonic() - last_status_sent >= .5:
                    await ws.send_json(current)
                    last_status_sent = time.monotonic()
                else:
                    await asyncio.sleep(.01)
        except (ConnectionError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
            pass
        return ws

    async def observer_socket(request):
        if not authorized(request):
            raise web.HTTPUnauthorized()
        ws = web.WebSocketResponse(max_msg_size=1024)
        await ws.prepare(request)
        last_sent = -1
        try:
            while not ws.closed:
                with bridge.lock:
                    image, sequence = bridge.observer_image, bridge.observer_seq
                if image is not None and sequence != last_sent:
                    await ws.send_bytes(image)
                    last_sent = sequence
                else:
                    try:
                        await ws.receive(timeout=.01)
                    except asyncio.TimeoutError:
                        pass
        except (ConnectionError, ConnectionResetError, RuntimeError, asyncio.CancelledError):
            pass
        return ws

    app.router.add_get("/", static)
    app.router.add_get("/client.js", static)
    app.router.add_get("/spectator", static)
    app.router.add_get("/spectator.html", static)
    app.router.add_get("/spectator.js", static)
    app.router.add_get("/status", status)
    app.router.add_get("/episodes", episodes)
    app.router.add_get("/catalog", catalog)
    app.router.add_post("/command", command)
    app.router.add_post("/physical-command", physical_command)
    app.router.add_post("/mode", mode_command)
    app.router.add_get("/episode/{name}/frame", frame)
    app.router.add_get("/input", input_socket)
    app.router.add_get("/video", video_socket)
    app.router.add_get("/observer", observer_socket)
    async def stop_leader(_app):
        leader.stop()
    app.on_cleanup.append(stop_leader)
    return app


class Server:
    def __init__(self, bridge, host, port, token, cert=None, key=None, output=None,
                 allow_http_lan=False):
        self.app = create_app(bridge, token, output)
        self.host, self.port = host, port
        self.ssl = None
        if cert:
            self.ssl = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self.ssl.load_cert_chain(cert, key)
        elif host not in ("127.0.0.1", "localhost", "::1") and not allow_http_lan:
            raise ValueError("LAN WebXR requires HTTPS: provide --cert and --key")
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(10):
            raise RuntimeError("Web server did not start")
        if self.error:
            raise self.error

    def _run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.runner = web.AppRunner(self.app)
        try:
            self.loop.run_until_complete(self.runner.setup())
            site = web.TCPSite(self.runner, self.host, self.port, ssl_context=self.ssl)
            self.loop.run_until_complete(site.start())
        except Exception as exc:
            self.error = exc
            self.ready.set()
            self.loop.run_until_complete(self.runner.cleanup())
            self.loop.close()
            return
        self.ready.set()
        self.loop.run_forever()
        self.loop.run_until_complete(self.runner.cleanup())
        self.loop.close()

    def close(self):
        if self.thread.is_alive():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
