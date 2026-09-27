const connection = new URLSearchParams(location.hash.slice(1));
const token = connection.get('token') || '';
const direct = connection.get('direct') || '';
const statusElement = document.querySelector('#status');
const preview = document.querySelector('#preview');
const previewContext = preview.getContext('2d');
const lifecycleBar = document.querySelector('#lifecycle-bar');
const lifecycleLabel = document.querySelector('#lifecycle-label');
const base = direct ? `ws://${direct}` : `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;
document.querySelector('#spectator-link').href = `/spectator#${new URLSearchParams({token})}`;
let input, video, session, reference, gl, program, texture, sequence = 0;
let lastPoseSent = 0, latestStatus = {}, lastVideoTime = 0, previousButtons = {};
let inputReconnectTimer, videoReconnectTimer, wasVisible = true, screenOrigin = [0, 1.4, -1.5];
let menu = null, menuIndex = 0, menuStack = [], axisReady = true, review = null, pendingTask = null, pendingScene = null;
let taskCatalog = {}, lastStatusTask = null, switchingTask = null;
let switchingScene = null;
let networkRtt = 0, videoTransferMs = 0, videoDecodeMs = 0, statusReceivedAt = 0;
let videoFrameNumber = 0, videoDrawnFrame = 0, lastVideoMessage = 0;
let pendingVideoFrame = null, videoDecodeRunning = false, lastStatusUiUpdate = 0;
let panelDirty = true, panelWarning = '';
let lastEnvEpoch = null, poseDroppedBuffered = 0;
let menuSyncSupported = false;
let debugMode = false, debugAnimation = 0, debugLastTime = 0, debugSelected = 'left';
let vrSelected = false;
let pendingDebugStart = false;
const debugKeys = new Set();
const debugHands = {
  left: {pose: [-.25, 1.3, -.4, 0, 0, 0, 1], trigger: 0, squeeze: 0},
  right: {pose: [.25, 1.3, -.4, 0, 0, 0, 1], trigger: 0, squeeze: 0},
};
const banner = document.createElement('canvas');
banner.width = 1024; banner.height = 880;
const context = banner.getContext('2d');

function show(text) { statusElement.textContent = text; }
async function selectMode(mode) {
  try {
    const response = await fetch(`/mode?token=${encodeURIComponent(token)}`, {
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})});
    const result = await response.json();
    if (result.accepted) return true;
    show(`无法切换控制入口：${result.error || response.status}`);
  } catch (error) { show(`控制入口连接失败：${error.message}`); }
  return false;
}
function renderLifecycleProgress(msg) {
  const lifecycle = msg.lifecycle || {};
  const operation = msg.operation || lifecycle.operation || {};
  const raw = lifecycle.phase || operation.phase || msg.phase || 'starting';
  const ready = (msg.phase === 'ready' || msg.phase === 'recording') && operation.status !== 'running';
  const stages = [
    [/shutdown|restart_requested/, 15, '正在关闭旧场景'],
    [/process_start|starting/, 28, '正在启动新空间'],
    [/initialize_env/, 45, '正在创建仿真环境'],
    [/initialize_reset|reset_seed|reset_physics/, 62, '正在重置场景'],
    [/initialize_episode/, 74, '正在初始化任务'],
    [/initialize_render|render_enter/, 86, '正在初始化渲染'],
    [/first_observation|capture/, 95, '正在获取三相机首帧'],
  ];
  const stage = ready ? [null, 100, '空间已就绪'] : stages.find(([pattern]) => pattern.test(raw)) || [null, 8, '已接受任务切换'];
  const target = operation.target_task || lifecycle.task || msg.task || '目标任务';
  const elapsed = Number(lifecycle.phase_duration_ms ?? operation.duration_ms);
  lifecycleBar.value = stage[1];
  lifecycleLabel.textContent = `${target} · ${stage[2]} · ${stage[1]}%${Number.isFinite(elapsed) ? ` · 本阶段 ${(elapsed / 1000).toFixed(1)} 秒` : ''}`;
}
function collectedCount(task) { return Number(latestStatus.episode_counts?.[task] || 0); }
function taskLabel(task) {
  const variant = task.endsWith('_random') ? ' · 官方随机干扰任务' : '';
  return `${taskCatalog[task]?.title || task}${variant} · 已采 ${collectedCount(task)} 条`;
}
function command(name, value) {
  const payload = {type: 'command', command: name};
  if (value !== undefined) payload.value = value;
  if (input?.readyState === WebSocket.OPEN) {
    input.send(JSON.stringify(payload));
    if (name === 'select_task') {
      switchingTask = value;
      show(`已发送任务切换请求：${value}\n正在关闭旧场景并启动新场景，通常需要 90–130 秒。`);
    } else if (name === 'set_random_scene' || name === 'next_scene') {
      switchingScene = name === 'next_scene' ? 'next' : value;
      show(name === 'next_scene'
        ? '正在当前 Isaac 进程内更换官方布局。'
        : `正在${value ? '开启' : '关闭'}官方布局轮换。`);
    }
  } else if (name === 'select_task' || name === 'set_random_scene' || name === 'next_scene') {
    if (name === 'select_task') {
      pendingTask = payload;
      switchingTask = value;
      show(`控制通道重连中；任务切换已排队：${value}`);
    } else {
      pendingScene = payload;
      switchingScene = name === 'next_scene' ? 'next' : value;
      show(name === 'next_scene'
        ? '控制通道重连中；更换官方布局请求已排队。'
        : `控制通道重连中；布局轮换${value ? '开启' : '关闭'}请求已排队。`);
    }
  } else if (name === 'record' || name === 'record_recovery') {
    show('控制通道未连接，录制命令未发送；重连后请重新按 X。');
  }
}
document.querySelectorAll('[data-command]').forEach(button => {
  button.onclick = () => command(button.dataset.command, button.dataset.value);
});
function sendNeutral() {
  command('pause');
}
function publishMenuState() {
  if (!menuSyncSupported || input?.readyState !== WebSocket.OPEN) return;
  if (!menu) {
    input.send(JSON.stringify({type: 'ui_state', open: false}));
    return;
  }
  input.send(JSON.stringify({type: 'ui_state', open: true, title: menu.title,
    items: menu.items.map(entry => entry.label), body: menu.body || [], selected: menuIndex}));
}
function scheduleInputReconnect() {
  if (inputReconnectTimer) return;
  inputReconnectTimer = setTimeout(() => {
    inputReconnectTimer = null;
    try { input?.close(); } catch {}
    connectInput(pendingDebugStart);
  }, 1500);
}
function scheduleVideoReconnect() {
  if (videoReconnectTimer) return;
  videoReconnectTimer = setTimeout(() => {
    videoReconnectTimer = null;
    try { video?.close(); } catch {}
    connectVideo();
  }, 1500);
}
async function showConnectionProblem(channel) {
  if (channel === '控制') {
    document.querySelector('#enter').disabled = true;
    stopDesktopDebug(false);
  }
  try {
    const response = await fetch(`/status?token=${encodeURIComponent(token)}`, {cache: 'no-store'});
    if (response.status === 401) {
      show('连接凭证已失效。请使用工作站当前输出的完整地址重新打开页面。');
    } else if (switchingScene !== null) {
      show('正在进程内重建官方布局；如果安全兜底触发，页面会自动重连…');
    } else if (switchingTask) {
      show(`正在切换到 ${switchingTask}，仿真服务正在重启，请等待 90–130 秒…`);
    } else {
      show(`${channel}连接已断开，正在自动重连…`);
    }
  } catch {
    show(switchingTask
      ? `正在切换到 ${switchingTask}，仿真服务暂时不可用，正在自动重连…`
      : `${channel}连接已断开，仿真服务暂时不可用，正在自动重连…`);
  }
}
function connectInput(takeover=false) {
  const takeoverQuery = takeover ? '&takeover=desktop-debug' : '';
  input = new WebSocket(`${base}/input?token=${encodeURIComponent(token)}${takeoverQuery}`);
  input.onopen = async () => {
    show('已连接。等待仿真画面…');
    if (pendingDebugStart) {
      pendingDebugStart=false;
      startDesktopDebug();
    }
    if (pendingTask) {
      input.send(JSON.stringify(pendingTask));
      pendingTask = null;
    }
    if (pendingScene) {
      input.send(JSON.stringify(pendingScene));
      pendingScene = null;
    }
    publishMenuState();
    fetchTaskCatalog();
    const available = window.isSecureContext && navigator.xr && await navigator.xr.isSessionSupported('immersive-vr');
    document.querySelector('#enter').disabled = !available || !vrSelected;
    if (!available && !debugMode) show('当前浏览器没有可用的 immersive-vr 会话。');
  };
  input.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.accepted === false || msg.error) {
      if (msg.command === 'select_task') switchingTask = null;
      if (msg.command === 'set_random_scene' || msg.command === 'next_scene') switchingScene = null;
      show(`操作未接受：${msg.error || '请稍后重试'}`);
    }
    else if (msg.type === 'pong' && Number.isFinite(msg.client_ms)) networkRtt = Math.max(0, Date.now() - msg.client_ms);
    else if (msg.accepted && msg.command === 'select_task') {
      switchingTask = msg.value;
      show(`服务已接受任务切换请求：${msg.value}\n操作编号：${msg.operation_id}\n正在受控重启仿真。`);
    } else if (msg.accepted && msg.command === 'set_random_scene') {
      switchingScene = msg.value;
      show(`服务已接受布局轮换${msg.value ? '开启' : '关闭'}请求。\n操作编号：${msg.operation_id}`);
    } else if (msg.accepted && msg.command === 'next_scene') {
      switchingScene = 'next';
      show(`服务已接受进程内更换官方布局请求。\n操作编号：${msg.operation_id}`);
    } else if (msg.accepted && msg.command === 'reset') {
      show(`服务已接受重置请求。\n操作编号：${msg.operation_id}`);
    }
  };
  input.onclose = () => { showConnectionProblem('控制'); scheduleInputReconnect(); };
  input.onerror = () => { stopDesktopDebug(false); showConnectionProblem('控制'); };
}
function connectVideo() {
  video = new WebSocket(`${base}/video?token=${encodeURIComponent(token)}`);
  video.binaryType = 'blob';
  video.onopen = () => { lastVideoMessage = performance.now(); };
  video.onmessage = async event => {
    lastVideoMessage = performance.now();
    if (typeof event.data === 'string') {
      const msg = JSON.parse(event.data);
      if (lastEnvEpoch !== null && msg.env_epoch !== lastEnvEpoch) {
        pendingVideoFrame = null;
        videoDrawnFrame = videoFrameNumber;
        previewContext.fillStyle='#080e15'; previewContext.fillRect(0,0,1280,720);
        sendNeutral();
      }
      lastEnvEpoch = msg.env_epoch;
      latestStatus = msg;
      vrSelected = msg.control_mode === 'vr';
      renderLifecycleProgress(msg);
      statusReceivedAt = performance.now();
      panelDirty = true;
      if (!menuSyncSupported && Object.hasOwn(msg, 'vr_ui')) {
        menuSyncSupported = true;
        publishMenuState();
      }
      if (switchingTask && msg.task === switchingTask && msg.phase === 'ready') switchingTask = null;
      if (switchingScene === 'next' && msg.phase === 'ready' && msg.last_operation?.name === 'next_scene') switchingScene = null;
      else if (switchingScene !== null && msg.random_scene === switchingScene && msg.phase === 'ready') switchingScene = null;
      if (!session && performance.now() - lastStatusUiUpdate >= 250) {
        const phase = msg.arm_homing ? '机械臂复位中' :
          ({starting: '启动中', ready: '已就绪', recording: '录制中'}[msg.phase] || msg.phase);
        const inputState = msg.input?.state || '未知';
        const lifecycle = msg.lifecycle?.phase || msg.operation?.phase || phase;
        const motion = ({waiting_for_recording:'等待录制，场景自动运动已冻结',recording_active:'录制中，场景自动运动已放行',not_gated:'普通静态任务'}[msg.task_motion] || '未知');
        show(`${msg.task || ''} · ${phase} · 已采 ${collectedCount(msg.task)} 条 · ${msg.frames || 0} 帧 · epoch ${msg.env_epoch ?? '-'}\n${msg.message || ''}\n场景运动 ${motion}\n输入 ${inputState} · 来源 ${msg.input?.active_source || '-'} · 页面 ${msg.input?.client_count ?? 0} · 收到/应用 ${msg.input?.received_seq ?? '-'}/${msg.input?.applied_seq ?? '-'} · 保持原因 ${msg.hold_reason || '-'}\n生命周期 ${lifecycle} · 实际仿真 ${msg.wall_hz || 0} Hz · 网络 RTT ${Math.round(networkRtt)} ms · 传输 ${Math.round(videoTransferMs)} ms · 解码 ${Math.round(videoDecodeMs)} ms`);
        refreshTaskSelect();
        refreshEpisodeSelect();
        const random = document.querySelector('#random-scene');
        random.checked = !!msg.random_scene;
        document.querySelector('#scene-seed').textContent = `当前场景 seed：${msg.scene_seed ?? 0} · 可用官方场景：${msg.layout_count ?? 1}`;
        lastStatusUiUpdate = performance.now();
      }
      return;
    }
    const frameNumber = ++videoFrameNumber;
    const transferMs = Math.max(0, performance.now() - statusReceivedAt);
    if (!review) queueVideoFrame(event.data, frameNumber, transferMs);
  };
  video.onclose = () => { pendingVideoFrame = null; showConnectionProblem('画面'); scheduleVideoReconnect(); };
}

function queueVideoFrame(blob, frameNumber, transferMs) {
  pendingVideoFrame = {blob, frameNumber, transferMs};
  if (!videoDecodeRunning) decodeLatestVideoFrame();
}

async function decodeLatestVideoFrame() {
  videoDecodeRunning = true;
  while (pendingVideoFrame) {
    const frame = pendingVideoFrame;
    pendingVideoFrame = null;
    try {
      const decodeStart = performance.now();
      const bitmap = await createImageBitmap(frame.blob);
      if (frame.frameNumber > videoDrawnFrame) {
        previewContext.drawImage(bitmap, 0, 0, 1280, 720);
        previewContext.fillStyle='#fff'; previewContext.font='24px sans-serif';
        previewContext.fillText('左夹爪',20,225); previewContext.fillText('头部',610,105); previewContext.fillText('右夹爪',1175,225);
        videoDrawnFrame = frame.frameNumber; lastVideoTime = performance.now();
        videoTransferMs = frame.transferMs; videoDecodeMs = lastVideoTime - decodeStart;
        preview.dataset.frame = String(frame.frameNumber);
        panelDirty = true;
      }
      bitmap.close();
    } catch (error) { console.error(error); show('画面解码失败，请刷新页面后重试。'); }
  }
  videoDecodeRunning = false;
}
function connect() {
  if (!token) { show('请使用包含 #token=… 的完整网址。'); return; }
  connectInput();
  connectVideo();
}

function refreshTaskSelect() {
  const select = document.querySelector('#task');
  if (!select || !latestStatus.tasks) return;
  if (!select.options.length) {
    for (const task of latestStatus.tasks) {
      const option = document.createElement('option');
      option.value = task; option.textContent = taskLabel(task);
      select.appendChild(option);
    }
  }
  for (const option of select.options) option.textContent = taskLabel(option.value);
  if (latestStatus.task && latestStatus.task !== lastStatusTask) {
    select.value = latestStatus.task;
    lastStatusTask = latestStatus.task;
  }
  refreshTaskDescription();
}
async function fetchTaskCatalog() {
  try {
    const response = await fetch(`/catalog?token=${encodeURIComponent(token)}`, {cache: 'no-store'});
    if (response.ok) {
      taskCatalog = await response.json();
      if (!latestStatus.tasks?.length) latestStatus.tasks = Object.keys(taskCatalog);
      refreshTaskSelect();
    }
    for (const option of document.querySelectorAll('#task option')) {
      option.textContent = taskLabel(option.value);
    }
    refreshTaskDescription();
  } catch {}
}
function taskInfo(task) {
  if (task === latestStatus.task && latestStatus.task_info) return latestStatus.task_info;
  return taskCatalog[task] || {title: task, instruction: '暂无任务说明', success: ''};
}
function refreshTaskDescription() {
  const select = document.querySelector('#task');
  if (!select?.value) return;
  const info = taskInfo(select.value);
  document.querySelector('#task-description').textContent = `已保存可训练轨迹：${collectedCount(select.value)} 条\n中文 Prompt：${info.prompt_zh || '暂无'}\n官方指令：${info.instruction}\n成功方式：${info.success}`;
  const activePrompt = latestStatus.task_info?.prompt_zh;
  document.querySelector('#active-prompt').textContent = activePrompt || '等待任务信息…';
  const current = select.value === latestStatus.task;
  document.querySelector('#scene-description').textContent = current
    ? `当前实际场景：${latestStatus.layout_name || '加载中'} · ${(latestStatus.layout_sha256 || '').slice(0, 12)}\n${(latestStatus.scene_objects || []).join('；')}`
    : '切换后将完整重启并加载该任务的官方 layout 0。';
}
function refreshEpisodeSelect() {
  const select = document.querySelector('#episode');
  const failures = (latestStatus.episodes || []).filter(episode => !episode.success);
  const signature = failures.map(episode => `${episode.name}:${episode.recoverable}`).join('|');
  if (select.dataset.signature === signature) return;
  select.dataset.signature = signature;
  select.replaceChildren();
  if (!failures.length) {
    const option = document.createElement('option'); option.value=''; option.textContent='暂无失败轨迹';
    select.appendChild(option); return;
  }
  for (const episode of failures) {
    const option = document.createElement('option'); option.value=episode.name;
    option.textContent=`${episode.recoverable ? '可接手' : '仅视频'} · ${episode.name}`;
    select.appendChild(option);
  }
}
function selectedEpisode() {
  const name = document.querySelector('#episode').value;
  return (latestStatus.episodes || []).find(episode => episode.name === name);
}

function item(label, run, disabled=false) { return {label, run, disabled}; }
function openMenu(title, items, push=true, body=[]) {
  if (push && menu) menuStack.push({menu, menuIndex});
  menu = {title, items, body}; menuIndex = 0; axisReady = false;
  panelDirty = true;
  sendNeutral();
  publishMenuState();
}
function backMenu() {
  if (review) { review = null; openMainMenu(false); return; }
  const previous = menuStack.pop();
  if (previous) { menu = previous.menu; menuIndex = previous.menuIndex; }
  else menu = null;
  panelDirty = true;
  publishMenuState();
}
function closeMenu() {
  menu = null; menuStack = []; axisReady = false; panelDirty = true;
  publishMenuState();
}
function confirmMenu() {
  const selected = menu?.items[menuIndex];
  if (selected && !selected.disabled) selected.run();
}
function confirmDelete(episode) {
  openMenu('确认删除已保存轨迹？', [
    item(`删除 ${episode.name}`, () => { command('delete_episode', episode.name); closeMenu(); }),
    item('取消', backMenu),
  ]);
}
function recoveryReason(episode) {
  const reason = episode.recovery_reason || '';
  if (!reason) return '没有可用的状态快照';
  if (reason.includes('particle state')) return '衣物或液体粒子状态尚未通过恢复验证';
  if (reason.includes('snapshot failed')) return '状态快照保存失败';
  return reason.match(/[\u3400-\u9fff]/) ? reason : '状态快照不可用或未通过恢复校验';
}
function episodeMenu(episode) {
  const options = [item('播放失败视频', () => playEpisode(episode))];
  options.push(item(episode.recoverable ? '从末状态接手并录制恢复' : `不可接手：${recoveryReason(episode)}`,
                    () => { command('takeover', episode.name); closeMenu(); }, !episode.recoverable));
  options.push(item('删除这条轨迹', () => confirmDelete(episode)));
  options.push(item('返回', backMenu));
  openMenu(episode.name, options);
}
function openRecoveryMenu() {
  const failures = (latestStatus.episodes || []).filter(episode => !episode.success);
  const options = [item('从当前错误状态开始恢复录制', () => { command('record_recovery'); closeMenu(); })];
  for (const episode of failures) options.push(item(`${episode.recoverable ? '↪' : '▶'} ${episode.name}`, () => episodeMenu(episode)));
  options.push(item('返回', backMenu));
  openMenu('错误恢复', options);
}
function openTaskMenu() {
  const options = (latestStatus.tasks || []).map(task =>
    item(`${task === latestStatus.task ? '●' : '○'} ${taskLabel(task)}`, () => openTaskConfirm(task)));
  options.push(item('返回', backMenu));
  openMenu('选择 RoboDojo 任务', options);
}
function openTaskConfirm(task) {
  const info = taskInfo(task);
  const current = task === latestStatus.task;
  openMenu(info.title, [
    item(current ? '当前任务，无需切换' : `确认切换到 ${info.title}`, () => {
      command('select_task', task); closeMenu();
    }, current),
    item('返回任务列表', backMenu),
  ], true, [`中文 Prompt：${info.prompt_zh || '暂无'}`, `官方指令：${info.instruction}`, `成功方式：${info.success}`]);
}
function openControlsMenu() {
  openMenu('Meta Touch / Pico 手柄说明', [item('返回', backMenu)], true, [
    '双手侧握键：按住时移动对应机械臂，松开后机械臂保持。首次进入、重连或重新对齐后，先松开双手侧握键一次。',
    '双手扳机：控制对应夹爪开合；菜单打开时用于确认选项。',
    '左 X：开始录制；左 Y：保存录制；右 A：双臂复位到初始位；右 B：快速重置当前场景。',
    '按下任一摇杆：打开或关闭菜单。菜单内摇杆上下选择，右 B 返回。',
  ]);
}
function openMainMenu(push=true) {
  const latest = (latestStatus.episodes || [])[0];
  openMenu('RoboDojo 数采菜单', [
    item('开始普通录制', () => { command('record'); closeMenu(); }),
    item('错误恢复', openRecoveryMenu),
    item('保存当前录制', () => { command('save'); closeMenu(); }),
    item('双臂复位到初始位', () => { command('home'); closeMenu(); }),
    item('丢弃当前未保存录制', () => openMenu('确认丢弃当前录制？', [
      item('确认丢弃', () => { command('discard'); closeMenu(); }), item('取消', backMenu)])),
    item(latest ? `删除最近保存：${latest.name}` : '没有已保存轨迹',
         () => confirmDelete(latest), !latest),
    item('选择任务', openTaskMenu),
    item('手柄按键说明', openControlsMenu),
    item(`官方布局轮换：${latestStatus.random_scene ? '开' : '关'}`, () => {
      command('set_random_scene', !latestStatus.random_scene); closeMenu();
    }),
    item('快速重置当前场景', () => { command('reset'); closeMenu(); }),
    item('更换下一份官方布局', () => { command('next_scene'); closeMenu(); }, !latestStatus.random_scene),
    item('退出 VR（保留网页）', () => openMenu('确认退出 VR？', [item('确认退出 VR', exitPage), item('取消', backMenu)])),
  ], push);
}

async function playEpisode(episode) {
  sendNeutral(); closeMenu();
  review = {episode, index: 0, stopped: false};
  while (review && !review.stopped) {
    try {
      const response = await fetch(`/episode/${encodeURIComponent(episode.name)}/frame?token=${encodeURIComponent(token)}&index=${review.index}`,
                                   {cache: 'no-store'});
      if (!response.ok) break;
      const total = Number(response.headers.get('X-Frame-Count')) || episode.frames;
      const bitmap = await createImageBitmap(await response.blob());
      previewContext.fillStyle='#000'; previewContext.fillRect(0,0,1280,720);
      previewContext.drawImage(bitmap, 320, 120, 640, 480); bitmap.close();
      lastVideoTime = performance.now();
      panelDirty = true;
      review.index += 1;
      if (review.index >= total) break;
      await new Promise(resolve => setTimeout(resolve, 40));
    } catch { break; }
  }
  if (review) {
    const selected = review.episode; review = null;
    if (session) episodeMenu(selected);
  }
}

async function exitPage() {
  sendNeutral(); closeMenu();
  if (session) await session.end();
  show('已退出 VR；网页和连接会保持打开，可随时再次进入 VR。');
}

function shader(type, source) {
  const obj = gl.createShader(type); gl.shaderSource(obj, source); gl.compileShader(obj);
  if (!gl.getShaderParameter(obj, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(obj));
  return obj;
}
function setupGL() {
  program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, `
    attribute vec3 position; attribute vec2 uv; varying vec2 v;
    uniform mat4 projection, view; uniform vec3 origin;
    void main(){v=uv;gl_Position=projection*view*vec4(position+origin,1.0);}`));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, `
    precision mediump float; varying vec2 v; uniform sampler2D screen;
    void main(){gl_FragColor=texture2D(screen,v);}`));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program,gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  const data = new Float32Array([-1,-.86,0,0,0, 1,-.86,0,1,0, -1,.86,0,0,1,
                                 -1,.86,0,0,1, 1,-.86,0,1,0, 1,.86,0,1,1]);
  const buffer = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
  for (const [name, size, offset] of [['position',3,0],['uv',2,12]]) {
    const loc = gl.getAttribLocation(program, name); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,size,gl.FLOAT,false,20,offset);
  }
  texture = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,texture);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,banner.width,banner.height,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
  panelDirty = true;
}

function drawMenu() {
  if (!menu) return;
  context.fillStyle='rgba(4,12,20,.92)'; context.fillRect(90,70,844,720);
  context.fillStyle='#73e5b3'; context.font='bold 32px sans-serif'; context.fillText(menu.title,125,120);
  let itemTop = 145;
  context.font='20px sans-serif'; context.fillStyle='#d7e3ec';
  for (const paragraph of menu.body || []) {
    const words = paragraph.split(' '); let line='';
    for (const word of words) {
      const next = line ? `${line} ${word}` : word;
      if (context.measureText(next).width > 730 && line) {
        context.fillText(line,145,itemTop+24); itemTop += 30; line=word;
      } else line=next;
    }
    if (line) { context.fillText(line,145,itemTop+24); itemTop += 38; }
  }
  const visibleItems = menu.body?.length ? 4 : 8;
  const start = Math.max(0, Math.min(menuIndex - Math.floor(visibleItems/2), Math.max(menu.items.length - visibleItems, 0)));
  const end = Math.min(menu.items.length, start + visibleItems);
  context.font='25px sans-serif';
  for (let i=start;i<end;i++) {
    context.fillStyle = i === menuIndex ? '#247d66' : 'rgba(255,255,255,.06)';
    context.fillRect(120,itemTop+(i-start)*70,784,56);
    context.fillStyle = menu.items[i].disabled ? '#657380' : '#fff';
    context.fillText(`${i === menuIndex ? '› ' : '  '}${menu.items[i].label}`,145,itemTop+37+(i-start)*70);
  }
  context.fillStyle='#aebdca'; context.font='20px sans-serif';
  context.fillText('摇杆上下选择 · 扳机确认 · B 返回 · 按下摇杆关闭菜单',150,755);
}
function drawPanel(time, viewer) {
  const sideRelease = latestStatus.side_release_required || {};
  const releaseHand = sideRelease.left && sideRelease.right ? '双手' : sideRelease.left ? '左手' : sideRelease.right ? '右手' : '';
  const warning = input?.readyState !== WebSocket.OPEN ? '控制通道未连接，录制命令未发送'
    : latestStatus.release_required ? '首次接管或重连后，请松开双手侧握键一次'
    : releaseHand ? `${releaseHand}追踪中断，请松开对应侧握键后重新按住`
    : time-lastVideoTime > 2000 ? '画面更新较慢，手柄连接保持中'
    : '';
  if (warning !== panelWarning) { panelWarning = warning; panelDirty = true; }
  if (panelDirty) {
    context.fillStyle = '#101b26'; context.fillRect(0,0,1024,880);
    context.drawImage(preview,0,80,1024,576);
    context.fillStyle = latestStatus.phase === 'recording' ? '#ff7165' : '#73e5b3';
    context.font = '26px sans-serif';
    const phase = latestStatus.arm_homing ? '机械臂复位中' :
      ({starting: '启动中', ready: '已就绪', recording: '录制中'}[latestStatus.phase] || '连接中');
    context.fillText(`${latestStatus.task || 'RoboDojo'} | ${phase} | 已采 ${collectedCount(latestStatus.task)} 条 | ${latestStatus.frames || 0} 帧`,24,34);
    context.textAlign='right'; context.font='20px sans-serif';
    context.fillText(`${latestStatus.random_scene ? '随机' : '固定'}场景 · seed ${latestStatus.scene_seed ?? 0}`,1000,34);
    context.textAlign='left';
    const prompt = latestStatus.task_info?.prompt_zh;
    if (prompt) {
      context.font='bold 24px sans-serif';
      const lines=[]; let line='';
      for (const char of `任务：${prompt}`) {
        if (context.measureText(line + char).width > 930 && line) { lines.push(line); line=char; }
        else line += char;
      }
      if (line) lines.push(line);
      const shown=lines.slice(0,3);
      if (lines.length > 3) shown[2] = shown[2].slice(0,-1) + '…';
      context.fillStyle='rgba(4,12,20,.82)'; context.fillRect(18,62,988,shown.length*34+22);
      context.fillStyle='#fff4b8';
      shown.forEach((text,index) => context.fillText(text,38,94+index*34));
    }
    context.fillStyle = '#ffffff'; context.font='20px sans-serif';
    if (menu) {
      context.fillText('菜单已暂停遥操作',15,846);
    } else {
      context.fillText('左 X：录制 · 左 Y：保存 · 右 A：双臂复位 · 右 B：重置场景',15,812);
      context.fillText('侧握键：按住移动/松开保持 · 扳机：夹爪 · 按下任一摇杆：菜单',15,846);
    }
    if (review) context.fillText(`失败视频 ${review.index}/${review.episode.frames}`,720,34);
    if (warning) { context.fillStyle='#ff7165'; context.fillText(warning,24,875); }
    if (switchingTask) {
      context.fillStyle='#ffcf70'; context.fillText(`正在切换到 ${switchingTask}，请等待场景重载…`,24,815);
    } else if (switchingScene !== null) {
      const layoutAction = switchingScene === 'next'
        ? '更换官方布局'
        : `${switchingScene ? '开启' : '关闭'}官方布局轮换`;
      context.fillStyle='#ffcf70'; context.fillText(`正在${layoutAction}，请等待场景重建…`,24,815);
    }
    drawMenu();
    gl.bindTexture(gl.TEXTURE_2D,texture);
    gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,gl.RGBA,gl.UNSIGNED_BYTE,banner);
    panelDirty = false;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER,session.renderState.baseLayer.framebuffer);
  gl.clearColor(.025,.04,.06,1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program); gl.uniform3fv(gl.getUniformLocation(program,'origin'),screenOrigin);
  for (const view of viewer.views) {
    const vp = session.renderState.baseLayer.getViewport(view); gl.viewport(vp.x,vp.y,vp.width,vp.height);
    gl.uniformMatrix4fv(gl.getUniformLocation(program,'projection'),false,view.projectionMatrix);
    gl.uniformMatrix4fv(gl.getUniformLocation(program,'view'),false,view.transform.inverse.matrix);
    gl.drawArrays(gl.TRIANGLES,0,6);
  }
}

function edge(key, pressed) {
  const result = pressed && !previousButtons[key];
  previousButtons[key] = pressed;
  return result;
}

function sendPosePacket(time, hands) {
  if (time-lastPoseSent < 1000/60 || input?.readyState !== WebSocket.OPEN) return;
  if (input.bufferedAmount === 0) {
    input.send(JSON.stringify({type:'pose',seq:sequence++,env_epoch:latestStatus.env_epoch,
      sample_time_ms:time,client_buffered_amount:input.bufferedAmount,
      client_dropped_poses:poseDroppedBuffered,hands}));
    poseDroppedBuffered=0; lastPoseSent=time;
  } else {
    poseDroppedBuffered += 1;
  }
}

function multiplyQuaternion(a, b) {
  return [
    a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
    a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
    a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],
    a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2],
  ];
}
function rotateDebugHand(axis, angle) {
  if (!angle) return;
  const half = angle/2, sine = Math.sin(half);
  const delta = [axis[0]*sine, axis[1]*sine, axis[2]*sine, Math.cos(half)];
  const pose = debugHands[debugSelected].pose;
  const q = multiplyQuaternion(pose.slice(3), delta);
  const length = Math.hypot(...q) || 1;
  pose.splice(3, 4, ...q.map(value => value/length));
}
function desktopDebugFrame(time) {
  if (!debugMode) return;
  debugAnimation = requestAnimationFrame(desktopDebugFrame);
  const dt = Math.min(Math.max((time-debugLastTime)/1000, 0), .05);
  debugLastTime = time;
  const pose = debugHands[debugSelected].pose;
  const move = .18*dt, turn = 1.0*dt;
  pose[0] += (debugKeys.has('KeyD')-debugKeys.has('KeyA'))*move;
  pose[1] += (debugKeys.has('KeyR')-debugKeys.has('KeyF'))*move;
  pose[2] += (debugKeys.has('KeyS')-debugKeys.has('KeyW'))*move;
  rotateDebugHand([1,0,0], (debugKeys.has('KeyI')-debugKeys.has('KeyK'))*turn);
  rotateDebugHand([0,1,0], (debugKeys.has('KeyJ')-debugKeys.has('KeyL'))*turn);
  rotateDebugHand([0,0,1], (debugKeys.has('KeyU')-debugKeys.has('KeyO'))*turn);
  for (const hand of Object.values(debugHands)) { hand.trigger=0; hand.squeeze=0; }
  debugHands[debugSelected].squeeze = debugKeys.has('Space') ? 1 : 0;
  debugHands[debugSelected].trigger = debugKeys.has('ShiftLeft') || debugKeys.has('ShiftRight') ? 1 : 0;
  const hands = Object.fromEntries(Object.entries(debugHands).map(([name, hand]) =>
    [name, {pose:[...hand.pose], trigger:hand.trigger, squeeze:hand.squeeze}]));
  sendPosePacket(time, hands);
  const selected = debugSelected === 'left' ? '左手' : '右手';
  document.querySelector('#debug-state').textContent = `${selected} · ${debugHands[debugSelected].squeeze ? '侧握按下' : '侧握松开'} · ${debugHands[debugSelected].trigger ? '扳机按下' : '扳机松开'}`;
}
function stopDesktopDebug(pause=true) {
  if (!debugMode) return;
  debugMode=false; pendingDebugStart=false; debugKeys.clear();
  if (debugAnimation) cancelAnimationFrame(debugAnimation);
  debugAnimation=0;
  document.querySelector('#debug').classList.remove('active');
  document.querySelector('#debug').textContent='桌面调试模式';
  document.querySelector('#debug-help').hidden=true;
  if (pause) {
    sendNeutral();
    const previousInput=input;
    if (previousInput) {
      previousInput.onclose=null; previousInput.onerror=null;
      try { previousInput.close(); } catch {}
      input=null;
      connectInput();
    }
  }
}
function releaseDesktopDebugControls() {
  if (!debugMode) return;
  debugKeys.clear();
  for (const hand of Object.values(debugHands)) { hand.trigger=0; hand.squeeze=0; }
  document.querySelector('#debug-state').textContent='窗口已失焦：按键已释放，返回页面后可继续';
  sendNeutral();
}
function startDesktopDebug() {
  if (session) { show('请先退出真实 VR 会话，再启动桌面调试模式。'); return; }
  if (input?.readyState !== WebSocket.OPEN) { show('控制通道尚未连接，不能启动桌面调试模式。'); return; }
  debugMode=true; debugKeys.clear(); debugLastTime=performance.now(); lastPoseSent=0;
  const button = document.querySelector('#debug');
  button.classList.add('active');
  button.textContent='退出桌面调试';
  button.blur();
  document.querySelector('#debug-help').hidden=false;
  show('桌面调试模式已启动。这是模拟输入，不代表 Meta Quest / WebXR 已验证。');
  debugAnimation=requestAnimationFrame(desktopDebugFrame);
}
async function requestDesktopDebug() {
  if (session) { show('请先退出真实 VR 会话，再启动桌面调试模式。'); return; }
  if (!await selectMode('vr')) return;
  pendingDebugStart=true;
  if (inputReconnectTimer) { clearTimeout(inputReconnectTimer); inputReconnectTimer=null; }
  if (input) {
    input.onclose=null; input.onerror=null;
    try { input.close(); } catch {}
  }
  show('正在暂停旧控制页并接管桌面调试连接…');
  connectInput(true);
}
function menuInput(sources) {
  let toggle=false, confirm=false, back=false, axis=0;
  for (const source of sources) {
    const buttons=source.gamepad?.buttons || [], axes=source.gamepad?.axes || [];
    toggle ||= edge(`${source.handedness}-menu`,!!buttons[3]?.pressed);
    confirm ||= edge(`${source.handedness}-confirm`,!!buttons[0]?.pressed);
    if (source.handedness === 'right') back ||= edge('right-back',!!buttons[5]?.pressed);
    const candidate = axes.length >= 4 ? axes[3] : (axes[1] || 0);
    if (Math.abs(candidate) > Math.abs(axis)) axis=candidate;
  }
  if (toggle) { if (menu) closeMenu(); else openMainMenu(false); }
  if (!menu) return false;
  if (Math.abs(axis) < .35) axisReady=true;
  if (axisReady && Math.abs(axis) > .65) {
    menuIndex = (menuIndex + (axis > 0 ? 1 : -1) + menu.items.length) % menu.items.length;
    axisReady=false; panelDirty=true; publishMenuState();
  }
  if (back) backMenu(); else if (confirm) confirmMenu();
  return true;
}

function frame(time, xrFrame) {
  if (!session) return;
  session.requestAnimationFrame(frame);
  const viewer = xrFrame.getViewerPose(reference);
  const visible = session.visibilityState === 'visible' && viewer && !viewer.emulatedPosition;
  const sources = Array.from(session.inputSources || []).filter(source =>
    ['left','right'].includes(source.handedness) && source.gripSpace && source.gamepad?.mapping === 'xr-standard');
  const inMenu = menuInput(sources);
  if (!visible || inMenu || review) {
    if (wasVisible) sendNeutral();
    wasVisible=false;
  } else {
    wasVisible=true;
    const hands = {};
    for (const source of sources) {
      const buttons = source.gamepad.buttons;
      for (const [index,name] of [[4,source.handedness==='left'?'record':'home'],[5,source.handedness==='left'?'save':'reset']]) {
        if (edge(`${source.handedness}-shortcut-${index}`,!!buttons[index]?.pressed)) command(name);
      }
      const pose = xrFrame.getPose(source.gripSpace,reference);
      if (!pose || pose.emulatedPosition) continue;
      const {position:p,orientation:q} = pose.transform;
      hands[source.handedness] = {pose:[p.x,p.y,p.z,q.x,q.y,q.z,q.w],trigger:buttons[0]?.value || 0,squeeze:buttons[1]?.value || 0};
    }
    sendPosePacket(time, hands);
  }
  if(viewer) drawPanel(time,viewer);
}

document.querySelector('#enter').onclick = async () => {
  try {
    if (!vrSelected) { show('请先选择 VR 双手柄入口。'); return; }
    stopDesktopDebug();
    previousButtons = {};
    session = await navigator.xr.requestSession('immersive-vr',{optionalFeatures:['local-floor']});
    const canvas=document.querySelector('#xr'); gl=canvas.getContext('webgl',{xrCompatible:true,alpha:false});
    if(!gl) throw Error('WebGL unavailable');
    await gl.makeXRCompatible();
    session.updateRenderState({baseLayer:new XRWebGLLayer(session,gl)});
    try { reference=await session.requestReferenceSpace('local-floor'); screenOrigin=[0,1.4,-1.5]; }
    catch { reference=await session.requestReferenceSpace('local'); screenOrigin=[0,0,-1.5]; }
    reference.addEventListener('reset',sendNeutral);
    session.addEventListener('visibilitychange',() => { if(session.visibilityState!=='visible') sendNeutral(); });
    session.addEventListener('end',() => {sendNeutral();previousButtons={};session=null;});
    setupGL(); sendNeutral(); session.requestAnimationFrame(frame);
  } catch(error) { console.error(error); show('无法进入 VR，请确认头显浏览器已允许 WebXR，然后刷新页面重试。'); if(session) await session.end(); session=null; }
};
document.querySelector('#vr-select').onclick = async () => {
  if (!await selectMode('vr')) return;
  vrSelected = true;
  show('已选择 VR 双手柄；请点击“进入 VR”。');
  if (window.isSecureContext && navigator.xr &&
      await navigator.xr.isSessionSupported('immersive-vr')) document.querySelector('#enter').disabled=false;
};
document.querySelector('#leader-entry').onclick = async () => {
  if (!await selectMode('leader')) return;
  location.href = `/spectator#${new URLSearchParams({token})}`;
};
document.querySelector('#debug').onclick = () => {
  if (debugMode) stopDesktopDebug();
  else requestDesktopDebug();
};
const debugKeyCodes = new Set(['Digit1','Digit2','KeyW','KeyS','KeyA','KeyD','KeyR','KeyF',
  'KeyI','KeyK','KeyJ','KeyL','KeyU','KeyO','Space','ShiftLeft','ShiftRight']);
window.addEventListener('keydown', event => {
  if (!debugMode || !debugKeyCodes.has(event.code) || ['INPUT','SELECT','TEXTAREA','BUTTON'].includes(event.target.tagName)) return;
  event.preventDefault();
  if (event.code === 'Digit1') debugSelected='left';
  else if (event.code === 'Digit2') debugSelected='right';
  else debugKeys.add(event.code);
});
window.addEventListener('keyup', event => {
  if (!debugKeyCodes.has(event.code)) return;
  debugKeys.delete(event.code);
  if (debugMode) event.preventDefault();
});
window.addEventListener('blur', releaseDesktopDebugControls);
document.addEventListener('visibilitychange', () => { if (document.hidden) stopDesktopDebug(); });
document.querySelector('#task').onchange = refreshTaskDescription;
document.querySelector('#random-scene').onchange = event => command('set_random_scene', event.target.checked);
document.querySelector('#task-apply').onclick = () => {
  const task = document.querySelector('#task').value;
  const info = taskInfo(task);
  if (task === latestStatus.task) { show(`已经是当前任务：${task}`); return; }
  if (window.confirm(`切换到 ${info.title}？\n\n任务目标：${info.instruction}\n\n${info.success}`)) command('select_task', task);
};
document.querySelector('#review').onclick = () => { const episode=selectedEpisode(); if (episode) playEpisode(episode); };
document.querySelector('#takeover').onclick = () => {
  const episode=selectedEpisode();
  if (episode?.recoverable) command('takeover', episode.name);
  else show('所选轨迹没有经过校验的末状态，只能播放视频。');
};
document.querySelector('#delete-selected').onclick = () => {
  const episode=selectedEpisode();
  if (episode && window.confirm(`删除 ${episode.name}？`)) command('delete_episode', episode.name);
};
document.querySelector('#exit').onclick = exitPage;
window.addEventListener('pagehide',() => { stopDesktopDebug(false); sendNeutral(); });
setInterval(() => {
  if (input?.readyState === WebSocket.OPEN) input.send(JSON.stringify({type:'ping',client_ms:Date.now()}));
}, 2000);
setInterval(() => {
  if (video?.readyState === WebSocket.OPEN && performance.now() - lastVideoMessage > 3000) video.close();
}, 500);
connect();
