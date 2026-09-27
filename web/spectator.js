const token = new URLSearchParams(location.hash.slice(1)).get('token') || '';
const base = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;
const statusElement = document.querySelector('#status');
const promptElement = document.querySelector('#prompt');
const recordingElement = document.querySelector('#recording');
const phaseElement = document.querySelector('#phase');
const taskCountElement = document.querySelector('#task-count');
const totalCountElement = document.querySelector('#total-count');
const frameCountElement = document.querySelector('#frame-count');
const qualityPanelElement = document.querySelector('#quality-panel');
const qualityBadgeElement = document.querySelector('#quality-badge');
const episodeProgressElement = document.querySelector('#episode-progress');
const qualityHzElement = document.querySelector('#quality-hz');
const qualityInputElement = document.querySelector('#quality-input');
const qualityWriterElement = document.querySelector('#quality-writer');
const qualityReasonsElement = document.querySelector('#quality-reasons');
const successTextElement = document.querySelector('#success-text');
const recentListElement = document.querySelector('#recent-list');
const viewportElement = document.querySelector('#viewport');
const fullscreenButtonElement = document.querySelector('#fullscreen-button');
const fullscreenRecordingElement = document.querySelector('#fullscreen-recording');
const fullscreenModeElement = document.querySelector('#fullscreen-mode');
const fullscreenPhaseElement = document.querySelector('#fullscreen-phase');
const fullscreenTaskCountElement = document.querySelector('#fullscreen-task-count');
const fullscreenTotalCountElement = document.querySelector('#fullscreen-total-count');
const leaderStartElement = document.querySelector('#leader-start');
const vrModeElement = document.querySelector('#vr-mode');
const leaderStopElement = document.querySelector('#leader-stop');
const leaderStateElement = document.querySelector('#leader-state');
const shortcutStateElement = document.querySelector('#shortcut-state');
const menuElement = document.querySelector('#menu');
const menuTitleElement = document.querySelector('#menu-title');
const menuBodyElement = document.querySelector('#menu-body');
const menuItemsElement = document.querySelector('#menu-items');
const menuHelpElement = document.querySelector('#menu-help');
const canvas = document.querySelector('#preview');
const context = canvas.getContext('2d');
const topCanvas = document.querySelector('#top-observer');
const topContext = topCanvas.getContext('2d');
const topLabel = document.querySelector('#top-label');
let socket, reconnectTimer, statusReceivedAt = 0, transferMs = 0, decodeMs = 0;
let topSocket, topReconnectTimer, topFrame = 0, topDrawn = 0, topLastMessage = 0;
let frameNumber = 0, drawnFrame = 0, lastMessage = 0;
let pendingFrame = null, decodeRunning = false;
let envEpoch = null;
let latestPhase = 'starting', latestMode = 'unselected', latestStatus = {}, recordShortcutPending = false;
let physicalMenu = null, physicalSelected = 0;

async function physicalCommand(command, value) {
  const response=await fetch(`/physical-command?token=${encodeURIComponent(token)}`, {
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command,value})});
  const result=await response.json();
  if (!result.accepted) throw new Error(result.error || String(response.status));
}

function physicalItems() {
  if (physicalMenu === 'tasks') return [...(latestStatus.tasks || []).map(task => ({
    label:`${task === latestStatus.task ? '●' : '○'} ${task}`, command:'select_task', value:task
  })), {label:'返回', back:true}];
  return [
    {label:latestPhase === 'recording' ? '保存当前录制' : '开始普通录制', command:'physical_record_toggle'},
    {label:'双臂归位', command:'home'},
    {label:'重置当前场景', command:'reset'},
    {label:'选择任务', menu:'tasks'},
    {label:'更换下一份官方布局', command:'next_scene'},
    {label:'丢弃当前录制', command:'discard'},
  ];
}

function renderPhysicalMenu() {
  if (!physicalMenu) { menuElement.hidden=true; menuItemsElement.replaceChildren(); return; }
  const items=physicalItems();
  physicalSelected=Math.min(physicalSelected,items.length-1);
  menuElement.hidden=false;
  menuTitleElement.textContent=physicalMenu === 'tasks' ? '选择 RoboDojo 任务' : '实体双主臂数采菜单';
  menuBodyElement.textContent=physicalMenu === 'tasks' ? '切换任务前请先保存或丢弃当前录制。' :
    '实体双主臂持续遥操；空格确认，Esc 返回。';
  menuHelpElement.textContent='↑/↓ 选择 · 空格确认 · Esc 返回 · Q 打开/关闭';
  menuItemsElement.replaceChildren(...items.map((item,index) => {
    const li=document.createElement('li');
    li.textContent=`${index === physicalSelected ? '› ' : ''}${item.label}`;
    li.classList.toggle('selected',index === physicalSelected);
    return li;
  }));
  menuItemsElement.querySelector('.selected')?.scrollIntoView({block:'nearest'});
}

window.addEventListener('keydown', async event => {
  if (latestMode !== 'leader' || event.repeat || event.altKey || event.ctrlKey || event.metaKey ||
      ['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName)) return;
  if (!['KeyQ','ArrowUp','ArrowDown','Space','Escape','KeyR','KeyE'].includes(event.code)) return;
  event.preventDefault();
  if (event.code === 'KeyQ') {
    physicalMenu=physicalMenu ? null : 'main'; physicalSelected=0; renderPhysicalMenu();
    return;
  }
  if (physicalMenu) {
    if (event.code === 'Escape') {
      physicalMenu=physicalMenu === 'tasks' ? 'main' : null;
      physicalSelected=0; renderPhysicalMenu(); return;
    }
    const items=physicalItems();
    if (event.code === 'ArrowUp' || event.code === 'ArrowDown') {
      physicalSelected=(physicalSelected+(event.code === 'ArrowDown' ? 1 : -1)+items.length)%items.length;
      renderPhysicalMenu(); return;
    }
    if (event.code !== 'Space') return;
    const item=items[physicalSelected];
    if (item.back) { physicalMenu='main'; physicalSelected=0; renderPhysicalMenu(); return; }
    if (item.menu) { physicalMenu=item.menu; physicalSelected=0; renderPhysicalMenu(); return; }
    if (item.command === 'discard' && latestPhase !== 'recording') return;
    if (item.command === 'select_task' && item.value === latestStatus.task) return;
    physicalMenu=null; renderPhysicalMenu();
    await sendPhysical(item.command,item.value);
    return;
  }
  const command={Space:'physical_record_toggle',Escape:'discard',KeyR:'reset',KeyE:'home'}[event.code];
  if (!command || (command === 'discard' && latestPhase !== 'recording')) return;
  await sendPhysical(command);
});

async function sendPhysical(command,value) {
  if (!['ready','recording'].includes(latestPhase) || recordShortcutPending) return;
  recordShortcutPending=true;
  let waitForPhase=command === 'physical_record_toggle' || command === 'discard';
  shortcutStateElement.textContent=`正在执行：${command === 'physical_record_toggle' ?
    (latestPhase === 'recording' ? '保存录制' : '开始录制') : command}…`;
  try {
    await physicalCommand(command,value);
  } catch (error) {
    shortcutStateElement.textContent=`操作失败：${error.message}`;
    waitForPhase=false;
  } finally {
    if (waitForPhase) setTimeout(() => { recordShortcutPending=false; },6000);
    else recordShortcutPending=false;
  }
}

async function modeAction(mode) {
  leaderStartElement.disabled=true;
  leaderStopElement.disabled=true;
  vrModeElement.disabled=true;
  try {
    const response=await fetch(`/mode?token=${encodeURIComponent(token)}`, {
      method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})});
    const result=await response.json();
    if (!result.accepted) leaderStateElement.textContent=`操作失败：${result.error || response.status}`;
  } catch (error) {
    leaderStateElement.textContent=`连接失败：${error.message}`;
  }
}
vrModeElement.addEventListener('click', () => modeAction('vr'));
leaderStartElement.addEventListener('click', () => modeAction('leader'));
leaderStopElement.addEventListener('click', () => modeAction('unselected'));

function renderLeader(msg) {
  const leader=msg.leader_can || {};
  const mode=msg.control_mode || 'unselected';
  const switchingAllowed=msg.phase === 'ready';
  leaderStartElement.disabled=!switchingAllowed || mode === 'leader';
  vrModeElement.disabled=!switchingAllowed || mode === 'vr';
  leaderStopElement.disabled=!switchingAllowed || mode === 'unselected';
  const modeLabel={unselected:'未选择',vr:'VR 双手柄',leader:'实体双主臂'}[mode] || mode;
  fullscreenModeElement.textContent=`控制入口：${modeLabel}`;
  if (mode !== 'leader') {
    leaderStateElement.textContent=`当前：${modeLabel} · 来源 ${msg.input?.active_source || '等待输入'}`;
    return;
  }
  const ready=leader.ready || {}, ages=leader.joint_age_ms || {};
  const side=which => `${which === 'left' ? '左' : '右'} ${ready[which] ? '就绪' : '待输入'}（${ages[which] ?? '-'} ms）`;
  leaderStateElement.textContent=leader.error ? `实体双主臂 CAN 错误：${leader.error}` :
    `当前：${modeLabel} · ${side('left')} · ${side('right')} · 输入帧 ${leader.frames || 0}`;
}

function syncFullscreenButton() {
  fullscreenButtonElement.textContent=document.fullscreenElement ? '退出全屏' : '⛶ 全屏仿真视角';
}

fullscreenButtonElement.addEventListener('click', async () => {
  if (document.fullscreenElement) await document.exitFullscreen();
  else await viewportElement.requestFullscreen();
});
document.addEventListener('fullscreenchange', syncFullscreenButton);

function renderQuality(msg, input, cycle, writer) {
  const frames=Number(msg.frames || 0);
  const wallHz=Number(msg.wall_hz || 0);
  const previewAge=Number(msg.preview_transport?.age_ms ?? 0);
  const isRecording=msg.phase === 'recording';
  const reasons=[];
  let level='ok';
  const warn=text => { reasons.push(text); if (level === 'ok') level='warning'; };
  const error=text => { reasons.push(text); level='error'; };

  if (writer.writer_error) error('录制写入器异常，请停止并检查本条数据。');
  if (isRecording && (!msg.input_fresh || input.timed_out)) error('控制输入已超时。');
  if (writer.backpressure) warn(`写入队列出现背压 ${writer.queue_depth || 0}/${writer.queue_capacity || 0}。`);
  if (isRecording && (input.seq_gaps || 0) > 0) warn(`本次连接累计发现 ${input.seq_gaps} 个输入序号间隙。`);
  if (isRecording && (!input.left_tracking || !input.right_tracking)) warn('左右输入未同时保持 tracking。');
  if (isRecording && (msg.ik_failures || 0) > 0) warn(`累计 IK 失败 ${msg.ik_failures} 次。`);
  if (isRecording && previewAge > 1000) warn(`监看画面已滞后 ${Math.round(previewAge)} ms。`);
  if (isRecording && (cycle.p95 || 0) > 200) warn(`控制周期 P95 为 ${cycle.p95} ms。`);

  qualityPanelElement.dataset.level=level;
  qualityBadgeElement.textContent=isRecording ? {ok:'状态良好',warning:'需要关注',error:'异常'}[level] : '待录制';
  episodeProgressElement.textContent=`${frames} 帧 · ${(frames / 25).toFixed(1)} 秒`;
  qualityHzElement.textContent=`${wallHz || 0} Hz`;
  qualityInputElement.textContent=msg.input_fresh ? `新鲜 · ${input.effective_age_ms ?? '-'} ms` : '未就绪';
  qualityWriterElement.textContent=writer.writer_alive ? `正常 · ${writer.queue_depth || 0}/${writer.queue_capacity || 0}` : (isRecording ? '未运行' : '待录制');
  qualityReasonsElement.replaceChildren(...reasons.map(text => {
    const item=document.createElement('li'); item.textContent=text; return item;
  }));
}

function renderRecent(episodes) {
  const recent=(episodes || []).slice(0,5);
  if (!recent.length) {
    const item=document.createElement('li'); item.className='empty'; item.textContent='当前任务暂无已验收记录';
    recentListElement.replaceChildren(item);
    return;
  }
  recentListElement.replaceChildren(...recent.map(episode => {
    const item=document.createElement('li'); item.className='recent-item';
    const title=document.createElement('b'); title.textContent=episode.name.replace('.hdf5','');
    const result=document.createElement('span'); result.className=episode.success ? 'recent-ok' : 'recent-fail';
    result.textContent=`${episode.success ? '成功' : '未成功'} · ${episode.frames} 帧`;
    item.append(title,result); return item;
  }));
}

function renderVrMenu(ui) {
  if (!ui?.open) {
    menuElement.hidden=true;
    menuItemsElement.replaceChildren();
    return;
  }
  menuElement.hidden=false;
  menuHelpElement.textContent='摇杆上下选择 · 扳机确认 · B 返回 · 按下摇杆关闭菜单';
  menuTitleElement.textContent=ui.title || 'VR 菜单';
  menuBodyElement.textContent=(ui.body || []).join('\n');
  menuItemsElement.replaceChildren(...(ui.items || []).map((label,index) => {
    const item=document.createElement('li');
    item.textContent=`${index === ui.selected ? '› ' : ''}${label}`;
    item.classList.toggle('selected',index === ui.selected);
    return item;
  }));
  menuItemsElement.querySelector('.selected')?.scrollIntoView({block:'nearest'});
}

function connect() {
  if (!token) { statusElement.textContent='缺少连接凭证。'; return; }
  socket = new WebSocket(`${base}/video?token=${encodeURIComponent(token)}`);
  socket.binaryType='blob';
  socket.onopen=() => {
    lastMessage=performance.now();
    recordingElement.dataset.state='starting';
    recordingElement.textContent='○ 已连接，等待录制状态';
  };
  socket.onmessage=async event => {
    lastMessage=performance.now();
    if (typeof event.data === 'string') {
      const msg=JSON.parse(event.data);
      if (envEpoch !== null && msg.env_epoch !== envEpoch) {
        pendingFrame=null; context.fillStyle='#080e15'; context.fillRect(0,0,1280,720);
        topContext.clearRect(0,0,320,240);
        topLabel.textContent='桌面顶视 · 等待新场景画面';
      }
      envEpoch=msg.env_epoch;
      statusReceivedAt=performance.now();
      const phase={starting:'启动中',ready:'已就绪',recording:'录制中'}[msg.phase] || msg.phase;
      const input=msg.input || {}, cycle=msg.cycle_ms || {}, writer=msg.recording_writer || {};
      if (msg.phase !== latestPhase) recordShortcutPending=false;
      latestPhase=msg.phase;
      latestMode=msg.control_mode || 'unselected';
      latestStatus=msg;
      if (latestMode !== 'leader') physicalMenu=null;
      if (!recordShortcutPending) shortcutStateElement.textContent=msg.phase === 'recording' ?
        (latestMode === 'leader' ? '空格保存 · Esc 丢弃 · Q 菜单' : '录制中（VR 手柄控制）') :
        (msg.phase !== 'ready' ? '等待运行状态' :
          latestMode === 'unselected' ? '请先选择控制入口' :
          latestMode === 'leader' ? '空格开始 · Q 菜单 · R 重置 · E 归位' : 'VR 手柄控制录制');
      const counts=msg.episode_counts || {};
      const taskCount=Number(counts[msg.task] || 0);
      const totalCount=Object.values(counts).reduce((sum,value) => sum + Number(value || 0),0);
      phaseElement.textContent=phase || '未知';
      taskCountElement.textContent=`${taskCount} 条`;
      totalCountElement.textContent=`${totalCount} 条`;
      frameCountElement.textContent=`${msg.frames || 0} 帧`;
      fullscreenPhaseElement.textContent=`运行状态：${phase || '未知'}`;
      fullscreenTaskCountElement.textContent=`当前任务已验收：${taskCount} 条`;
      fullscreenTotalCountElement.textContent=`全部已验收：${totalCount} 条`;
      if (latestMode === 'leader') renderPhysicalMenu();
      else renderVrMenu(msg.vr_ui);
      renderLeader(msg);
      if (writer.writer_error) {
        recordingElement.dataset.state='error';
        recordingElement.textContent='● 录制写入异常';
        fullscreenRecordingElement.dataset.state='error';
        fullscreenRecordingElement.textContent='● 录制写入异常';
      } else if (msg.phase === 'recording') {
        recordingElement.dataset.state='recording';
        recordingElement.textContent=`● 正在录制 · ${msg.frames || 0} 帧`;
        fullscreenRecordingElement.dataset.state='recording';
        fullscreenRecordingElement.textContent=`● 录制中 · ${msg.frames || 0} 帧`;
      } else {
        recordingElement.dataset.state='idle';
        recordingElement.textContent='○ 当前未录制';
        fullscreenRecordingElement.dataset.state='idle';
        fullscreenRecordingElement.textContent='○ 未录制';
      }
      const step=msg.step_profile || {}, obs=msg.observation_profile || {};
      renderQuality(msg,input,cycle,writer);
      successTextElement.textContent=msg.task_info?.success || '等待任务成功标准…';
      renderRecent(msg.episodes);
      statusElement.textContent=`${msg.task || ''} · ${phase} · epoch ${msg.env_epoch ?? '-'} · tick ${msg.sim_tick ?? '-'} · ${msg.wall_hz || 0} Hz\n`+
        `录制 ${msg.phase === 'recording' ? '是' : '否'} · 当前 ${msg.frames || 0} 帧 · 当前任务已验收 ${taskCount} 条 · 全部任务已验收 ${totalCount} 条\n`+
        `输入 ${input.state || '未知'} · 收到/应用 ${input.received_seq ?? '-'}/${input.applied_seq ?? '-'} · 年龄 ${input.effective_age_ms ?? '-'} ms · tracking L/R ${input.left_tracking ? '有' : '无'}/${input.right_tracking ? '有' : '无'}\n`+
        `遥操 ${msg.teleop_allowed ? '允许' : '保持'} · 原因 ${msg.hold_reason || '-'} · IK ${JSON.stringify(msg.ik_status || {})}\n`+
        `周期 ms 当前/p50/p95/p99/max ${cycle.current ?? '-'}/${cycle.p50 ?? '-'}/${cycle.p95 ?? '-'}/${cycle.p99 ?? '-'}/${cycle.max ?? '-'}\n`+
        `IK ${step.solve_ms ?? '-'} · physics ${step.physics_ms ?? '-'} · render ${obs.render_ms ?? '-'} (${obs.render_calls ?? '-'}次) · capture ${obs.capture_ms ?? '-'} · writer ${writer.queue_depth ?? 0}/${writer.queue_capacity ?? 0}${writer.backpressure ? ' 背压' : ''}\n`+
        `预览帧 ${msg.preview_transport?.frame_seq ?? '-'} · 年龄 ${msg.preview_transport?.age_ms ?? '-'} ms · 传输 ${Math.round(transferMs)} ms · 解码 ${Math.round(decodeMs)} ms · 生命周期 ${msg.lifecycle?.phase || msg.operation?.phase || phase}`;
      promptElement.textContent=`任务：${msg.task_info?.prompt_zh || '等待任务 Prompt…'}`;
      return;
    }
    const currentFrame=++frameNumber;
    pendingFrame={blob:event.data,frame:currentFrame,transfer:Math.max(0,performance.now()-statusReceivedAt)};
    if (!decodeRunning) decodeLatest();
  };
  socket.onclose=() => {
    statusElement.textContent='画面连接已断开，正在重连…';
    recordingElement.dataset.state='error';
    recordingElement.textContent='● 监看连接已断开';
    fullscreenRecordingElement.dataset.state='error';
    fullscreenRecordingElement.textContent='● 监看连接已断开';
    fullscreenPhaseElement.textContent='运行状态：连接断开';
    phaseElement.textContent='连接断开';
    physicalMenu=null; renderVrMenu(null);
    if (!reconnectTimer) reconnectTimer=setTimeout(() => { reconnectTimer=null; connect(); },1500);
  };
}
async function decodeLatest() {
  decodeRunning=true;
  while (pendingFrame) {
    const item=pendingFrame; pendingFrame=null;
    const started=performance.now();
    const bitmap=await createImageBitmap(item.blob);
    if (item.frame > drawnFrame) {
      context.drawImage(bitmap,0,0,1280,720);
      context.fillStyle='#fff'; context.font='24px sans-serif';
      context.fillText('左夹爪',20,225); context.fillText('头部',610,105); context.fillText('右夹爪',1175,225);
      drawnFrame=item.frame; transferMs=item.transfer; decodeMs=performance.now()-started;
      canvas.dataset.frame=String(item.frame);
    }
    bitmap.close();
  }
  decodeRunning=false;
}
connect();
function connectTopObserver() {
  if (!token) return;
  topSocket = new WebSocket(`${base}/observer?token=${encodeURIComponent(token)}`);
  topSocket.binaryType='blob';
  topSocket.onopen=() => { topLastMessage=performance.now(); };
  topSocket.onmessage=async event => {
    topLastMessage=performance.now();
    const frame=++topFrame;
    const bitmap=await createImageBitmap(event.data);
    if (frame > topDrawn) {
      topContext.drawImage(bitmap,0,0,320,240);
      topDrawn=frame;
      topCanvas.dataset.frame=String(frame);
      topLabel.textContent='桌面顶视 · 实时';
    }
    bitmap.close();
  };
  topSocket.onclose=() => {
    topLabel.textContent='桌面顶视 · 连接中';
    if (!topReconnectTimer) topReconnectTimer=setTimeout(() => { topReconnectTimer=null; connectTopObserver(); },1500);
  };
}
connectTopObserver();
setInterval(() => { if (socket?.readyState===WebSocket.OPEN && performance.now()-lastMessage>3000) socket.close(); },500);
setInterval(() => { if (topSocket?.readyState===WebSocket.OPEN && performance.now()-topLastMessage>3000 && topLastMessage) topSocket.close(); },500);
