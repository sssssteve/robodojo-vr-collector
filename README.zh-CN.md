# RoboDojo VR Collector

[English](README.md) | **简体中文**

面向 RoboDojo 双臂仿真的浏览器 WebXR 遥操作与示范数据采集工具。

工作站负责运行 Isaac Sim、RoboDojo、控制、渲染和 HDF5 录制；输入可选择 WebXR 手柄或只读 CAN 的 Piper 实体双主臂，电脑页面负责监看和模式控制。

> 本项目是研究与内部数采工具，不是通用产品。当前验证尚未覆盖全部任务、头显或真实机器人，详见 [VALIDATION.md](VALIDATION.md)。

## 环境要求

- Ubuntu Linux 与 NVIDIA GPU
- Isaac Sim 5.1 和 RoboDojo/Isaac Lab
- Miniconda、Git LFS、`adb` 和 `tmux`
- 一台通过 USB 连接、支持 WebXR 的头显，例如 Meta Quest 或 Pico

仓库不包含 RoboDojo 资产、Isaac Sim、采集数据或机器凭据。

## 快速开始

```bash
git clone https://github.com/YOUR_ACCOUNT/robodojo-vr-collector.git
cd robodojo-vr-collector

git clone --recurse-submodules \
  https://github.com/RoboDojo-Benchmark/RoboDojo.git external/RoboDojo
git -C external/RoboDojo checkout e0703b03bb1af6075400e9d60dc17a792793960c
git -C external/RoboDojo submodule update --init --recursive
git -C external/RoboDojo apply ../../patches/robodojo_piper_runtime.patch

cd external/RoboDojo
bash scripts/install.sh
bash scripts/init_assets.sh
cd ../..

cp config.example.env config.env
conda activate RoboDojo
python -m pip install -r requirements.txt
./start_desktop.sh
```

只有当 RoboDojo、Isaac Lab、输出目录或 `adb` 位于其他位置时，才需要修改 `config.env`。

启动前应解锁头显、允许 USB 调试，并确认 `adb devices -l` 中只有一台目标设备处于 `device` 状态。

## 采集流程

1. 等待页面显示 `ready`，然后进入 VR。
2. 先松开双手侧握键，再按住侧握键控制对应机械臂。
3. 重置到标准初态。
4. 开始录制，完成任务后保存。
5. 失败或调试片段不得标记为成功示范。

| 输入 | 功能 |
|---|---|
| 左右侧握键 | 控制对应机械臂 |
| 左右扳机 | 控制对应夹爪 |
| 左手 X / 左手 Y | 开始录制 / 保存 |
| 右手 A / 右手 B | 暂停或重新对齐 / 重置或返回 |
| 按下任一摇杆 | 打开或关闭任务菜单 |

更换头显型号或浏览器后，应先在非录制状态核对实际按键映射。

## 数据

每条轨迹保存为原始 HDF5，包含三路相机、14D 双臂状态与动作、控制命令、时间戳以及任务和 layout 元数据。

```bash
./launch.sh --inspect data/episode_xxx.hdf5
```

将完整、成功且通过质量检查的轨迹导出为 RoboDojo 官方 HDF5 字段结构：

```bash
python export_official_hdf5.py data/episode_xxx.hdf5 data/episode_xxx_official.hdf5
```

合格数据要求：三路相机可解码、状态和动作逐帧对齐、保留真实时间戳，并且 recorder 无 backpressure。超过 200 ms 的真实时间间隔以及缺失控制输入的 transition 会被记录供验收复核，但不会仅凭这些项目自动拒绝操作员保存的轨迹。调试和合成数据只能放入 `validation/`，不得混入 `data/`。

运行时 token、日志、PID、TLS 材料、场景状态、外部资产和采集数据均不得提交到仓库。

## 文档

- [验证证据与已知限制](VALIDATION.md)
- [机器配置示例](config.example.env)

## 更新记录

每次修改代码、配置或文档，都必须在这里记录日期、具体改动、实际验证以及仍未验证的内容。

### 2026-09-27 · 实体双主臂与顶视角更新

- 增加互斥的 VR 与 Piper 实体双主臂模式、只读 `can0`/`can1` 输入、电脑键盘录制控制，以及实体录制的双侧输入完整性检查。
- 增加独立顶视相机监看流，并让传送带 surface/graph 与录制门控同步启停，同时保留已有的单手追踪安全逻辑。
- 验证：候选版本在 Piper 独立目录通过 51 个 CPU/协议测试，并通过 Python 编译、JavaScript 与 Shell 语法检查。
- 剩余边界：本次未重启 Isaac Sim，未连接真实 CAN 主臂，也未执行头显端到端采集。

### 2026-09-26 · 追踪与监看更新

- 增加单手 150 ms 追踪宽限。单手丢失超过宽限后只释放对应机械臂，并要求该手松开侧握键后才能恢复；仍在追踪的另一侧可以继续操作。
- 重做电脑监看页面，增加仿真画面全屏、录制与运行状态浮层、实时质量预警、任务成功标准和最近 5 条已验收轨迹。
- 将缺失控制输入的 transition 写入 HDF5 元数据，不再仅凭这一项拒绝已保存轨迹；相关计数仍需在验收时复核。
- 启用 Isaac 传送带扩展，并在传送带任务等待录制期间持续刷新预览 observation。
- 验证：候选版本在 Piper 独立目录通过 42 个 CPU/协议测试，并通过 Python 编译、JavaScript 与 Shell 语法检查。本次未重启 Isaac Sim，也未执行头显端到端采集。

### 2026-09-26

- 支持带脚本辅助机械臂的任务，并在录制开始前冻结相关排序、Kong、井字棋和传送带任务的自动运动。
- 在头显和电脑监看页面增加空间重建进度、自动运动状态，以及控制通道断开时无法开始录制的明确提示。
- 为孤立的 partial 轨迹保留 episode 编号，并在导出官方格式时保留源 HDF5 的分块和压缩设置。
- 将超过 200 ms 的真实时间间隔从自动拒绝条件改为需要人工复核的验收证据。
- 验证：候选版本在 Piper 独立目录通过 40 个 CPU/协议测试，并通过 Python 编译、JavaScript 与 Shell 语法检查。本次未重启 Isaac Sim，也未执行头显端到端采集。

### 2026-09-25

- 对齐当前 Piper 数采代码：同步 `run.py`、`watchdog.py` 及对应测试，增加受 watchdog 管理的 CPU/CUDA 任务设备切换和同进程任务重载。
- 合并 Mac 工作区较新的 Meta WebXR/菜单与 PC 监看实现，并加入质量门控的官方 HDF5 导出。Piper 的测试已引用这些文件，但活动 Web 目录此前尚未同步。
- 部署脚本继续通过 `config.env` 保持可移植；没有把 Piper 的机器绝对路径和运行状态带入仓库。
- 验证：准备发布的 Mac 包已在 Piper 独立临时目录通过 37 个 CPU/协议测试，以及 Python 编译、JavaScript 和 Shell 语法检查。
- 剩余边界：本次同步期间独立的 `shucai` 主机不可达，且未重启 Isaac Sim 或头显会话。
- 精简中英文 README，只保留环境配置、采集流程、数据验收和证据边界。
- 从项目首页移除重复的部署细节、任务计划和历史验证结果；验证证据仍保留在 `VALIDATION.md`。
- 文档检查：已核对语言链接、相对链接、代码块、固定 commit、与端口无关的命令及中英文章节一致性。
- 未重新运行：单元测试、Isaac Sim 和头显端到端采集；本次只修改文档。

### 2026-09-24

- 增加中英文文档，并为 GitHub 发布整理代码包。
- 清理机器身份信息，扩充数据、凭据、外部依赖和运行时文件的忽略规则。
