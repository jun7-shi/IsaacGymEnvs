# train_play.py - 训练与可视化分离模式

## 概述

`train_play.py` 是一个增强版的训练脚本，支持同时运行训练进程和可视化进程。这样可以在headless模式下高效训练，同时在另一个窗口实时查看训练效果。

## 环境要求

运行前需要激活rlgpu环境：
```bash
mamba activate rlgpu
# 或 conda activate rlgpu
```

## 主要特性

1. **训练模式 (mode=train)**
   - Headless运行，性能最优
   - 定期保存checkpoint到固定文件名（默认：`runs/<task_name>/nn/latest.pth`）
   - 支持所有原有的训练功能

2. **可视化模式 (mode=play)**
   - 非headless运行，实时显示
   - 使用少量环境（默认4个）减少渲染负担
   - 自动等待训练进程生成第一个checkpoint
   - 定期重新加载最新checkpoint，实时展示训练进展

## 使用方法

### 快速开始

**1. 启动训练进程（终端1）：**
```bash
cd /path/to/IsaacGymEnvs
mamba activate rlgpu
python isaacgymenvs/train_play.py task=Ant mode=train
```

**2. 启动可视化进程（终端2）：**
```bash
cd /path/to/IsaacGymEnvs
mamba activate rlgpu
# 需要指定训练进程的experiment目录名（从训练输出获取）
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant_2025-01-14_10-30-45

# 或者使用部分名称，脚本会自动找到最新的匹配目录
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant

# 推荐：启动时自动打开控制面板（只需2个终端）
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant enable_control_panel=True
```

**方式二：单独启动Checkpoint控制面板（需要3个终端）：**
```bash
cd /path/to/IsaacGymEnvs
mamba activate rlgpu
# 如果没有使用enable_control_panel=True，可以单独启动控制面板
python isaacgymenvs/checkpoint_control_panel.py --experiment=Ant
```

### 进阶配置

#### 训练模式参数

```bash
# 基础训练
python isaacgymenvs/train_play.py task=Ant mode=train

# 自定义checkpoint保存频率（每50个epoch保存一次）
python isaacgymenvs/train_play.py task=Ant mode=train checkpoint_save_freq=50

# 指定实验名称
python isaacgymenvs/train_play.py task=Ant mode=train experiment=my_ant_experiment

# 修改环境数量
python isaacgymenvs/train_play.py task=Ant mode=train num_envs=8192

# 多GPU训练
torchrun --standalone --nnodes=1 --nproc_per_node=2 isaacgymenvs/train_play.py task=Ant mode=train multi_gpu=True
```

#### 可视化模式参数

```bash
# 基础可视化（必须指定experiment参数）
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant_2025-01-14_10-30-45

# 使用部分名称自动匹配最新目录
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant

# 自定义可视化环境数量（1-4个推荐）
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant num_envs=2

# 自定义checkpoint重载间隔（秒）
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant checkpoint_reload_interval=60

# 匹配训练进程的自定义实验名称
python isaacgymenvs/train_play.py task=Ant mode=play experiment=my_ant_experiment
```

### 完整示例

#### 示例1：Ant训练 + 可视化

**终端1 - 训练：**
```bash
python isaacgymenvs/train_play.py task=Ant mode=train checkpoint_save_freq=100 num_envs=4096
# 训练会输出类似: Training will save to: runs/Ant_2025-01-14_10-30-45
```

**终端2 - 可视化：**
```bash
# 使用训练输出的目录名，或使用部分名称自动匹配
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant num_envs=4 checkpoint_reload_interval=30
```

#### 示例2：Humanoid训练 + 可视化（自定义实验名）

**终端1 - 训练：**
```bash
python isaacgymenvs/train_play.py task=Humanoid mode=train experiment=humanoid_v1 checkpoint_save_freq=50
# 训练会输出类似: Training will save to: runs/humanoid_v1_2025-01-14_10-30-45
```

**终端2 - 可视化：**
```bash
# 使用自定义实验名的部分名称即可自动匹配
python isaacgymenvs/train_play.py task=Humanoid mode=play experiment=humanoid_v1 num_envs=2
```

#### 示例3：使用WandB追踪

**终端1 - 训练：**
```bash
python isaacgymenvs/train_play.py task=Ant mode=train \
    wandb_activate=True \
    wandb_entity=your_entity \
    wandb_project=isaacgym_training \
    checkpoint_save_freq=100
# 训练会输出类似: Training will save to: runs/Ant_2025-01-14_10-30-45
```

**终端2 - 可视化：**
```bash
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant num_envs=4
```

## 配置参数说明

### 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mode` | `train` | 运行模式：`train`（训练）或 `play`（可视化） |
| `checkpoint_save_freq` | `100` | 训练模式下每N个**epoch**（训练迭代周期）保存一次checkpoint。例如100表示在第100、200、300...个epoch时保存 |
| `checkpoint_reload_interval` | `30` | 可视化模式下每N**秒**检查并重载checkpoint |

**重要说明：**
- `checkpoint_save_freq` 的单位是 **epoch**（训练迭代），不是秒或步数
- 一个epoch通常包含多个训练步骤，具体取决于环境和批次大小
- 如果想更频繁地更新可视化，可以设置更小的值，如 `checkpoint_save_freq=50` 或 `checkpoint_save_freq=20`

### 自动设置的参数

根据 `mode` 的值，以下参数会被自动设置：

**训练模式 (mode=train):**
- `headless=True` - 无渲染，最大性能
- `test=False` - 训练模式

**可视化模式 (mode=play):**
- `headless=False` - 启用渲染
- `test=True` - 推理模式
- `num_envs=4`（如果未指定）- 少量环境用于可视化

## Checkpoint管理

### Checkpoint存储位置

训练模式checkpoint保存在：`runs/<experiment_name>_<timestamp>/nn/latest.pth`

- `<experiment_name>` 默认为任务名称（如 `Ant`）,会自动添加时间戳
- `<timestamp>` 格式为 `YYYY-MM-DD_HH-MM-SS`
- 可通过 `experiment=` 参数自定义实验名称

### Checkpoint保存策略

**训练模式：**
- 每 `checkpoint_save_freq` 个epoch保存一次
- 覆盖写入 `latest.pth`，确保可视化进程始终加载最新模型
- 原有的checkpoint保存机制（rl_games自带）仍然正常工作

**可视化模式：**
- 必须指定 `experiment` 参数来指定要可视化的训练运行
- 如果使用部分名称（如 `experiment=Ant`），会自动找到最新匹配的目录
- 启动时等待checkpoint出现（如果不存在）
- 每 `checkpoint_reload_interval` 秒检查checkpoint是否更新
- 如果文件有更新，自动重新加载并继续可视化

## 视角控制

可视化时可以使用以下方式调整相机视角：

### 鼠标控制
- **左键拖动** - 旋转相机视角
- **中键拖动**（滚轮按下） - 平移相机位置
- **滚轮滚动** - 缩放（拉近/拉远）

### 键盘控制
- **W/S** - 前进/后退
- **A/D** - 左移/右移
- **Q/E** - 上升/下降
- **ESC** - 退出可视化

**提示**：如果启动时视角离得太远，使用鼠标滚轮拉近，然后用鼠标左键调整到合适的角度。

## Checkpoint控制面板

### 功能说明

Checkpoint控制面板是一个可选的GUI工具，允许您在可视化过程中灵活切换不同的checkpoint模型。

**主要特性：**
- 图形化界面，操作简单直观
- 实时显示所有可用的checkpoint文件
- 支持两种模式：
  - **自动模式**：始终加载最新的`latest.pth`（跟随训练进度）
  - **手动模式**：选择任意已保存的checkpoint进行查看
- 自动刷新checkpoint列表（每5秒）
- 显示checkpoint文件大小和修改时间

### 使用方法

**方式一：集成启动（推荐，只需2个终端）**

```bash
# 终端1 - 启动训练
python isaacgymenvs/train_play.py task=Ant mode=train

# 终端2 - 启动可视化和控制面板
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant enable_control_panel=True
```

控制面板会自动在后台启动，弹出GUI窗口。

**方式二：单独启动（需要3个终端）**

如果您想手动控制控制面板的启动时机：

```bash
# 终端1 - 启动训练
python isaacgymenvs/train_play.py task=Ant mode=train

# 终端2 - 启动可视化
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant

# 终端3 - 启动控制面板
python isaacgymenvs/checkpoint_control_panel.py --experiment=Ant
```

**控制面板界面说明：**

- **Mode（模式选择）**：
  - `Auto`：自动模式，始终加载latest.pth，随训练进度更新
  - `Manual`：手动模式，可以选择特定的checkpoint进行查看

- **Available Checkpoints（可用的Checkpoint列表）**：
  - 显示实验目录下所有.pth文件
  - 按修改时间排序（最新的在上面）
  - 格式：`文件名 (大小) - 修改时间`

- **按钮**：
  - `Refresh List`：手动刷新checkpoint列表
  - `Load Selected`：加载当前选中的checkpoint

- **Current Checkpoint（当前加载的Checkpoint）**：
  - 显示正在可视化的checkpoint信息

**在控制面板中的操作：**
- 默认为Auto模式，自动跟随训练进度
- 切换到Manual模式，可以选择之前的任意checkpoint
- 选中一个checkpoint后，点击"Load Selected"，可视化窗口会自动切换到该checkpoint
- 切换回Auto模式，继续跟随最新的训练进度

### 工作原理

控制面板通过在实验目录下创建`.checkpoint_state.json`文件与可视化进程通信：

1. 控制面板监测用户的选择，更新state文件
2. 可视化进程（play mode）定期检查state文件
3. 如果检测到checkpoint变更，自动重新加载新的checkpoint
4. 整个过程无需重启可视化进程

### 典型使用场景

**场景1：训练过程中回顾早期checkpoint**
- 训练进行到500个epoch
- 想看看第100个epoch时的表现
- 在控制面板切换到Manual模式，选择对应的checkpoint
- 查看完毕后切换回Auto模式，继续观察最新进度

**场景2：比较不同checkpoint的表现**
- 在控制面板列表中依次选择不同的checkpoint
- 观察不同训练阶段agent的行为变化
- 找出表现最好的checkpoint

**场景3：调试特定checkpoint**
- 训练曲线显示某个epoch出现异常
- 使用控制面板加载该checkpoint
- 在可视化中仔细观察agent的行为
- 分析问题原因

## 常见问题

### Q: 可视化进程显示"Waiting for checkpoint"？
A: 这是正常的。可视化进程在等待训练进程生成第一个checkpoint。请确保训练进程正在运行并已经开始训练。

### Q: 启动play模式时提示需要experiment参数？
A: play模式必须指定experiment参数来指定要可视化哪个训练运行。可以使用完整目录名（如 `experiment=Ant_2025-01-14_10-30-45`）或部分名称（如 `experiment=Ant`），后者会自动找到最新匹配的目录。

### Q: 如何停止训练或可视化？
A:
- 可视化进程：按 `ESC` 键或 `Ctrl+C`
- 训练进程：`Ctrl+C`

### Q: 可以在同一台机器的不同GPU上运行训练和可视化吗？
A: 可以。使用不同的device参数：
```bash
# 终端1 - 在GPU 0上训练
python isaacgymenvs/train_play.py task=Ant mode=train experiment=Ant sim_device=cuda:0 rl_device=cuda:0

# 终端2 - 在GPU 1上可视化
python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant sim_device=cuda:1 rl_device=cuda:1 graphics_device_id=1
```

### Q: 可视化进程会影响训练性能吗？
A: 影响很小。因为：
1. 训练进程运行在headless模式，无渲染开销
2. 可视化进程只使用少量环境（默认4个）
3. 两个进程独立运行，互不干扰

### Q: 如何保存checkpoint的历史版本而不是覆盖？
A: `train_play.py` 的 `latest.pth` 是专门用于可视化的固定checkpoint。rl_games仍会按照配置保存历史checkpoint到实验目录的 `nn/` 文件夹，这些checkpoint不会被覆盖。

### Q: Checkpoint控制面板是必须的吗？
A: 不是必须的。控制面板是可选功能。如果您只需要跟随训练进度实时查看最新效果，只需运行train和play两个进程即可。控制面板主要用于需要在可视化过程中切换查看不同checkpoint的场景。

### Q: 在控制面板中切换checkpoint会中断可视化吗？
A: 不会。切换checkpoint时，可视化进程会自动重新加载模型并继续运行，整个过程平滑无缝，无需重启可视化窗口。

## 与原版train.py的区别

| 特性 | train.py | train_play.py |
|------|----------|--------------|
| 训练功能 | ✓ | ✓ |
| 测试功能 | ✓ | ✓ |
| 同时训练+可视化 | ✗ | ✓ |
| 自动checkpoint管理 | ✗ | ✓ |
| 可视化自动重载 | ✗ | ✓ |
| 自动时间戳目录 | ✗ | ✓ |
| Checkpoint控制面板 | ✗ | ✓ |

## 技术细节

### CheckpointObserver

`train_new.py` 实现了一个自定义的 `CheckpointObserver` 类，继承自rl_games的observer接口：

```python
class CheckpointObserver:
    def on_epoch_end(self, runner, update_num, epoch_num):
        if epoch_num % self.save_freq == 0 and epoch_num > 0:
            runner.save(self.checkpoint_path)
```

这个observer会在每个epoch结束时检查，并在达到保存频率时保存checkpoint。

### 可视化循环

可视化模式运行在一个循环中：
1. 加载checkpoint
2. 运行推理（显示环境）
3. 等待指定时间
4. 检查checkpoint是否更新
5. 如果更新，重新加载并返回步骤2

## 进一步扩展

如果你想添加更多功能，可以考虑：

1. **添加热键切换**：在可视化中按键手动触发重载
2. **性能监控**：在可视化窗口显示训练统计信息
3. **多任务支持**：同时可视化多个训练任务
4. **远程可视化**：通过网络加载远程机器的checkpoint

## 许可证

与主项目相同，采用NVIDIA的开源许可证。
