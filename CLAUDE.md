# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Isaac Gym Benchmark Environments is a repository for high-performance RL environments using NVIDIA Isaac Gym. It provides vectorized GPU-accelerated physics simulation for robot learning tasks with the rl_games training library.

## Prerequisites

- Requires NVIDIA Isaac Gym Preview 4 to be installed separately (download from developer.nvidia.com/isaac-gym)
- After installing Isaac Gym, install this package: `pip install -e .`
- Python 3.6+ required

## Common Development Commands

### Training

**Basic training:**
```bash
python train.py task=<TaskName>
```

**Training examples:**
```bash
# Quick test with Cartpole
python train.py task=Cartpole

# Headless training (faster)
python train.py task=Ant headless=True

# With specific number of environments
python train.py task=Ant num_envs=4096

# Multi-GPU training
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py multi_gpu=True task=Ant
```

**Loading checkpoints:**
```bash
# Continue training from checkpoint
python train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth

# Inference only (testing)
python train.py task=Ant checkpoint=runs/Ant/nn/Ant.pth test=True num_envs=64
```

**Note:** Escape special characters in checkpoint paths: `checkpoint="./runs/Ant/nn/last_Antep\=501rew\[5981.31\].pth"`

### Train + Visualize Mode (train_new.py)

`train_new.py` (located in `isaacgymenvs/`) supports simultaneous training and visualization in separate processes:

**Prerequisites:**
```bash
mamba activate rlgpu  # or conda activate rlgpu
```

**Start training (Terminal 1):**
```bash
python isaacgymenvs/train_new.py task=Ant mode=train
```

**Start visualization (Terminal 2):**
```bash
python isaacgymenvs/train_new.py task=Ant mode=play
```

**How it works:**
- Train mode runs headless, saves checkpoint to `runs/<task>/nn/latest.pth` periodically
- Play mode waits for checkpoint, then visualizes with 1-4 environments, auto-reloads on checkpoint update
- Both processes run independently without interfering with each other

**Configuration options:**
- `checkpoint_save_freq=100`: Save checkpoint every N **epochs** (training iterations, not seconds) - train mode
- `checkpoint_reload_interval=30`: Check for updates every N **seconds** - play mode
- `num_envs=4`: Number of visualization environments (play mode)

**Note:** An epoch is one training iteration cycle in rl_games, not a time period.

See `README_TRAIN_NEW.md` for detailed usage examples.

### Hydra Configuration

The repository uses Hydra for configuration management. You can override any config parameter from the command line:

```bash
# Override task config parameters
python train.py task=Ant task.env.numEnvs=512 task.env.episodeLength=1000

# Override training hyperparameters
python train.py task=Ant train.params.config.gamma=0.999 train.params.config.learning_rate=0.0001

# Device configuration
python train.py task=Ant sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0
```

**Pipeline modes:**
- `pipeline=gpu` (default): All data stays on GPU for maximum performance
- `pipeline=cpu`: Simulation can run on GPU but data copied to CPU each step

### WandB Integration

```bash
python train.py task=Ant wandb_activate=True wandb_entity=<entity> wandb_project=<project>
```

### Video Capture

```bash
# Capture videos during training
python train.py task=Ant capture_video=True capture_video_freq=1500 capture_video_len=100 force_render=False

# With WandB upload
python train.py task=Ant wandb_activate=True wandb_entity=nvidia capture_video=True
```

### Code Quality

```bash
# Run pre-commit checks (required before committing)
pre-commit run --all-files
```

## Architecture

### Main Entry Point

- `isaacgymenvs/train.py`: Main training script with Hydra decorator. Handles:
  - Environment creation via `isaacgymenvs.make()`
  - rl_games Runner setup with custom AMP (Adversarial Motion Priors) builders
  - Multi-GPU and PBT (Population Based Training) setup
  - WandB integration and video recording wrappers

### Task System

All tasks inherit from `VecTask` base class (in `isaacgymenvs/tasks/base/vec_task.py`), which provides:

- Vectorized environment interface compatible with gym
- Device management (CPU/GPU pipeline handling)
- Domain randomization support via `apply_randomizations()`
- Reset and step logic with pre/post physics hooks

**Task lifecycle:**
1. `create_sim()`: Set up simulation, create ground plane and environments
2. `pre_physics_step(actions)`: Apply actions before physics step
3. Physics simulation runs
4. `post_physics_step()`: Compute observations and rewards after physics

### Task Registration

Tasks are registered in `isaacgymenvs/tasks/__init__.py` in the `isaacgym_task_map` dictionary. Some tasks use resolver functions (e.g., `resolve_allegro_kuka`) to select subtasks based on config.

### Configuration Structure

```
isaacgymenvs/cfg/
├── config.yaml          # Root config with defaults and device settings
├── task/                # Task-specific configs (e.g., Ant.yaml, Cartpole.yaml)
│   └── <TaskName>.yaml  # Must have 'name' field matching task map key
├── train/               # RL training configs (e.g., AntPPO.yaml)
│   └── <TaskName>PPO.yaml
└── pbt/                 # Population Based Training configs
```

**Config hierarchy:** Hydra resolves with interpolations like `${task.env.numEnvs}` where dots navigate up the config tree.

### Task Directory Organization

```
isaacgymenvs/tasks/
├── base/
│   └── vec_task.py      # VecTask base class
├── <task_name>.py       # Individual task implementations
├── allegro_kuka/        # Multi-file task with subtasks
├── factory/             # Factory simulation tasks
├── industreal/          # IndustReal tasks
├── dextreme/            # DeXtreme domain randomization tasks
└── amp/                 # AMP (motion priors) related utilities
```

### Custom Learning Algorithms

The `isaacgymenvs/learning/` directory contains custom rl_games extensions:

- `amp_continuous.py`, `amp_models.py`, `amp_network_builder.py`, `amp_players.py`: AMP implementation
- `common_agent.py`, `common_player.py`: Base agent/player classes
- `hrl_continuous.py`, `hrl_models.py`: Hierarchical RL support

These are registered in train.py via `runner.algo_factory.register_builder()` and `model_builder.register_model()`.

## Creating New Tasks

1. Create `isaacgymenvs/tasks/my_task.py` inheriting from `VecTask`
2. Implement required methods: `create_sim()`, `pre_physics_step(actions)`, `post_physics_step()`
3. Add to `isaacgymenvs/tasks/__init__.py`:
   ```python
   from .my_task import MyTask
   isaacgym_task_map = {
       ...
       "MyTask": MyTask,
   }
   ```
4. Create `isaacgymenvs/cfg/task/MyTask.yaml` with `name: MyTask` field
5. Create `isaacgymenvs/cfg/train/MyTaskPPO.yaml` with rl_games parameters
6. Run: `python train.py task=MyTask`

**Key VecTask config parameters:**
- `env.numEnvs`: Number of parallel environments
- `env.numObservations`: Observation vector size
- `env.numActions`: Action vector size
- `env.controlFrequencyInv`: Physics steps per RL action (decimation)
- `env.clipObservations` / `env.clipActions`: Value clipping ranges
- `env.enableCameraSensors`: Enable camera sensors

## Domain Randomization

IsaacGymEnvs supports on-the-fly domain randomization without asset reloading. Configure in task YAML under `task.randomize` or call `apply_randomizations()` directly. Can randomize:

- `observations`: Add noise to observations
- `actions`: Add noise to actions
- `sim_params`: Physical scene parameters (gravity, etc.)
- `actor_params`: Per-actor properties (mass, friction, etc.)

See `docs/domain_randomization.md` for full details.

## Population Based Training (PBT)

PBT support for hyperparameter search and difficult environments. Enable with PBT config files in `isaacgymenvs/cfg/pbt/`. See `docs/pbt.md` for usage.

## Important Implementation Notes

- **Device handling:** Tasks can run with `pipeline=gpu` (all GPU) or `pipeline=cpu` (CPU copy each step)
- **Tensor wrapping:** Use `gymtorch.wrap_tensor()` to wrap Isaac Gym tensor pointers into PyTorch tensors
- **State tensors:** Acquire tensors like DOF states via `gym.acquire_dof_state_tensor(sim)` in task `__init__`
- **Resets:** Implement `reset_idx(env_ids)` to reset specific environments by indices
- **Asset paths:** Assets are in `assets/` directory (migrated from Isaac Gym examples)
- **Viewer controls:** 'V' toggle rendering, 'R' record video, 'Tab' toggle panel, 'ESC' quit

## Available Tasks

Common tasks include: Ant, Anymal, AnymalTerrain, BallBalance, Cartpole, FrankaCabinet, FrankaCubeStack, Humanoid, HumanoidAMP, Ingenuity, Quadcopter, ShadowHand, AllegroHand, Trifinger, and various Factory/IndustReal tasks.

Full list in `isaacgymenvs/tasks/__init__.py` `isaacgym_task_map`.
