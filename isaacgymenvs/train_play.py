# train_new.py
# Script to train policies in Isaac Gym with simultaneous visualization support
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import hydra
import time
import os

from omegaconf import DictConfig, OmegaConf
from isaacgym import gymapi


def preprocess_train_config(cfg, config_dict):
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict


class CheckpointObserver:
    """Observer to periodically save checkpoint to a fixed filename for visualization process

    This observer saves checkpoint every N epochs (training iterations) to a fixed filename,
    allowing the visualization process to load the latest trained model.

    Args:
        checkpoint_dir: Directory to save checkpoints
        checkpoint_name: Fixed filename for the checkpoint (default: 'latest.pth')
        save_freq: Save checkpoint every N epochs/iterations (default: 100)
    """

    def __init__(self, checkpoint_dir, checkpoint_name='latest', save_freq=100):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name
        self.save_freq = save_freq
        # Don't add .pth here, algo.save() will add it automatically
        self.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        self.algo = None
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"CheckpointObserver: Will save to {self.checkpoint_path}.pth every {save_freq} epochs")

    def before_init(self, base_name, config, experiment_name):
        """Called before algorithm initialization"""
        pass

    def after_init(self, algo):
        """Called after algorithm initialization"""
        self.algo = algo

    def process_infos(self, infos, done_indices):
        """Called to process environment info"""
        pass

    def after_steps(self):
        """Called after environment steps"""
        pass

    def after_clear_stats(self):
        """Called after clearing stats"""
        pass

    def after_print_stats(self, frame, epoch_num, total_time):
        """Called after printing stats - we use this to save checkpoints"""
        # Save checkpoint every save_freq epochs
        if epoch_num % self.save_freq == 0 and epoch_num > 0:
            print(f"\nSaving checkpoint to {self.checkpoint_path} at epoch {epoch_num}")
            # Use algo.save() method to save checkpoint
            if self.algo is not None:
                self.algo.save(self.checkpoint_path)


def wait_for_checkpoint(checkpoint_path, check_interval=5):
    """Wait for checkpoint file to appear"""
    print(f"Waiting for checkpoint at {checkpoint_path}...")
    while not os.path.exists(checkpoint_path):
        time.sleep(check_interval)
        print(f"Still waiting for checkpoint... (checking every {check_interval}s)")
    print(f"Checkpoint found at {checkpoint_path}!")


def setup_camera_for_env0(envs, task_name):
    """Setup camera to focus on env0 with close distance

    Args:
        envs: The environment/task object (isaacgymenvs.make() returns the task directly)
        task_name: Name of the task for determining camera position
    """
    try:
        # The envs object IS the task object (from isaacgym_task_map)
        task = envs

        # Try various ways to access the task object
        if hasattr(envs, 'task'):
            task = envs.task
        elif hasattr(envs, 'env'):
            task = envs.env
        elif hasattr(envs, 'unwrapped'):
            task = envs.unwrapped

        # Get gym and viewer
        if not hasattr(task, 'gym'):
            print(f"Warning: Task object doesn't have 'gym' attribute. Type: {type(task)}")
            print(f"Available attributes: {[a for a in dir(task) if not a.startswith('_')][:20]}")
            return

        if not hasattr(task, 'viewer'):
            print(f"Warning: Task doesn't have 'viewer' attribute")
            return

        if task.viewer is None:
            print(f"Warning: Viewer is None (probably running in headless mode)")
            return

        gym = task.gym
        viewer = task.viewer

        # Get env spacing to calculate env0 center
        env_spacing = getattr(task.cfg['env'], 'envSpacing', 2.0)

        # Calculate env0 position (usually at origin)
        # Most tasks arrange envs in a grid pattern
        env0_x = 0.0
        env0_y = 0.0
        env0_z = 0.5  # Default height

        # Task-specific adjustments for better viewing - MUCH CLOSER
        if 'franka' in task_name.lower():
            if 'cube' in task_name.lower() or 'stack' in task_name.lower():
                # Franka cube stacking - very close view
                cam_pos = gymapi.Vec3(1.0, 1.0, 1.2)
                cam_target = gymapi.Vec3(env0_x, env0_y, 0.6)
            elif 'cabinet' in task_name.lower():
                # Franka cabinet - close view
                cam_pos = gymapi.Vec3(1.2, 1.2, 1.0)
                cam_target = gymapi.Vec3(env0_x, env0_y, 0.5)
            else:
                # Other Franka tasks
                cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
                cam_target = gymapi.Vec3(env0_x, env0_y, 0.5)
        elif 'hand' in task_name.lower() or 'allegro' in task_name.lower():
            # Hand tasks - very close view
            cam_pos = gymapi.Vec3(0.6, 0.6, 0.6)
            cam_target = gymapi.Vec3(env0_x, env0_y, 0.3)
        elif 'ant' in task_name.lower() or 'humanoid' in task_name.lower():
            # Locomotion tasks - follow from behind
            cam_pos = gymapi.Vec3(1.5, 1.5, 0.8)
            cam_target = gymapi.Vec3(env0_x, env0_y, 0.0)
        else:
            # Default: close diagonal view
            cam_pos = gymapi.Vec3(1.2, 1.2, 1.0)
            cam_target = gymapi.Vec3(env0_x, env0_y, 0.5)

        # Set camera
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
        print(f"Camera focused on env0 at position ({env0_x:.1f}, {env0_y:.1f}, {env0_z:.1f})")
        print(f"Camera position: ({cam_pos.x:.1f}, {cam_pos.y:.1f}, {cam_pos.z:.1f})")

    except Exception as e:
        print(f"Warning: Could not set camera position: {e}")


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    import logging
    from datetime import datetime

    # noinspection PyUnresolvedReferences
    import isaacgym
    from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
    from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
    from hydra.utils import to_absolute_path
    from isaacgymenvs.tasks import isaacgym_task_map
    import gym
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed

    # Add default mode parameter if not present
    if 'mode' not in cfg:
        cfg.mode = 'train'

    # Validate mode
    if cfg.mode not in ['train', 'play']:
        raise ValueError(f"mode must be 'train' or 'play', got '{cfg.mode}'")

    print(f"\n{'='*60}")
    print(f"Running in {cfg.mode.upper()} mode")
    print(f"{'='*60}\n")

    # Auto-configure based on mode
    if cfg.mode == 'train':
        # Training mode: headless with many environments
        cfg.headless = True
        cfg.test = False
        print(f"Training mode: headless={cfg.headless}, num_envs={cfg.num_envs if cfg.num_envs else cfg.task.env.numEnvs}")
    else:  # play mode
        # Visualization mode: non-headless with few environments
        cfg.headless = False
        cfg.test = True
        # Override num_envs for visualization if not explicitly set
        if not cfg.num_envs or cfg.num_envs == '':
            cfg.num_envs = 4
            cfg.task.env.numEnvs = 4
        print(f"Visualization mode: headless={cfg.headless}, num_envs={cfg.num_envs}")

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    import isaacgymenvs

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    # Setup experiment directory and checkpoint path
    if cfg.mode == 'train':
        # Training mode: use timestamped directory
        if cfg.experiment:
            experiment_dir_name = f"{cfg.experiment}_{time_str}"
        else:
            experiment_dir_name = f"{cfg.task_name}_{time_str}"
        experiment_dir = os.path.join('runs', experiment_dir_name)
        checkpoint_dir = os.path.join(experiment_dir, 'nn')
        checkpoint_filename = 'latest'  # algo.save() will add .pth automatically

        print(f"\nTraining will save to: {experiment_dir}")
        print(f"Checkpoint will be saved to: {checkpoint_dir}/latest.pth\n")

    else:  # play mode
        # Play mode: must specify experiment directory
        if not cfg.experiment:
            raise ValueError(
                "Play mode requires 'experiment' parameter to specify which training run to visualize.\n"
                "Example: python isaacgymenvs/train_play.py task=Ant mode=play experiment=Ant_14-01-05-06\n"
                "Available runs in 'runs/' directory:"
            )

        experiment_dir = os.path.join('runs', cfg.experiment)
        checkpoint_dir = os.path.join(experiment_dir, 'nn')
        checkpoint_filename = 'latest'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename + '.pth')

        # Check if experiment directory exists
        if not os.path.exists(experiment_dir):
            # Try to find the most recent matching directory
            import glob
            pattern = os.path.join('runs', f"{cfg.experiment}*")
            matching_dirs = sorted(glob.glob(pattern), reverse=True)
            if matching_dirs:
                experiment_dir = matching_dirs[0]
                checkpoint_dir = os.path.join(experiment_dir, 'nn')
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename + '.pth')
                print(f"\nFound experiment directory: {experiment_dir}")
            else:
                raise ValueError(
                    f"Experiment directory not found: {experiment_dir}\n"
                    f"Please check available runs in 'runs/' directory"
                )

        # Wait for checkpoint if it doesn't exist
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found yet: {checkpoint_path}")
            wait_for_checkpoint(checkpoint_path)

        # Set checkpoint to load
        cfg.checkpoint = checkpoint_path
        print(f"\nWill load checkpoint from: {cfg.checkpoint}\n")

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )

        # Setup camera for play mode
        if cfg.mode == 'play':
            print(f"\n[Initial setup] Configuring camera to focus on env0...")
            setup_camera_for_env0(envs, cfg.task_name)
            print()

        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

    if dict_cls:

        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}

        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:

        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    # Add checkpoint observer for train mode
    checkpoint_observer = None
    if cfg.mode == 'train':
        checkpoint_save_freq = cfg.get('checkpoint_save_freq', 100)
        checkpoint_observer = CheckpointObserver(
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_filename,
            save_freq=checkpoint_save_freq
        )
        observers.append(checkpoint_observer)

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)

    # register new AMP network builder and agent
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        return runner

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict
    # Note: experiment_dir is already set in train/play mode setup above
    if not cfg.test and os.path.exists(experiment_dir):
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    # For play mode, run in a loop to reload checkpoint periodically
    if cfg.mode == 'play':
        import datetime
        import json

        reload_interval = cfg.get('checkpoint_reload_interval', 30)  # seconds
        state_file = os.path.join(experiment_dir, '.checkpoint_state.json')

        print(f"\n{'='*60}")
        print(f"VISUALIZATION MODE")
        print(f"{'='*60}")
        print(f"Checkpoint: {cfg.checkpoint}")
        print(f"Reload interval: {reload_interval} seconds")
        print(f"Number of environments: {cfg.num_envs}")
        print(f"\nControl Panel:")
        print(f"  Launch control panel to switch checkpoints:")
        print(f"  python isaacgymenvs/checkpoint_control_panel.py --experiment={cfg.experiment}")
        print(f"\nViewer Controls:")
        print(f"  Mouse Left - Rotate camera")
        print(f"  Mouse Middle - Pan camera")
        print(f"  Mouse Wheel - Zoom in/out")
        print(f"  W/A/S/D - Move camera")
        print(f"  Q/E - Move camera up/down")
        print(f"  ESC - Exit visualization")
        print(f"{'='*60}\n")

        iteration_count = 0
        last_state_update = None

        while True:
            try:
                iteration_count += 1
                need_reload = False
                reload_reason = ""

                # Check control panel state file first
                if os.path.exists(state_file):
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)

                        state_update_time = state.get('last_update', None)

                        # Check if state has been updated
                        if state_update_time != last_state_update:
                            new_checkpoint = state.get('checkpoint_path', None)
                            if new_checkpoint and new_checkpoint != cfg.checkpoint:
                                if os.path.exists(new_checkpoint):
                                    cfg.checkpoint = new_checkpoint
                                    need_reload = True
                                    reload_reason = "Control panel changed checkpoint"
                                    last_state_update = state_update_time
                                else:
                                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Warning: Checkpoint from control panel not found: {new_checkpoint}")

                    except Exception as e:
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Warning: Error reading state file: {e}")

                # Get the latest modification time of current checkpoint
                last_modified = os.path.getmtime(cfg.checkpoint) if os.path.exists(cfg.checkpoint) else 0

                if iteration_count == 1:
                    # First load
                    file_size = os.path.getsize(cfg.checkpoint) / (1024 * 1024)  # MB
                    mod_time = datetime.datetime.fromtimestamp(last_modified).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading initial checkpoint...")
                    print(f"  File: {cfg.checkpoint}")
                    print(f"  Size: {file_size:.2f} MB")
                    print(f"  Modified: {mod_time}")
                    print(f"  Starting visualization...\n")

                # Run inference
                runner.run({
                    'train': False,
                    'play': True,
                    'checkpoint': cfg.checkpoint,
                    'sigma': cfg.sigma if cfg.sigma != '' else None
                })

                # Wait and check if checkpoint has been updated
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Waiting {reload_interval}s for checkpoint updates...")
                time.sleep(reload_interval)

                # Check if checkpoint file has been modified (for auto mode with latest.pth)
                if os.path.exists(cfg.checkpoint):
                    current_modified = os.path.getmtime(cfg.checkpoint)
                    if current_modified > last_modified:
                        need_reload = True
                        reload_reason = "Checkpoint file updated"

                # Reload if needed
                if need_reload:
                    file_size = os.path.getsize(cfg.checkpoint) / (1024 * 1024)  # MB
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(cfg.checkpoint)).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"\n{'='*60}")
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] NEW CHECKPOINT DETECTED!")
                    print(f"  Reason: {reload_reason}")
                    print(f"{'='*60}")
                    print(f"  File: {cfg.checkpoint}")
                    print(f"  Size: {file_size:.2f} MB")
                    print(f"  Modified: {mod_time}")
                    print(f"  Reloading model...")
                    # Reset and reload
                    runner.reset()
                    runner.load(rlg_config_dict)
                    print(f"  Model reloaded successfully!")
                    print(f"{'='*60}\n")
                else:
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] No updates. Continuing with current model...")

            except KeyboardInterrupt:
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Visualization stopped by user")
                break
            except Exception as e:
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Error in visualization loop: {e}")
                print(f"Will retry in {reload_interval} seconds...")
                time.sleep(reload_interval)
    else:
        # Normal training mode
        runner.run({
            'train': not cfg.test,
            'play': cfg.test,
            'checkpoint': cfg.checkpoint,
            'sigma': cfg.sigma if cfg.sigma != '' else None
        })


if __name__ == "__main__":
    launch_rlg_hydra()
