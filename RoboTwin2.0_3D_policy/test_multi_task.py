import numpy as np
import os
import cv2
from datetime import datetime
from script.eval_3dpolicy import Env
import sys

import torch
from collections import deque
import json
from typing import Dict
import argparse

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# print(current_script_path)
# Get the directory containing the script, which is the RoboTwin2.0_3D_policy/ directory
project_root_dir = os.path.dirname(current_script_path)
# print(project_root_dir) # /data/sea_disk0/wushr/3D-Policy/DP3Encoder-to-Uni3D/3D-Diffusion-Policy/RoboTwin2.0_3D_policy
# Add the project root directory to sys.path for proper module imports
sys.path.append(project_root_dir)

# Add 3D-Diffusion-Policy path to import DP3 related modules
dp3_path = os.path.join(os.path.dirname(project_root_dir), '3D-Diffusion-Policy')
# print(dp3_path) # /data/sea_disk0/wushr/3D-Policy/DP3Encoder-to-Uni3D/3D-Diffusion-Policy/3D-Diffusion-Policy
sys.path.insert(0, dp3_path)

def load_dp3_multi_task_policy(task_name="multi_task_robotwin2", checkpoint_num=1000, run_dir='multi_task_robotwin2-dp3_multi_task-0100_seed0', policy='dp3_multi_task'):
    """Load multi-task DP3 policy"""
    try:
        # Add DP3 policy path
        dp3_policy_path = dp3_path
        if dp3_policy_path not in sys.path:
            sys.path.insert(0, dp3_policy_path)

        # Import DP3 classes
        from eval import DP3_policy

        # Use existing evaluation script methods
        import hydra

        # Set configuration path - use multi-task configuration from 3D-Diffusion-Policy
        config_path = "../3D-Diffusion-Policy/diffusion_policy_3d/config"

        # import pdb
        # pdb.set_trace()

        # Create multi-task configuration
        with hydra.initialize(config_path=config_path, version_base=None):
            # Use multi-task configuration file
            cfg = hydra.compose(config_name=f"{policy}.yaml", overrides=[f"task={task_name}"])

            # Create DP3 instance
            dp3_policy, n_obs_steps = DP3_policy(cfg, checkpoint_num=checkpoint_num, run_dir=run_dir).get_policy()
            print(f"✓ Successfully loaded multi-task DP3 policy, task: {task_name}, checkpoint: {checkpoint_num}, run_dir: {run_dir}")
            return dp3_policy, n_obs_steps

    except Exception as e:
        print(f"✗ Failed to load multi-task DP3 policy: {e}")
        import traceback
        traceback.print_exc()
        return None



class MultiTaskEnvironmentManager:
    """Multi-task environment manager"""

    def __init__(self):
        self.env_manager = Env()
        self.current_task = None
        self.current_instruction = None

        # List of tasks supported by RoboTwin2.0
        self.available_tasks = [
            "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
            "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
            "handover_block", "handover_mic", "hanging_mug", "lift_pot",
            "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
            "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
            "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
            "place_burger_fries", "place_can_basket", "place_cans_plasticbox", "place_container_plate",
            "place_dual_shoes", "place_empty_cup", "place_fan", "place_mouse_pad",
            "place_object_basket", "place_object_scale", "place_object_stand", "place_phone_stand",
            "place_shoe", "press_stapler", "put_bottles_dustbin", "put_object_cabinet",
            "rotate_qrcode", "scan_object", "shake_bottle", "shake_bottle_horizontally",
            "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two",
            "stamp_seal", "turn_switch"
        ]

        # Evaluation result statistics
        self.task_results = {}
        self.reset_statistics()

    def reset_statistics(self):
        """Reset statistics information"""
        for task in self.available_tasks:
            self.task_results[task] = {
                'total_episodes': 0,
                'successful_episodes': 0,
                'success_rate': 0.0
            }

    def create_task_environment(self, task_name: str, head_camera_type: str = "D435",
                              seed: int = 42, task_num: int = 1, task_config: str = "demo_clean",
                              instruction_type: str = "unseen"):
        """Create environment for specified task"""
        try:
            result = self.env_manager.Create_env(
                task_name=task_name,
                head_camera_type=head_camera_type,
                seed=seed,
                task_num=task_num,
                instruction_type=instruction_type,
                task_config=task_config
            )

            if not result:
                print(f"✗ Failed to create task environment {task_name}: unable to get valid seeds")
                return [], [], []

            seed_list, id_list, episode_info_list_total = result
            self.current_task = task_name
            print(f"✓ Successfully created task environment: {task_name}, found {len(seed_list)} valid seeds")
            return seed_list, id_list, episode_info_list_total

        except Exception as e:
            print(f"✗ Failed to create task environment {task_name}: {e}")
            return [], [], []

    def init_task_episode(self, task_name, seed: int, task_id: int, episode_info_list, run_dir, checkpoint_num):
        """Initialize task episode"""
        save_dir = run_dir + '/' + task_name
        return self.env_manager.Init_task_env(seed, task_id, episode_info_list, save_dir, checkpoint_num)

    def get_observation(self):
        """Get observation from environment"""
        return self.env_manager.get_observation()

    def take_action(self, action, obs_history, n_obs_steps):
        """Execute action and return updated observation history"""
        return self.env_manager.Take_action(action, obs_history, n_obs_steps)

    def close_environment(self):
        """Close environment"""
        self.env_manager.Close_env()

    def update_task_result(self, task_name: str, success: bool):
        """Update task result statistics"""
        if task_name in self.task_results:
            self.task_results[task_name]['total_episodes'] += 1
            if success:
                self.task_results[task_name]['successful_episodes'] += 1

            # Update success rate
            total = self.task_results[task_name]['total_episodes']
            successful = self.task_results[task_name]['successful_episodes']
            self.task_results[task_name]['success_rate'] = successful / total if total > 0 else 0.0

    def get_statistics_summary(self) -> Dict:
        """Get statistics summary"""
        total_episodes = sum(result['total_episodes'] for result in self.task_results.values())
        total_successful = sum(result['successful_episodes'] for result in self.task_results.values())
        overall_success_rate = total_successful / total_episodes if total_episodes > 0 else 0.0

        return {
            'task_results': self.task_results,
            'overall_success_rate': overall_success_rate,
            'total_episodes': total_episodes,
            'total_successful': total_successful
        }

# Create image save directory
def create_image_dir(task_name, seed, epoch=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if epoch is not None:
        # 使用epoch信息构建路径
        image_base_dir = os.path.join(project_root_dir, "videos", f"{task_name}-epoch_{epoch}")
        image_dir = os.path.join(image_base_dir, f"seed_{seed}_{timestamp}")
    else:
        image_base_dir = os.path.join(project_root_dir, "observation_images")
        image_dir = os.path.join(image_base_dir, f"{task_name}_{seed}_{timestamp}")
    os.makedirs(image_dir, exist_ok=True)
    return image_dir

# Save image function
def save_image(image_array, save_dir, step, camera_name="front_camera"):
    """Save numpy format image to file"""
    # Ensure correct image format (RGB format numpy array)
    if image_array.dtype != np.uint8:
        # If float type and range is [0,1], convert to [0,255]
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    
    # Save using OpenCV (need to convert RGB to BGR)
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_path = os.path.join(save_dir, f"{camera_name}_step_{step:04d}.png")
    cv2.imwrite(image_path, image_bgr)
    return image_path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RoboTwin2.0 Multi-task DP3 Policy Evaluation')
    parser.add_argument('--task_name', type=str, default='multi_task_robotwin2',
                        help='Task name to evaluate (default: multi_task_robotwin2)')
    parser.add_argument('--epoch', '--checkpoint_epoch', type=int, default=200,
                        help='Training epoch number for model checkpoint to load (default: 200)')
    parser.add_argument('--num_episodes', type=int, default=3,
                        help='Number of episodes to evaluate per task (default: 3)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--head_camera_type', type=str, default='D435',
                        help='Head camera type (default: D435)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--run_dir', type=str, default='multi_task_robotwin2-dp3_multi_task-0100_seed0',
                        help='Run directory path')
    parser.add_argument('--alg_name', type=str, default='dp3_multi_task',
                        help='Policy algorithm to use')
    parser.add_argument('--tasks', nargs='*', default=None,
                        help='List of tasks to evaluate, if not specified all tasks will be evaluated')
    parser.add_argument('--task_config', type=str, default='demo_clean',
                        help='Task configuration (default: demo_clean)')
    parser.add_argument('--instruction_type', type=str, default='unseen',
                        help='Instruction type (default: unseen)')
    return parser.parse_args()

def run_multi_task_evaluation(tasks_to_evaluate = None,
                             episodes_per_task = 3,
                             checkpoint_num = 200,
                             head_camera_type = "D435",
                             max_steps = 1000,
                             eval_seed=1,
                             run_dir='multi_task_robotwin2-dp3_multi_task-9999_seed0',
                             policy='dp3_multi_task',
                             task_name='multi_task_robotwin2',
                             task_config='demo_clean',
                             instruction_type='unseen'):
    """Run multi-task evaluation"""

    # 1. Initialize multi-task environment manager
    multi_env = MultiTaskEnvironmentManager()

    # 2. Set task list to evaluate
    if tasks_to_evaluate is None:
        # Default to evaluate all available tasks
        tasks_to_evaluate = multi_env.available_tasks

    print(f"Starting multi-task evaluation, task list: {tasks_to_evaluate}")
    print(f"Evaluating {episodes_per_task} episodes per task")

    # 3. Initialize multi-task DP3 policy
    print(f"Loading multi-task DP3 policy, task: {task_name}, checkpoint: {checkpoint_num}, run_dir: {run_dir}, policy: {policy}")
    dp3_policy, n_obs_steps = load_dp3_multi_task_policy(task_name=task_name, checkpoint_num=checkpoint_num, run_dir=run_dir, policy=policy)
    if dp3_policy is None:
        print("Will use random actions as fallback")

    # 4. Evaluate each task
    for task_idx, current_task_name in enumerate(tasks_to_evaluate):
        print(f"\n{'='*60}")
        print(f"Evaluating task {task_idx+1}/{len(tasks_to_evaluate)}: {current_task_name}")
        print(f"{'='*60}")

        # Create task environment
        seed_list, id_list, episode_info_list_total = multi_env.create_task_environment(
            task_name=current_task_name,
            head_camera_type=head_camera_type,
            seed=eval_seed,
            task_num=episodes_per_task,
            task_config=task_config,
            instruction_type=instruction_type
        )

        print(f"Found {len(seed_list)} valid task seeds: {seed_list}")

        if not seed_list:
            print(f"Skipping task {current_task_name}: unable to create environment")
            continue

        # Evaluate each seed
        for episode_idx, (seed, task_id, episode_info_list) in enumerate(zip(seed_list, id_list, episode_info_list_total)):
            print(f"\n--- Task {current_task_name}, Episode {episode_idx+1}/{len(seed_list)}, Seed: {seed} ---")

            # Initialize task episode and get language instruction directly from inst
            inst = multi_env.init_task_episode(current_task_name, seed, task_id, episode_info_list, run_dir, checkpoint_num)
            language_instruction = inst  # Language instruction is returned directly from init_task_episode
            print(f"Language instruction: {language_instruction}")

            # Run single episode
            episode_success = run_single_episode(
                multi_env, dp3_policy, language_instruction,
                max_steps, n_obs_steps
            )

            # if episode_success:
            #     video_path = os.path.join(project_root_dir, f'eval_video/{checkpoint_num}-{run_dir}/{current_task_name}/seed{seed}')
            #     new_video_path = os.path.join(project_root_dir, f'eval_video/{checkpoint_num}-{run_dir}/{current_task_name}/success_seed{seed}')
            #     if os.path.exists(video_path):
            #         os.rename(video_path, new_video_path)
            # else:
            #     video_path = os.path.join(project_root_dir, f'eval_video/{checkpoint_num}-{run_dir}/{current_task_name}/seed{seed}')
            #     new_video_path = os.path.join(project_root_dir, f'eval_video/{checkpoint_num}-{run_dir}/{current_task_name}/fail_seed{seed}')
            #     if os.path.exists(video_path):
            #         os.rename(video_path, new_video_path)

            # Update statistics
            multi_env.update_task_result(current_task_name, episode_success)

            print(f"Episode result: {'Success' if episode_success else 'Failed'}")

    # 5. Print final statistics
    print_final_statistics(multi_env, checkpoint_num, run_dir)

def run_single_episode(multi_env: MultiTaskEnvironmentManager,
                      dp3_policy,
                      language_instruction: str,
                      max_steps: int = 1000,
                      n_obs_steps: int = 2) -> bool:
    """Run single evaluation episode"""

    # Initialize observation history following test.py pattern
    obs_history = deque(maxlen=n_obs_steps)  # Store recent n_obs_steps observations
    observation = multi_env.get_observation()

    # Add current observation to history
    current_obs = {
        'point_cloud': torch.from_numpy(observation['pointcloud']),
        'agent_pos': torch.from_numpy(observation['joint_action']['vector'])
    }
    obs_history.append(current_obs)

    # Fill history if insufficient
    while len(obs_history) < n_obs_steps:
        # Copy current observation to fill history
        obs_history.appendleft(current_obs.copy())

    episode_success = False

    for step in range(max_steps):
        # Calculate action
        action = predict_action_with_language(
            dp3_policy, language_instruction, obs_history
        )

        # Execute action and get updated observation history
        status, obs_history = multi_env.take_action(action, obs_history, n_obs_steps)
        print(f"Step {step}: status = {status}")

        # Check episode termination conditions
        if status == "success":
            episode_success = True
            break
        elif status == "fail":
            break
        elif status != "run":
            break

    # Ensure environment is closed
    if status == "run":
        multi_env.close_environment()

    return episode_success

def predict_action_with_language(dp3_policy, language_instruction: str, obs_history):
    """Predict action using language instruction"""
    if dp3_policy is not None:
        try:
            # Build time-series observation data with language instruction
            obs_dict = {
                'point_cloud': torch.stack([obs['point_cloud'] for obs in obs_history], dim=0).unsqueeze(0),
                'agent_pos': torch.stack([obs['agent_pos'] for obs in obs_history], dim=0).unsqueeze(0),
                'language': language_instruction  # Add language instruction
            }

            action_dict = dp3_policy.predict_action(obs_dict)
            action = action_dict['action'].squeeze(0).detach().cpu().numpy()
            return action

        except Exception as e:
            print(f"DP3 policy prediction failed: {e}, using random action")
            return np.random.uniform(-1, 1, (1, 16))
    else:
        # Use random action as fallback
        return np.random.uniform(-1, 1, (1, 16))

def print_final_statistics(multi_env: MultiTaskEnvironmentManager, checkpoint_num, run_dir):
    """Print final statistics results"""
    stats = multi_env.get_statistics_summary()

    print(f"\n{'='*80}")
    print("Multi-task Evaluation Results Statistics")
    print(f"{'='*80}")

    print(f"Overall success rate: {stats['overall_success_rate']:.2%}")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Successful episodes: {stats['total_successful']}")

    print(f"\nDetailed results by task:")
    print(f"{'Task Name':<25} {'Total':<8} {'Success':<8} {'Rate':<8}")
    print("-" * 60)

    for task_name, result in stats['task_results'].items():
        if result['total_episodes'] > 0:
            print(f"{task_name:<25} {result['total_episodes']:<8} {result['successful_episodes']:<8} {result['success_rate']:.2%}")

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(project_root_dir, f"eval_video/{checkpoint_num}-{run_dir}/multi_task_results_{timestamp}.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_file}")

# Main program entry point
if __name__ == "__main__":
    args = parse_arguments()

    # If no tasks specified, use default task list
    if args.tasks is None:
        tasks_to_evaluate = [
            "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
            "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller",
            "handover_block", "handover_mic", "hanging_mug", "lift_pot",
            "move_can_pot", "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
            "open_laptop", "open_microwave", "pick_diverse_bottles", "pick_dual_bottles",
            "place_a2b_left", "place_a2b_right", "place_bread_basket", "place_bread_skillet",
            "place_burger_fries", "place_can_basket", "place_cans_plasticbox", "place_container_plate",
            "place_dual_shoes", "place_empty_cup", "place_fan", "place_mouse_pad",
            "place_object_basket", "place_object_scale", "place_object_stand", "place_phone_stand",
            "place_shoe", "press_stapler", "put_bottles_dustbin", "put_object_cabinet",
            "rotate_qrcode", "scan_object", "shake_bottle", "shake_bottle_horizontally",
            "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two",
            "stamp_seal", "turn_switch"
        ]
    else:
        tasks_to_evaluate = args.tasks

    print("Starting RoboTwin2.0 Multi-task DP3 Evaluation")
    print(f"Evaluation tasks: {tasks_to_evaluate}")
    print(f"Policy: {args.alg_name}")

    # Run multi-task evaluation
    run_multi_task_evaluation(
        tasks_to_evaluate=tasks_to_evaluate,
        episodes_per_task=args.num_episodes,
        checkpoint_num=args.epoch,
        head_camera_type=args.head_camera_type,
        max_steps=args.max_steps,
        eval_seed=args.seed,
        run_dir=args.run_dir,
        policy=args.alg_name,
        task_name=args.task_name,
        task_config=args.task_config,
        instruction_type=args.instruction_type
    )
