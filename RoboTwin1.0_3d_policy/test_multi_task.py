import numpy as np
import os
import cv2
from datetime import datetime
from script.eval_3dpolicy_new import Env
import sys
import pickle
import torch
from collections import deque
import json
import random
from typing import Dict, List
import argparse

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# print(current_script_path)
# 获取脚本所在的目录，也就是 RoboTwin1.0_3d_policy/ 目录
project_root_dir = os.path.dirname(current_script_path)
# print(project_root_dir) # /data/sea_disk0/wushr/3D-Policy/DP3Encoder-to-Uni3D/3D-Diffusion-Policy/RoboTwin1.0_3d_policy
# 将项目根目录添加到sys.path，以便正确导入项目内部模块
sys.path.append(project_root_dir)

# 添加3D-Diffusion-Policy路径以导入DP3相关模块
dp3_path = os.path.join(os.path.dirname(project_root_dir), '3D-Diffusion-Policy')
# print(dp3_path) # /data/sea_disk0/wushr/3D-Policy/DP3Encoder-to-Uni3D/3D-Diffusion-Policy/3D-Diffusion-Policy
sys.path.insert(0, dp3_path)

# # 添加openpoints模块路径
# openpoints_path = os.path.join(os.path.dirname(project_root_dir), 'openpoints')
# if openpoints_path not in sys.path:
#     sys.path.insert(0, openpoints_path)
#     print(f"Added openpoints path: {openpoints_path}")

def load_dp3_multi_task_policy(task_name="multi_task_robotwin", checkpoint_num=1000, run_dir='multi_task_robotwin-dp3_multi_task-0100_seed0', policy='dp3_multi_task'):
    """加载支持多任务的DP3策略"""
    try:
        # 添加DP3策略路径
        dp3_policy_path = dp3_path
        if dp3_policy_path not in sys.path:
            sys.path.insert(0, dp3_policy_path)

        # 导入DP3类
        from eval import DP3_policy

        # 使用现有的评估脚本中的方法
        import hydra

        # 设置配置路径 - 使用3D-Diffusion-Policy中的多任务配置
        config_path = "../3D-Diffusion-Policy/diffusion_policy_3d/config"

        # 创建多任务配置
        with hydra.initialize(config_path=config_path, version_base=None):
            # 使用多任务配置文件
            cfg = hydra.compose(config_name=f"{policy}.yaml", overrides=[f"task={task_name}"])

            # 创建DP3实例
            dp3_policy, n_obs_steps = DP3_policy(cfg, checkpoint_num=checkpoint_num, run_dir=run_dir).get_policy()
            print(f"✓ 成功加载多任务DP3策略，任务: {task_name}, checkpoint: {checkpoint_num}, run_dir: {run_dir}")
            return dp3_policy, n_obs_steps

    except Exception as e:
        print(f"✗ 加载多任务DP3策略失败: {e}")
        import traceback
        traceback.print_exc()
        return None

class MultiTaskLanguageManager:
    """多任务语言指令管理器"""

    def __init__(self, instructions_dir=None):
        self.instructions_dir = instructions_dir or os.path.join(dp3_path, "data/multi-task-data/instructions")
        self.task_instructions = {}
        self.load_all_instructions()

    def load_all_instructions(self):
        """加载所有任务的语言指令"""
        if not os.path.exists(self.instructions_dir):
            print(f"警告：语言指令目录不存在: {self.instructions_dir}")
            return

        for filename in os.listdir(self.instructions_dir):
            if filename.endswith('.json'):
                task_name = filename[:-5]  # 移除.json后缀
                instruction_file = os.path.join(self.instructions_dir, filename)
                try:
                    with open(instruction_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self.task_instructions[task_name] = data['instructions']
                    print(f"✓ 加载任务 {task_name} 的语言指令: {len(self.task_instructions[task_name])} 条")
                except Exception as e:
                    print(f"✗ 加载任务 {task_name} 的语言指令失败: {e}")

    def get_random_instruction(self, task_name: str) -> str:
        """获取指定任务的随机语言指令"""
        if task_name in self.task_instructions:
            return random.choice(self.task_instructions[task_name])
        else:
            # 如果没有找到对应的指令，返回默认指令
            print(f"✗ 获取任务 {task_name} 语言指令失败，语言指令初始化为: Complete the task using arm movement.")
            return f"Complete the {task_name} task using arm movement."

    def get_all_instructions(self, task_name: str) -> List[str]:
        """获取指定任务的所有语言指令"""
        return self.task_instructions.get(task_name, [])

class MultiTaskEnvironmentManager:
    """多任务环境管理器"""

    def __init__(self):
        self.env_manager = Env()
        self.language_manager = MultiTaskLanguageManager()
        self.current_task = None
        self.current_instruction = None

        # RoboTwin支持的任务列表
        self.available_tasks = [
            "block_hammer_beat", "block_handover", "blocks_stack_easy", "blocks_stack_hard",
            "bottle_adjust", "container_place", "diverse_bottles_pick", "dual_bottles_pick_easy",
            "dual_bottles_pick_hard", "dual_shoes_place", "empty_cup_place", "mug_hanging_easy",
            "mug_hanging_hard", "pick_apple_messy", "put_apple_cabinet", "shoe_place", "tool_adjust"
        ]

        # 评估结果统计
        self.task_results = {}
        self.reset_statistics()

    def reset_statistics(self):
        """重置统计信息"""
        for task in self.available_tasks:
            self.task_results[task] = {
                'total_episodes': 0,
                'successful_episodes': 0,
                'success_rate': 0.0
            }

    def create_task_environment(self, task_name: str, head_camera_type: str = "D435",
                              seed: int = 42, task_num: int = 1):
        """创建指定任务的环境"""
        try:
            seed_list, id_list = self.env_manager.Create_env(
                task_name=task_name,
                head_camera_type=head_camera_type,
                seed=seed,
                task_num=task_num
            )

            self.current_task = task_name
            print(f"✓ 成功创建任务环境: {task_name}, 找到 {len(seed_list)} 个有效种子")
            return seed_list, id_list

        except Exception as e:
            print(f"✗ 创建任务环境失败 {task_name}: {e}")
            return [], []

    def get_task_instruction(self, task_name: str = None) -> str:
        """获取任务的语言指令"""
        task_name = task_name or self.current_task
        if task_name:
            self.current_instruction = self.language_manager.get_random_instruction(task_name)
            return self.current_instruction
        else:
            print(f"✗ 获取任务 {task_name} 语言指令失败，语言指令初始化为: Complete the task using arm movement.")
            return "Complete the task using arm movement."

    def init_task_episode(self, task_name, seed: int, task_id: int, run_dir, checkpoint_num):
        """初始化任务回合"""
        save_dir = run_dir + '/' + task_name
        self.env_manager.Init_task_env(seed, task_id, save_dir, checkpoint_num)

    def get_observation_with_language(self):
        """获取包含语言指令的观测"""
        observation = self.env_manager.get_observation()

        # 添加当前的语言指令
        if self.current_instruction:
            observation['language'] = self.current_instruction

        return observation

    def take_action(self, action):
        """执行动作"""
        return self.env_manager.Take_action(action)

    def close_environment(self):
        """关闭环境"""
        self.env_manager.Close_env()

    def update_task_result(self, task_name: str, success: bool):
        """更新任务结果统计"""
        if task_name in self.task_results:
            self.task_results[task_name]['total_episodes'] += 1
            if success:
                self.task_results[task_name]['successful_episodes'] += 1

            # 更新成功率
            total = self.task_results[task_name]['total_episodes']
            successful = self.task_results[task_name]['successful_episodes']
            self.task_results[task_name]['success_rate'] = successful / total if total > 0 else 0.0

    def get_statistics_summary(self) -> Dict:
        """获取统计摘要"""
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
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RoboTwin多任务DP3策略评估')
    parser.add_argument('--task_name', type=str, default='multi_task_robotwin',
                        help='要评估的任务名称 (默认: multi_task_robotwin)')
    parser.add_argument('--epoch', '--checkpoint_epoch', type=int, default=200,
                        help='要加载的模型checkpoint对应的训练epoch数 (默认: 200)')
    parser.add_argument('--num_episodes', type=int, default=3,
                        help='每个任务要评估的episode数量 (默认: 3)')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子 (默认: 1)')
    parser.add_argument('--head_camera_type', type=str, default='D435',
                        help='头部相机类型 (默认: D435)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='每个episode的最大步数 (默认: 1000)')
    parser.add_argument('--run_dir', type=str, default='multi_task_robotwin-dp3_multi_task-0100_seed0',
                        help='运行的文件路径')
    parser.add_argument('--alg_name', type=str, default='dp3_multi_task',
                        help='使用的policy')
    parser.add_argument('--tasks', nargs='*', default=None,
                        help='要评估的任务列表，如果不指定则评估所有任务')
    return parser.parse_args()

def run_multi_task_evaluation(tasks_to_evaluate = None,
                             episodes_per_task = 3,
                             checkpoint_num = 200,
                             head_camera_type = "D435",
                             max_steps = 1000,
                             eval_seed=1,
                             run_dir='multi_task_robotwin-dp3_multi_task-0100_seed0', 
                             policy='dp3_multi_task',
                             task_name='multi_task_robotwin'):
    """运行多任务评估"""

    # 1. 初始化多任务环境管理器
    multi_env = MultiTaskEnvironmentManager()

    # 2. 设置要评估的任务列表
    if tasks_to_evaluate is None:
        # 默认评估所有可用任务
        tasks_to_evaluate = multi_env.available_tasks

    print(f"开始多任务评估，任务列表: {tasks_to_evaluate}")
    print(f"每个任务评估 {episodes_per_task} 个回合")

    # 3. 初始化多任务DP3策略
    print(f"正在加载多任务DP3策略，任务: {task_name}, checkpoint: {checkpoint_num}, run_dir: {run_dir}, policy: {policy}")
    dp3_policy, n_obs_steps = load_dp3_multi_task_policy(task_name=task_name, checkpoint_num=checkpoint_num, run_dir=run_dir, policy=policy)
    if dp3_policy is None:
        print("将使用随机动作作为备选方案")

    # 4. 对每个任务进行评估
    for task_idx, task_name in enumerate(tasks_to_evaluate):
        print(f"\n{'='*60}")
        print(f"评估任务 {task_idx+1}/{len(tasks_to_evaluate)}: {task_name}")
        print(f"{'='*60}")

        # 创建任务环境
        seed_list, id_list = multi_env.create_task_environment(
            task_name=task_name,
            head_camera_type=head_camera_type,
            seed=eval_seed,
            task_num=episodes_per_task
        )

        print(f"Found {len(seed_list)} valid task seeds: {seed_list}")

        if not seed_list:
            print(f"跳过任务 {task_name}：无法创建环境")
            continue

        # 对每个种子进行评估
        for episode_idx, (seed, task_id) in enumerate(zip(seed_list, id_list)):
            print(f"\n--- 任务 {task_name}, 回合 {episode_idx+1}/{len(seed_list)}, 种子: {seed} ---")

            # 获取任务的语言指令
            language_instruction = multi_env.get_task_instruction(task_name)
            print(f"语言指令: {language_instruction}")

            # # 创建图像保存目录
            # if save_images:
            #     image_dir = create_image_dir(task_name, seed, epoch)
            #     print(f"图像将保存到: {image_dir}")

            # 初始化任务回合
            # import pdb
            # pdb.set_trace()
            multi_env.init_task_episode(task_name, seed, task_id, run_dir, checkpoint_num)

            # 运行单个回合
            episode_success = run_single_episode(seed,
                multi_env, dp3_policy, language_instruction,
                max_steps, n_obs_steps
            )

            if episode_success:
                video_path = os.path.join(project_root_dir, f'eval_video/{checkpoint_num}-{run_dir}/{task_name}/seed{seed}')
                new_video_path = os.path.join(project_root_dir, f'eval_video/{checkpoint_num}-{run_dir}/{task_name}/success_seed{seed}')
                os.rename(video_path, new_video_path)
            else:
                video_path = os.path.join(project_root_dir, f'eval_video/{checkpoint_num}-{run_dir}/{task_name}/seed{seed}')
                new_video_path = os.path.join(project_root_dir, f'eval_video/{checkpoint_num}-{run_dir}/{task_name}/fail_seed{seed}')
                os.rename(video_path, new_video_path)

            # 更新统计信息
            multi_env.update_task_result(task_name, episode_success)

            print(f"回合结果: {'成功' if episode_success else '失败'}")

    # 5. 输出最终统计结果
    print_final_statistics(multi_env, checkpoint_num, run_dir)

def run_single_episode(seed, multi_env: MultiTaskEnvironmentManager,
                      dp3_policy,
                      language_instruction: str,
                      max_steps: int = 1000,
                      n_obs_steps: int = 2) -> bool:
    """运行单个评估回合"""

    obs_history = deque(maxlen=n_obs_steps)  # 保存最近n_obs_steps个时间步的观测
    episode_success = False

    for step in range(max_steps):
        # 获取包含语言指令的观测
        observation = multi_env.get_observation_with_language()

        # # 保存图像（如果需要）
        # if image_dir:
        #     save_observation_images(observation, image_dir, step)

        # 计算动作
        action = predict_action_with_language(
            dp3_policy, observation, language_instruction, obs_history, n_obs_steps
        )


        # # 如果 project_root_dir 不在全局范围，可以指定一个已知位置，例如当前工作目录
        # actions_base_dir = os.path.join(os.getcwd(), "debug_actions") # 示例：放在当前工作目录的 debug_actions 文件夹
        # os.makedirs(actions_base_dir, exist_ok=True) # 确保基础目录存在

        # # 为当前回合创建独特的子目录
        # current_episode_actions_dir = os.path.join(actions_base_dir, f"{seed}_actions")
        # os.makedirs(current_episode_actions_dir, exist_ok=True)
        # print(f"本回合的动作数据将保存到: {current_episode_actions_dir}")

        # # --- 新增代码：在每一步循环内保存动作 ---
        # action_filename = os.path.join(current_episode_actions_dir, f"action_step_{step:04d}.pkl")
        # try:
        #     with open(action_filename, 'wb') as f: # 'wb' 模式用于写入二进制文件
        #         pickle.dump(action, f)
        #     print(f"动作 {step} 保存成功到: {action_filename}") # 可选：打印确认信息
        # except Exception as e:
        #     print(f"警告: 动作保存失败 {action_filename}: {e}")
        # # ----------------------------------------


        # 执行动作
        # print(action)
        status = multi_env.take_action(action)
        print(f"步骤 {step}: 状态 = {status}")

        # 检查回合结束条件
        if status == "success":
            episode_success = True
            break
        elif status == "fail":
            break
        elif status != "run":
            break

    # 确保环境关闭
    if status == "run":
        multi_env.close_environment()

    return episode_success

def predict_action_with_language(dp3_policy, observation, language_instruction: str, obs_history, n_obs_steps):
    """使用语言指令预测动作"""
    if dp3_policy is not None:
        try:
            # 将当前观测添加到历史中
            current_obs = {
                'point_cloud': torch.from_numpy(observation['pointcloud']),
                'agent_pos': torch.from_numpy(observation['joint_action'])
            }
            obs_history.append(current_obs)

            # 如果历史不足2步，用当前观测填充
            while len(obs_history) < n_obs_steps:
                obs_history.appendleft(current_obs.copy())

            # 构建时间序列观测数据，包含语言指令
            obs_dict = {
                'point_cloud': torch.stack([obs['point_cloud'] for obs in obs_history], dim=0).unsqueeze(0),
                'agent_pos': torch.stack([obs['agent_pos'] for obs in obs_history], dim=0).unsqueeze(0),
                'language': language_instruction  # 添加语言指令
            }

            action_dict = dp3_policy.predict_action(obs_dict)
            action = action_dict['action'].squeeze(0).detach().cpu().numpy()
            return action

        except Exception as e:
            print(f"DP3策略预测失败: {e}，使用随机动作")
            return np.random.uniform(-1, 1, (1, 16))
    else:
        # 使用随机动作作为备选方案
        return np.random.uniform(-1, 1, (1, 16))

def save_observation_images(observation, image_dir: str, step: int):
    """保存观测中的图像"""
    try:
        # Front camera image
        if 'observation' in observation and 'front_camera' in observation['observation'] and 'rgb' in observation['observation']['front_camera']:
            front_image = observation['observation']['front_camera']['rgb']
            save_image(front_image, image_dir, step, "front_camera")

        # Head camera image (if exists)
        if 'observation' in observation and 'head_camera' in observation['observation'] and 'rgb' in observation['observation']['head_camera']:
            head_image = observation['observation']['head_camera']['rgb']
            save_image(head_image, image_dir, step, "head_camera")

        # Wrist camera image (if exists)
        if 'observation' in observation and 'wrist_camera' in observation['observation'] and 'rgb' in observation['observation']['wrist_camera']:
            wrist_image = observation['observation']['wrist_camera']['rgb']
            save_image(wrist_image, image_dir, step, "wrist_camera")

    except KeyError as e:
        print(f"警告：无法访问图像数据 - {e}")

def print_final_statistics(multi_env: MultiTaskEnvironmentManager, checkpoint_num, run_dir):
    """打印最终统计结果"""
    stats = multi_env.get_statistics_summary()

    print(f"\n{'='*80}")
    print("多任务评估结果统计")
    print(f"{'='*80}")

    print(f"总体成功率: {stats['overall_success_rate']:.2%}")
    print(f"总回合数: {stats['total_episodes']}")
    print(f"成功回合数: {stats['total_successful']}")

    print(f"\n各任务详细结果:")
    print(f"{'任务名称':<25} {'总回合':<8} {'成功回合':<8} {'成功率':<8}")
    print("-" * 60)

    for task_name, result in stats['task_results'].items():
        if result['total_episodes'] > 0:
            print(f"{task_name:<25} {result['total_episodes']:<8} {result['successful_episodes']:<8} {result['success_rate']:.2%}")

    # 保存结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(project_root_dir, f"eval_video/{checkpoint_num}-{run_dir}/multi_task_results_{timestamp}.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_file}")

# 主程序入口
if __name__ == "__main__":
    args = parse_arguments()

    # 如果没有指定任务，使用默认任务列表
    if args.tasks is None:
        tasks_to_evaluate = [
            "block_hammer_beat", "block_handover", "blocks_stack_easy", "blocks_stack_hard",
            "bottle_adjust", "container_place", "diverse_bottles_pick", "dual_bottles_pick_easy",
            "dual_bottles_pick_hard", "dual_shoes_place", "empty_cup_place", "mug_hanging_easy",
            "mug_hanging_hard", "pick_apple_messy", "put_apple_cabinet", "shoe_place", "tool_adjust"
        ]
    else:
        tasks_to_evaluate = args.tasks

    print("开始RoboTwin多任务DP3评估")
    print(f"评估任务: {tasks_to_evaluate}")
    print(f"Policy: {args.alg_name}")

    # 运行多任务评估
    run_multi_task_evaluation(
        tasks_to_evaluate=tasks_to_evaluate,
        episodes_per_task=args.num_episodes,
        checkpoint_num=args.epoch,
        head_camera_type=args.head_camera_type,
        max_steps=args.max_steps,
        eval_seed=args.seed,
        run_dir=args.run_dir,
        policy=args.alg_name,
        task_name=args.task_name
    )