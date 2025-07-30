import numpy as np
import os
# import cv2
# from PIL import Image
from datetime import datetime
from script.eval_3dpolicy import Env
import sys
import torch
import pathlib
import dill
from collections import deque
import pdb
import pickle
import argparse

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# print(current_script_path)
# 获取脚本所在的目录，也就是 RoboTwin2.0_3D_policy/ 目录
project_root_dir = os.path.dirname(current_script_path)
# print(project_root_dir) # /data/sea_disk0/wushr/3D-Policy/DP3Encoder-to-Uni3D/3D-Diffusion-Policy/RoboTwin2.0_3D_policy
# 将项目根目录添加到sys.path，以便正确导入项目内部模块
sys.path.append(project_root_dir)

# 添加3D-Diffusion-Policy路径以导入DP3相关模块
dp3_path = os.path.join(os.path.dirname(project_root_dir), '3D-Diffusion-Policy')
# print(dp3_path) # /data/sea_disk0/wushr/3D-Policy/DP3Encoder-to-Uni3D/3D-Diffusion-Policy/3D-Diffusion-Policy
sys.path.insert(0, dp3_path)

from diffusion_policy_3d.common.pytorch_util import dict_apply

def load_dp3_policy(task_name="beat_block_hammer", checkpoint_num=400, run_dir='robotwin2_beat_block_hammer-dp3_robotwin2-demo_clean-9999_seed0', policy='dp3_robotwin2'):
    """加载DP3策略的简化函数"""
    try:
        # 添加DP3策略路径
        dp3_policy_path = dp3_path
        if dp3_policy_path not in sys.path:
            sys.path.insert(0, dp3_policy_path)

        # 导入DP3类
        from eval import DP3_policy

        # 使用现有的评估脚本中的方法
        import hydra

        # 设置配置路径 - 使用3D-Diffusion-Policy中的DP3配置
        config_path = "../3D-Diffusion-Policy/diffusion_policy_3d/config"

        # 创建一个基本配置
        with hydra.initialize(config_path=config_path, version_base=None):
            cfg = hydra.compose(config_name=f"{policy}.yaml", overrides=[f"task_name={task_name}"])

            # 创建DP3实例
            dp3_policy, n_obs_steps = DP3_policy(cfg, checkpoint_num=checkpoint_num, run_dir=run_dir).get_policy()
            print(f"✓ 成功加载DP3策略，任务: {task_name}, checkpoint: {checkpoint_num}, run_dir: {run_dir}")
            return dp3_policy, n_obs_steps

    except Exception as e:
        print(f"✗ 加载DP3策略失败: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    parser = argparse.ArgumentParser(description='RoboTwin2.0单任务DP3策略评估')
    parser.add_argument('--task_name', type=str, default='beat_block_hammer',
                        help='要评估的任务名称 (默认: beat_block_hammer)')
    parser.add_argument('--epoch', '--checkpoint_epoch', type=int, default=400,
                        help='要加载的模型checkpoint对应的训练epoch数 (默认: 400)')
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='要评估的episode数量 (默认: 20)')
    parser.add_argument('--seed', type=int, default=1,
                        help='随机种子 (默认: 1)')
    parser.add_argument('--head_camera_type', type=str, default='D435',
                        help='头部相机类型 (默认: D435)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='每个episode的最大步数 (默认: 1000)')
    parser.add_argument('--run_dir', type=str, default='robotwin2_beat_block_hammer-dp3_robotwin2-9999_seed0',
                        help='运行的文件路径')
    parser.add_argument('--alg_name', type=str, default='dp3_robotwin2',
                        help='使用的policy')
    parser.add_argument('--task_config', type=str, default='demo_clean',
                        help='任务配置 (默认: demo_clean)')
    parser.add_argument('--instruction_type', type=str, default='unseen',
                        help='指令类型 (默认: unseen)')
    return parser.parse_args()

def add_noise(data, noise_std=0.01, clip_range=0.02):
    """
    为点云添加经过clip的高斯噪声

    Args:
        point_cloud: 输入的点云数据 (numpy array)
        noise_std: 高斯噪声的标准差，默认0.01
        clip_range: 噪声的clip范围，默认0.02 (即噪声被限制在[-0.02, 0.02]范围内)

    Returns:
        添加噪声后的点云数据
    """
    if data is None:
        return None

    # 生成与点云同样形状的高斯噪声
    noise = np.random.normal(0, noise_std, data.shape)

    # 对噪声进行clip，避免噪声过大
    noise = np.clip(noise, -clip_range, clip_range)

    # 将噪声添加到点云上
    noisy_data = data + noise

    return noisy_data

def run_single_task_evaluation(task_name, epoch, num_episodes, seed=1,
                              head_camera_type="D435", max_steps=1000, run_dir='robotwin2_beat_block_hammer-dp3_robotwin2-9999_seed0',
                              policy='dp3_robotwin2', task_config='demo_clean', instruction_type='unseen'):
    """运行单任务评估"""

    # 1. Create environment manager instance
    env_manager = Env()

    # 2. 初始化DP3策略
    print(f"正在加载DP3策略，任务: {task_name}, checkpoint: {epoch}, run_dir: {run_dir}, policy: {policy}")
    dp3_policy, n_obs_steps = load_dp3_policy(task_name=task_name, checkpoint_num=epoch, run_dir=run_dir, policy=policy)
    if dp3_policy is None:
        print("将使用随机动作作为备选方案")

    # 3. Create specific task environment and validate seeds
    result = env_manager.Create_env(
        task_name=task_name,
        head_camera_type=head_camera_type,
        seed=seed,
        task_num=num_episodes,
        instruction_type=instruction_type,
        task_config=task_config
    )

    if not result:
        print("Failed to get valid seeds")
        return 0, 0

    seed_list, id_list, episode_info_list_total = result
    print(f"Found {len(seed_list)} valid task seeds: {seed_list}")

    success_total = 0

    # 4. Run tasks for each valid seed
    for i, (episode_seed, task_id, episode_info_list) in enumerate(zip(seed_list, id_list, episode_info_list_total)):
        print(f"\nExecuting task {i+1}/{len(seed_list)}, seed: {episode_seed}")

        # Initialize task environment
        inst = env_manager.Init_task_env(episode_seed, task_id, episode_info_list, run_dir, epoch)

        # Run task loop
        obs_history = deque(maxlen=n_obs_steps)  # 保存最近2个时间步的观测
        observation = env_manager.get_observation()
        # 将当前观测添加到历史中
        current_obs = {
            'point_cloud': torch.from_numpy(observation['pointcloud']),
            'agent_pos': torch.from_numpy(observation['joint_action']['vector'])
        }
        obs_history.append(current_obs)

        # 如果历史不足，用当前观测填充
        while len(obs_history) < n_obs_steps:
            # 复制当前观测来填充历史
            obs_history.appendleft(current_obs.copy())

        for step in range(max_steps):
            # Get observation
            # import pdb
            # pdb.set_trace()
            
            

            # Calculate action based on observation
            if dp3_policy is not None:
                try:
                    # # 将当前观测添加到历史中
                    # current_obs = {
                    #     'point_cloud': torch.from_numpy(observation['pointcloud']),
                    #     'agent_pos': torch.from_numpy(observation['joint_action']['vector'])
                    # }
                    # obs_history.append(current_obs)

                    # # 如果历史不足，用当前观测填充
                    # while len(obs_history) < n_obs_steps:
                    #     # 复制当前观测来填充历史
                    #     obs_history.appendleft(current_obs.copy())

                    # 构建时间序列观测数据 (batch=1, time=2, ...)
                    obs_dict = {
                        'point_cloud': torch.stack([obs['point_cloud'] for obs in obs_history], dim=0).unsqueeze(0),
                        'agent_pos': torch.stack([obs['agent_pos'] for obs in obs_history], dim=0).unsqueeze(0)
                    }

                    action_dict = dp3_policy.predict_action(obs_dict)
                    action = action_dict['action'].squeeze(0).detach().cpu().numpy()

                except Exception as e:
                    print(f"Step {step}: DP3策略预测失败: {e}，使用随机动作")
                    action = np.random.uniform(-1, 1, (1, 16))
                    action = np.array(action)
            else:
                # 使用随机动作作为备选方案
                action = np.random.uniform(-1, 1, (1, 16))
                action = np.array(action)
                print(f"Step {step}: 使用随机动作，形状: {action.shape}")
            
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
            # # -----------------------------------

            status, obs_history = env_manager.Take_action(action, obs_history)
            print(f"Step {step}: status = {status}")

            if status == "success":
                success_total += 1
                # video_path = os.path.join(project_root_dir, f'eval_video/{epoch}-{run_dir}/seed{episode_seed}')
                # new_video_path = os.path.join(project_root_dir, f'eval_video/{epoch}-{run_dir}/success_seed{episode_seed}')
                # if os.path.exists(video_path):
                #     os.rename(video_path, new_video_path)
            if status == "fail":
                # video_path = os.path.join(project_root_dir, f'eval_video/{epoch}-{run_dir}/seed{episode_seed}')
                # new_video_path = os.path.join(project_root_dir, f'eval_video/{epoch}-{run_dir}/fail_seed{episode_seed}')
                # if os.path.exists(video_path):
                #     os.rename(video_path, new_video_path)
                pass

            # Exit loop if task is completed or failed
            if status != "run":
                break

        # Ensure environment is closed
        if status == "run":
            env_manager.Close_env()

    print(f"Total success rate: {success_total}/{len(seed_list)}")
    print(f"\nAll tasks completed, images have been saved.")

    return success_total, len(seed_list)

# 主程序入口
if __name__ == "__main__":
    args = parse_arguments()

    print("开始RoboTwin2.0单任务DP3评估")
    print(f"Policy: {args.alg_name}")

    # 运行单任务评估
    success_count, total_count = run_single_task_evaluation(
        task_name=args.task_name,
        epoch=args.epoch,
        num_episodes=args.num_episodes,
        seed=args.seed,
        head_camera_type=args.head_camera_type,
        max_steps=args.max_steps,
        run_dir=args.run_dir,
        policy=args.alg_name,
        task_config=args.task_config,
        instruction_type=args.instruction_type
    )

    print(f"\n{args.epoch}-{args.task_name}最终结果: {success_count}/{total_count} = {success_count/total_count:.2%}")