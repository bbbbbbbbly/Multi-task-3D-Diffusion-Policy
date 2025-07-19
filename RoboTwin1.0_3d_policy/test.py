import numpy as np
import os
import cv2
from PIL import Image
from datetime import datetime
from script.eval_3dpolicy_new import Env
import sys
import torch
import pathlib
import dill
from collections import deque
import pdb
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

from diffusion_policy_3d.common.pytorch_util import dict_apply

def load_dp3_policy(task_name="block_hammer_beat", checkpoint_num=400, run_dir='robotwin_block_hammer_beat-dp3-9999_seed0', policy='dp3'):
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
            cfg = hydra.compose(config_name=f"{policy}.yaml", overrides=[f"task=robotwin_{task_name}"])

            # 创建DP3实例
            dp3_policy, n_obs_steps = DP3_policy(cfg, checkpoint_num=checkpoint_num, run_dir=run_dir).get_policy()
            print(f"✓ 成功加载DP3策略，任务: {task_name}, checkpoint: {checkpoint_num}, run_dir: {run_dir}")
            return dp3_policy, n_obs_steps

    except Exception as e:
        print(f"✗ 加载DP3策略失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# def load_RoboTwin_runner():
#     """加载RoboTwin的runner"""
#     try:
#         # 添加路径
#         runner_path = os.path.join(project_root_dir, 'policy/3D-Diffusion-Policy/3D-Diffusion-Policy')
#         if runner_path not in sys.path:
#             sys.path.insert(0, runner_path)

#         # 导入runner
#         from dp3_policy import 

#         # 使用现有的评估脚本中的方法
#         import hydra

#         # 设置配置路径 - 使用RoboTwin中的DP3配置
#         config_path = os.path.join(dp3_policy_path, 'diffusion_policy_3d', 'config')

#         # 创建一个基本配置
#         with hydra.initialize(config_path=config_path, version_base=None):
#             cfg = hydra.compose(config_name="dp3.yaml", overrides=["task=block_hammer_beat"])

#             # # 修改配置以匹配我们的任务
#             # # cfg.training.seed = 0
#             # cfg.checkpoint_num = 2800
#             # cfg.expert_data_num = 0
#             # cfg.head_camera_type = "D435"

#             # 创建DP3实例，使用checkpoint 2800
#             dp3_policy = DP3_policy(cfg, checkpoint_num=2800).get_policy()
#             print(f"✓ 成功加载DP3策略，checkpoint: 2800")
#             return dp3_policy

#     except Exception as e:
#         print(f"✗ 加载DP3策略失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

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
    parser = argparse.ArgumentParser(description='RoboTwin单任务DP3策略评估')
    parser.add_argument('--task_name', type=str, default='block_hammer_beat',
                        help='要评估的任务名称 (默认: block_hammer_beat)')
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
    parser.add_argument('--run_dir', type=str, default='robotwin_block_hammer_beat-dp3-9999_seed0',
                        help='运行的文件路径')
    parser.add_argument('--alg_name', type=str, default='dp3',
                        help='使用的policy')
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
                              head_camera_type="D435", max_steps=1000, run_dir='robotwin_block_hammer_beat-dp3-9999_seed0', policy='dp3'):
    """运行单任务评估"""

    # 1. Create environment manager instance
    env_manager = Env()

    # 2. 初始化DP3策略
    print(f"正在加载DP3策略，任务: {task_name}, checkpoint: {epoch}, run_dir: {run_dir}, policy: {policy}")
    dp3_policy, n_obs_steps = load_dp3_policy(task_name=task_name, checkpoint_num=epoch, run_dir=run_dir, policy=policy)
    if dp3_policy is None:
        print("将使用随机动作作为备选方案")

    # 3. Create specific task environment and validate seeds
    seed_list, id_list = env_manager.Create_env(
        task_name=task_name,
        head_camera_type=head_camera_type,
        seed=seed,
        task_num=num_episodes
    )

    print(f"Found {len(seed_list)} valid task seeds: {seed_list}")

    success_total = 0

    # 4. Run tasks for each valid seed
    for i, (episode_seed, task_id) in enumerate(zip(seed_list, id_list)):
        print(f"\nExecuting task {i+1}/{len(seed_list)}, seed: {episode_seed}")

        # # Create image save directory for current task
        # image_dir = create_image_dir(task_name, episode_seed, epoch)
        # print(f"Images will be saved to: {image_dir}")

        # Initialize task environment
        # import pdb
        # pdb.set_trace()
        env_manager.Init_task_env(episode_seed, task_id, run_dir, epoch)

        # 重置DP3策略的观测历史（如果策略加载成功）
        # if dp3_policy is not None and hasattr(dp3_policy, 'env_runner'):
        #     dp3_policy.env_runner.reset_obs()

        # Run task loop
        obs_history = deque(maxlen=n_obs_steps)  # 保存最近2个时间步的观测
        for step in range(max_steps):
            # Get observation
            observation = env_manager.get_observation()
            # print(observation['pointcloud'])
            # pdb.set_trace()
            
            # 为point cloud添加高斯噪声来缓解关键帧卡顿问题
            # if 'pointcloud' in observation and observation['pointcloud'] is not None:
            #     # 添加小的高斯噪声，标准差0.01，clip范围0.02
            #     observation['pointcloud'] = add_noise(
            #         observation['pointcloud'], 
            #         noise_std=0.002, 
            #         clip_range=0.02
            #     )
            #     print(f"Step {step}: 已为point cloud添加高斯噪声")

            # Calculate action based on observation
            if dp3_policy is not None:
                try:
                    # 将当前观测添加到历史中
                    current_obs = {
                        'point_cloud': torch.from_numpy(observation['pointcloud']),
                        'agent_pos': torch.from_numpy(observation['joint_action'])
                    }
                    obs_history.append(current_obs)

                    # 如果历史不足，用当前观测填充
                    while len(obs_history) < n_obs_steps:
                        # 复制当前观测来填充历史
                        obs_history.appendleft(current_obs.copy())

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

            # if action is not None:
            #     # 添加小的高斯噪声，标准差0.01，clip范围0.02
            #     action = add_noise(action, 0.0008, 0.02)
            #     print(f"Step {step}: 已为action添加高斯噪声")
            
            status = env_manager.Take_action(action)
            print(f"Step {step}: status = {status}")

            if status == "success":
                success_total += 1
                video_path = os.path.join(project_root_dir, f'eval_video/{epoch}-{run_dir}/seed{episode_seed}')
                new_video_path = os.path.join(project_root_dir, f'eval_video/{epoch}-{run_dir}/success_seed{episode_seed}')
                os.rename(video_path, new_video_path)
            if status == "fail":
                video_path = os.path.join(project_root_dir, f'eval_video/{epoch}-{run_dir}/seed{episode_seed}')
                new_video_path = os.path.join(project_root_dir, f'eval_video/{epoch}-{run_dir}/fail_seed{episode_seed}')
                os.rename(video_path, new_video_path)

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

    print("开始RoboTwin单任务DP3评估")
    # print(f"任务名称: {args.task_name}")
    # print(f"Checkpoint epoch: {args.epoch}")
    # print(f"评估回合数: {args.num_episodes}")
    # print(f"随机种子: {args.seed}")
    # print(f"运行任务: {args.run_dir}")
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
        policy=args.alg_name
    )

    print(f"\n{args.epoch}-{args.task_name}最终结果: {success_count}/{total_count} = {success_count/total_count:.2%}")