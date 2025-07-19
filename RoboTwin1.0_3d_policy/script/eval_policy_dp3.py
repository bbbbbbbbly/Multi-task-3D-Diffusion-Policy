import sys
sys.path.insert(0, './policy/3D-Diffusion-Policy/3D-Diffusion-Policy')
sys.path.append('./')

import torch  
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
import hydra
import pathlib

from dp3_policy import *
import cv2

import yaml
from datetime import datetime
import importlib
from script.eval_3dpolicy import Env
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
def create_image_dir(task_name, seed):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_dir = f"./observation_images/{task_name}_{seed}_{timestamp}"
    os.makedirs(image_dir, exist_ok=True)
    return image_dir
def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, '../task_config/_camera_config.yml')

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]

def load_model(model_path):
    model = torch.load(model_path)
    model.eval() 
    return model

TASK = None

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        '/data/sea_disk0/cuihz/code/RoboTwin/policy/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d', 'config'))
)
def main(cfg):
    taskname= 'block_hammer_beat'
    head_camera_type='D435'
    checkpoint_num=3000
    seed=0
    task_num=1
    env_manager = Env()
    seed_list, id_list = env_manager.Create_env(
    task_name=taskname,
    head_camera_type="D435", 
    seed=seed,
    task_num=task_num
    )
    success=0
    dp3=DP3(cfg,checkpoint_num)
    for i, (seed, task_id) in enumerate(zip(seed_list, id_list)):
        print(f"\nExecuting task {i+1}/{len(seed_list)}, seed: {seed}")
        
        # Create image save directory for current task
        image_dir = create_image_dir(taskname, seed)
        print(f"Images will be saved to: {image_dir}")
        
        # Initialize task environment
        env_manager.Init_task_env(seed, task_id)
        
        # Run task loop
        max_steps = 1000
        for step in range(max_steps):
            observation = env_manager.get_observation()
            obs = dict()
            obs['point_cloud'] = observation['pointcloud']
            if env_manager.dual_arm():
                obs['agent_pos'] = observation['joint_action']
                assert obs['agent_pos'].shape[0] == 14, 'agent_pose shape, error'
            else:
                obs['agent_pos'] = observation['joint_action']
                assert obs['agent_pos'].shape[0] == 7, 'agent_pose shape, error'
            actions = dp3.get_action(obs)
            status = env_manager.Take_action(actions)
            print(f"Step {step}: status = {status}")
            if status == 'success':
                success += 1
                print(f"Task completed successfully!", success)
            if status != "run":
                break
        dp3.env_runner.reset_obs()
        if status == "run":
            env_manager.Close_env()
    print(f"Task completed, success rate: {success}/{len(seed_list)}")
if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    main()
