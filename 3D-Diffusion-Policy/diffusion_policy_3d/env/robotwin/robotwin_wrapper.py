import numpy as np
import gym
from gym import spaces
import sys
import os
import traceback
import pdb

# Add RoboTwin environment path
robotwin_env_dir_path = os.path.dirname(__file__)
if robotwin_env_dir_path not in sys.path:
    sys.path.append(robotwin_env_dir_path)

try:
    from .robotwin_env.block_hammer_beat import block_hammer_beat
except ImportError:
    # Fallback if RoboTwin environment is not available
    print("Warning: RoboTwin environment not found. Using mock environment.")
    block_hammer_beat = None

# print(robotwin_env_path)

import yaml
import importlib
# current_file_path = os.path.abspath(__file__)
# parent_directory = os.path.dirname(current_file_path)
# 为了确保 eval_3dpolicy.py 内部的路径解析正确，我们获取它自己的文件路径。
current__path = os.path.abspath(__file__)
task_config_dir = os.path.dirname(current__path)

class RobotwinEnv(gym.Env):
    """
    RoboTwin environment wrapper for DP3 integration.
    Provides standardized interface for RoboTwin tasks.
    """
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name="block_hammer_beat", use_point_cloud=True, seed=42, task_num=5, **kwargs):
        super(RobotwinEnv, self).__init__()
        
        # self.use_point_crop = use_point_crop

        self.task_name = task_name
        self.use_point_cloud = use_point_cloud

        
        
        # # Initialize the specific RoboTwin task
        # if task_name == "block_hammer_beat":
        #     if block_hammer_beat is None:
        #         raise ImportError("RoboTwin environment not available. Please ensure RoboTwin1.0_3d_policy is properly installed.")
        #     self.env = block_hammer_beat()
        # else:
        #     raise NotImplementedError(f"Task {task_name} not implemented yet")
        

        self.seed_list, self.id_list = self.Create_env(task_name=task_name, seed=seed, task_num=task_num)
        
        # # Setup the environment
        # self.env.setup_demo(now_ep_num=0, is_test=True, ** self.args)
        
        # Define observation and action spaces based on DP3 requirements
        self._setup_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.max_episode_steps = getattr(self.task, 'step_lim', 400)

    @staticmethod
    def class_decorator(task_name):
        envs_module = importlib.import_module(f'robotwin_env.{task_name}')
        try:
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
        except:
            raise SystemExit("No Task")
        return env_instance
    
    def get_camera_config(self, camera_type):
        # camera_config_path = os.path.join(parent_directory, 'script/task_config/_camera_config.yml')
        camera_config_path = os.path.join(task_config_dir, 'task_config', '_camera_config.yml')

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, 'r', encoding='utf-8') as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        assert camera_type in args, f'camera {camera_type} is not defined'
        return args[camera_type]
    
    def Create_env(self,task_name,seed,task_num):
        task_config_file_path = os.path.join(task_config_dir, 'task_config', f'{task_name}.yml')
        with open(task_config_file_path, 'r', encoding='utf-8') as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        head_camera_config = self.get_camera_config(self.args['head_camera_type'])
        self.args['head_camera_fovy'] = head_camera_config['fovy']
        self.args['head_camera_w'] = head_camera_config['w']
        self.args['head_camera_h'] = head_camera_config['h']
        head_camera_config = 'fovy' + str(self.args['head_camera_fovy']) + '_w' + str(self.args['head_camera_w']) + '_h' + str(self.args['head_camera_h'])
        
        wrist_camera_config = self.get_camera_config(self.args['wrist_camera_type'])
        self.args['wrist_camera_fovy'] = wrist_camera_config['fovy']
        self.args['wrist_camera_w'] = wrist_camera_config['w']
        self.args['wrist_camera_h'] = wrist_camera_config['h']
        wrist_camera_config = 'fovy' + str(self.args['wrist_camera_fovy']) + '_w' + str(self.args['wrist_camera_w']) + '_h' + str(self.args['wrist_camera_h'])

        front_camera_config = self.get_camera_config(self.args['front_camera_type'])
        self.args['front_camera_fovy'] = front_camera_config['fovy']
        self.args['front_camera_w'] = front_camera_config['w']
        self.args['front_camera_h'] = front_camera_config['h']
        front_camera_config = 'fovy' + str(self.args['front_camera_fovy']) + '_w' + str(self.args['front_camera_w']) + '_h' + str(self.args['front_camera_h'])

        # output camera config
        print('============= Camera Config =============\n')
        print('Head Camera Config:\n    type: '+ str(self.args['head_camera_type']) + '\n    fovy: ' + str(self.args['head_camera_fovy']) + '\n    camera_w: ' + str(self.args['head_camera_w']) + '\n    camera_h: ' + str(self.args['head_camera_h']))
        print('Wrist Camera Config:\n    type: '+ str(self.args['wrist_camera_type']) + '\n    fovy: ' + str(self.args['wrist_camera_fovy']) + '\n    camera_w: ' + str(self.args['wrist_camera_w']) + '\n    camera_h: ' + str(self.args['wrist_camera_h']))
        print('Front Camera Config:\n    type: '+ str(self.args['front_camera_type']) + '\n    fovy: ' + str(self.args['front_camera_fovy']) + '\n    camera_w: ' + str(self.args['front_camera_w']) + '\n    camera_h: ' + str(self.args['front_camera_h']))
        print('\n=======================================')
        self.task= RobotwinEnv.class_decorator(task_name)
        self.st_seed = seed
        self.task.set_actor_pose(True)
        return self.Check_seed(task_num)

    def Init_task_env(self,seed,id):
        self.env_state=0 #0:running 1:success 2:fail
        self.task.setup_demo(now_ep_num=id, seed = seed, is_test = True, ** self.args)

    def Check_seed(self,test_num):
        expert_check=True
        print("Task name: ", self.args["task_name"])
        suc_seed_list=[]
        now_id_list = []
        succ_tnt=0
        now_seed=self.st_seed
        now_id = 0
        self.task.cus=0
        self.task.test_num = 0
        while succ_tnt<test_num:
            render_freq = self.args['render_freq']
            self.args['render_freq'] = 0
            if expert_check:
                try:
                    self.task.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** self.args)
                    self.task.play_once()
                    self.task.close()
                    suc_seed_list.append(now_seed)
                    now_id_list.append(now_id)
                    now_id += 1
                    succ_tnt += 1
                    now_seed += 1
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    print(' -------------')
                    print('Error: ', stack_trace)
                    print(' -------------')
                    self.task.close()
                    now_seed += 1
                    self.args['render_freq'] = render_freq
                    print('error occurs !')
                    continue
            self.args['render_freq'] = render_freq
        return suc_seed_list, now_id_list
        
    def _setup_spaces(self):
        """Setup observation and action spaces to match DP3 expectations"""
        
        # Action space: 14-dimensional for dual arm robot (7 per arm)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32
        )
        
        # Observation space
        obs_spaces = {}
        
        if self.use_point_cloud:
            # Point cloud: 1024 points with 6 features (xyz + rgb)
            obs_spaces['point_cloud'] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1024, 6),
                dtype=np.float32
            )
        
        # Agent position: 14-dimensional joint state
        obs_spaces['agent_pos'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Dict(obs_spaces)
    
    def reset(self, **kwargs):
        """Reset the environment and return initial observation"""
        if not self.seed_list or not self.id_list:
            seed = 42
            id = 0
        else:
            seed = self.seed_list.pop(0)
            id = self.id_list.pop(0)

        self.Init_task_env(seed, id)
        # Reset the RoboTwin environment
        episode_idx = kwargs.get('episode_idx')

        if episode_idx is None:
            # 抛出一个自定义的错误，明确指出缺少哪个参数
            raise ValueError("Required keyword argument 'episode_idx' not provided to reset() method.")

        self.current_step = 0
        
        # Get initial observation
        obs = self._get_observation()
        # print('============= reset_obs =============\n')
        # print(obs)
        # print('============= reset_obs =============\n')

        return obs
    
    def step(self, action):
        """Execute one step in the environment"""
        # Ensure action is the right shape and type
        action = np.array(action, dtype=np.float32)
        if action.shape != (14,):
            raise ValueError(f"Expected action shape (14,), got {action.shape}")
        
        # print(action)
        # Apply action to the environment
        success = self.task.apply_action(action)
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward and done status
        reward = 1.0 if success else 0.0
        done = success or self.current_step >= self.max_episode_steps or not self.task.get_actor_pose()
        
        # Create info dict
        info = {
            'success': success,
            'goal_achieved': success,
            'step': self.current_step
        }
        
        self.current_step += 1
        
        return obs, reward, done, info
    
    def _get_observation(self):
        """Get current observation from the environment"""
        # Get raw observation from RoboTwin environment
        raw_obs = self.task.Get_observation()
        # print('============= raw_obs =============\n')
        # print(raw_obs)
        # print('============= raw_obs =============\n')
        
        obs_dict = {}
        
        if self.use_point_cloud:
            # Extract point cloud (1024, 6) - xyz + rgb
            point_cloud = raw_obs['pointcloud']
            if point_cloud.shape[0] != 1024:
                # Resample to 1024 points if needed
                point_cloud = self._resample_point_cloud(point_cloud, 1024)
            obs_dict['point_cloud'] = point_cloud.astype(np.float32)
        
        # Extract agent position (joint states)
        agent_pos = raw_obs['joint_action']
        if agent_pos.shape[0] != 14:
            raise ValueError(f"Expected agent_pos shape (14,), got {agent_pos.shape}")
        obs_dict['agent_pos'] = agent_pos.astype(np.float32)
        
        return obs_dict
    
    def _resample_point_cloud(self, point_cloud, target_num_points):
        """Resample point cloud to target number of points"""
        current_num_points = point_cloud.shape[0]
        
        if current_num_points == target_num_points:
            return point_cloud
        elif current_num_points > target_num_points:
            # Downsample using random selection
            indices = np.random.choice(current_num_points, target_num_points, replace=False)
            return point_cloud[indices]
        else:
            # Upsample by repeating points
            repeat_factor = target_num_points // current_num_points
            remainder = target_num_points % current_num_points
            
            repeated = np.tile(point_cloud, (repeat_factor, 1))
            if remainder > 0:
                extra_indices = np.random.choice(current_num_points, remainder, replace=False)
                extra_points = point_cloud[extra_indices]
                repeated = np.vstack([repeated, extra_points])
            
            return repeated
    
    def render(self, mode='rgb_array'):
        """Render the environment"""
        if mode == 'rgb_array':
            # Get RGB image from the environment
            obs = self.task.Get_observation()
            if 'observation' in obs and 'head_camera' in obs['observation']:
                rgb_img = obs['observation']['head_camera']['rgb']
                return rgb_img
            else:
                # Return a dummy image if camera data is not available
                return np.zeros((84, 84, 3), dtype=np.uint8)
        else:
            raise NotImplementedError(f"Render mode {mode} not supported")
    
    def close(self):
        """Close the environment"""
        pass
    
    def seed(self, seed=None):
        """Set random seed"""
        if seed is not None:
            np.random.seed(seed)
        return [seed]
    
    def get_success_metric(self):
        """Get success metric for evaluation"""
        return self.task.check_success()
