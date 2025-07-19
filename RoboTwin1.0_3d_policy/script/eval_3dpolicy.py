import sys
# sys.path.append('./') 
import torch  
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
# current_file_path = os.path.abspath(__file__)
# parent_directory = os.path.dirname(current_file_path)
# 为了确保 eval_3dpolicy.py 内部的路径解析正确，我们获取它自己的文件路径。
current_eval_script_path = os.path.abspath(__file__)
eval_script_dir = os.path.dirname(current_eval_script_path)
class Env:
    def __init__(self):
        pass
    @staticmethod
    def class_decorator(task_name):
        envs_module = importlib.import_module(f'envs.{task_name}')
        try:
            env_class = getattr(envs_module, task_name)
            env_instance = env_class()
        except:
            raise SystemExit("No Task")
        return env_instance
    @staticmethod
    def get_camera_config(camera_type):
        # camera_config_path = os.path.join(parent_directory, 'script/task_config/_camera_config.yml')
        camera_config_path = '/data/sea_disk0/cuihz/code/RoboTwin/task_config/_camera_config.yml'

        assert os.path.isfile(camera_config_path), "task config file is missing"

        with open(camera_config_path, 'r', encoding='utf-8') as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        assert camera_type in args, f'camera {camera_type} is not defined'
        return args[camera_type]
    def dual_arm(self):
        return self.task.get_dual_arm()
    def Create_env(self,task_name,head_camera_type,seed,task_num):
        task_config_file_path = os.path.join(eval_script_dir, '..', 'task_config', f'{task_name}.yml')
        with open(task_config_file_path, 'r', encoding='utf-8') as f:
            self.args = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.args['head_camera_type'] = head_camera_type
        head_camera_config = Env.get_camera_config(self.args['head_camera_type'])
        self.args['head_camera_fovy'] = head_camera_config['fovy']
        self.args['head_camera_w'] = head_camera_config['w']
        self.args['head_camera_h'] = head_camera_config['h']
        head_camera_config = 'fovy' + str(self.args['head_camera_fovy']) + '_w' + str(self.args['head_camera_w']) + '_h' + str(self.args['head_camera_h'])
        
        wrist_camera_config = Env.get_camera_config(self.args['wrist_camera_type'])
        self.args['wrist_camera_fovy'] = wrist_camera_config['fovy']
        self.args['wrist_camera_w'] = wrist_camera_config['w']
        self.args['wrist_camera_h'] = wrist_camera_config['h']
        wrist_camera_config = 'fovy' + str(self.args['wrist_camera_fovy']) + '_w' + str(self.args['wrist_camera_w']) + '_h' + str(self.args['wrist_camera_h'])

        front_camera_config = Env.get_camera_config(self.args['front_camera_type'])
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
        self.task= Env.class_decorator(task_name)
        self.st_seed = seed
        self.task.set_actor_pose(True)
        return self.Check_seed(task_num)
    def Init_task_env(self,seed,id):
        self.env_state=0 #0:running 1:success 2:fail
        self.step=0
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
    def Detect_env_state(self):
        if self.step>self.task.get_step_lim():
            self.env_state=2
        if self.task.check_success():
            self.env_state=1
        if self.task.get_actor_pose()==False:
            self.env_state=2
    def Take_action(self,action):
        # actions=[]
        # actions.append(action)
        # actions=np.array(actions)
        # self.task.apply_action(actions)
        actions=action
        self.task.apply_action(actions)
        self.step+=actions.shape[0]
        self.Detect_env_state()
        if self.env_state==1:
            print('Task Success!')
            self.Close_env()
            return "success"
        elif self.env_state==2:
            print('Task Failed!')
            self.Close_env()
            return "fail"
        else:
            return "run"
    def Close_env(self):
        self.task.close()
        if self.task.render_freq:
            self.task.viewer.close()
        print ('Env Closed!')
        self.task._take_picture()
    def get_observation(self):
        return self.task.Get_observation()