import sys
sys.path.append('./') 
import torch  
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque
import traceback
import json
import yaml
from datetime import datetime
import importlib
import dill
import subprocess
from pathlib import Path
from argparse import ArgumentParser
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
        camera_config_path = os.path.join(eval_script_dir, '../task_config/_camera_config.yml')
        # camera_config_path = '/data/sea_disk0/cuihz/code/RoboTwin/task_config/_camera_config.yml'

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
        self.st_seed = (seed+1)*(10000)
        self.task.set_actor_pose(True)
        return self.find_seed(task_num)
    def Init_task_env(self,seed,id,run_dir,epoch):
        self.env_state=0 #0:running 1:success 2:fail
        self.step=0
        self.task.setup_demo(now_ep_num=id, seed = seed, is_test = True, ** self.args)
        self.eval_video_log = True
        self.video_size = str(self.args['head_camera_w']) + 'x' + str(self.args['head_camera_h'])
        self.save_dir = str(epoch) + '-' + run_dir + '/' + 'seed' + str(seed)
        if self.eval_video_log:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_dir = Path('RoboTwin1.0_3d_policy/eval_video') / self.save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)
            log_file = open(f'{self.save_dir}/{time_str}_ffmpeg_log.txt', 'w')
            
            self.ffmpeg = subprocess.Popen([
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', self.video_size,
            '-framerate', '10',
            '-i', '-',
            '-pix_fmt', 'yuv420p',
            '-vcodec', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-g', '15',
            '-threads', '0',
            f'{self.save_dir}/{time_str}.mp4'
        ], stdin=subprocess.PIPE, stdout=log_file, stderr=log_file)
    def save_seed(self, seedlist, st_seed=None):
        if st_seed is None:
            st_seed = self.st_seed
        st_seed_key = str(st_seed)
        
        save_path = f'./RoboTwin1.0_3d_policy/seeds_list'
        file_path = os.path.join(save_path, f'{self.args["task_name"]}.json')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        new_seeds = set(seedlist)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if st_seed_key in data:
                    existing_seeds = set(data[st_seed_key])
                    all_seeds = sorted(list(existing_seeds.union(new_seeds)))
                    data[st_seed_key] = all_seeds
                else:
                    data[st_seed_key] = sorted(list(new_seeds))
            except (json.JSONDecodeError, FileNotFoundError):
                data = {st_seed_key: sorted(list(new_seeds))}
        else:
            data = {st_seed_key: sorted(list(new_seeds))}
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    def find_seed(self, task_num):
        save_path = f'./RoboTwin1.0_3d_policy/seeds_list'
        file_path = os.path.join(save_path, f'{self.args["task_name"]}.json')
        st_seed_key = str(self.st_seed)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        existing_seeds = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                existing_seeds = data.get(st_seed_key, [])
            except (json.JSONDecodeError, FileNotFoundError):
                existing_seeds = []
        
        valid_seeds = existing_seeds
        
        if len(valid_seeds) >= task_num:
            selected_seeds = valid_seeds[:task_num]
            id_list = list(range(task_num))
            print(f"Found {len(selected_seeds)} valid seeds in group {st_seed_key}")
            return selected_seeds, id_list
        
        print(f"Insufficient seeds in group {st_seed_key} ({len(valid_seeds)}/{task_num}), starting to find new seeds...")
        
        needed_seeds = task_num - len(valid_seeds)
        start_seed = self.st_seed
        if valid_seeds:
            start_seed = max(valid_seeds) + 1
        
        new_seeds, new_ids = self.Check_seed(needed_seeds, start_seed)
        
        final_seeds = valid_seeds + new_seeds
        final_seeds = final_seeds[:task_num]
        final_ids = list(range(len(final_seeds)))
        
        self.save_seed(new_seeds)
        
        print(f"Total found {len(final_seeds)} valid seeds in group {st_seed_key}")
        return final_seeds, final_ids
    def Check_seed(self,test_num,start_seed):
        expert_check=True
        print("Task name: ", self.args["task_name"])
        suc_seed_list=[]
        now_id_list = []
        succ_tnt=0
        now_seed=start_seed
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
                except Exception as e:
                    stack_trace = traceback.format_exc()
                    print(' -------------')
                    print('Error: ', stack_trace)
                    print(' -------------')
                    self.task.close()
                    now_seed += 1
                    self.args['render_freq'] = render_freq
                    print('Error occurred!')
                    continue
            if (not expert_check) or ( self.task.plan_success and self.task.check_success() ):
                now_id_list.append(now_id)
                suc_seed_list.append(now_seed)
                succ_tnt +=1
                now_seed += 1
                now_id += 1
                
            else:
                now_seed += 1
                self.args['render_freq'] = render_freq
                continue
        return suc_seed_list, now_id_list
    def Detect_env_state(self):
        if self.step>self.task.get_step_lim():
            self.env_state=2
        if self.task.check_success():
            self.env_state=1
        if self.task.get_actor_pose()==False:
            self.env_state=2
    def Take_action(self,action,model=None):
        # actions=[]
        # actions.append(action)
        # actions=np.array(actions)
        # self.task.apply_action(actions)
        observation = self.get_observation()
        self.ffmpeg.stdin.write(observation['observation']['head_camera']['rgb'].tobytes())
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
        if self.eval_video_log:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            del self.ffmpeg
        if self.task.render_freq:
            self.task.viewer.close()
        print ('Environment Closed!')
        self.task._take_picture()
    def get_observation(self):
        return self.task.Get_observation()