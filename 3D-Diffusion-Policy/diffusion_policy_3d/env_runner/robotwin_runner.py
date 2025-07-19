import wandb
import numpy as np
import torch
import tqdm
import pdb
from diffusion_policy_3d.env import RobotwinEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint


class RobotwinRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 seed,
                 eval_episodes=20,
                 max_steps=400,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name="block_hammer_beat",
                 use_point_crop=True,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    RobotwinPointcloudWrapper(
                        env=RobotwinEnv(task_name=task_name, use_point_cloud=True, seed=seed, task_num=eval_episodes),
                        env_name='robotwin_'+task_name, 
                        use_point_crop=use_point_crop
                    )
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self.env = env_fn()

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        all_rewards = []

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), 
                                     desc=f"Eval in RoboTwin {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):
                
            # start rollout
            obs = env.reset(episode_idx=episode_idx)
            policy.reset()

            done = False
            num_goal_achieved = 0
            episode_reward = 0
            actual_step_count = 0
            
            while not done:
                
                pdb.set_trace()
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # run policy
                with torch.no_grad():
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                    # print('============= obs_dict_input =============\n')
                    # print(obs_dict_input)
                    # print('============= obs_dict_input =============\n')
                    action_dict = policy.predict_action(obs_dict_input)
                    # print('============= action_dict =============\n')
                    # print(action_dict)
                    # print('============= action_dict =============\n')
                    

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                # step env
                obs, reward, done, info = env.step(action)
                
                # accumulate metrics
                episode_reward += reward
                if 'goal_achieved' in info:
                    num_goal_achieved += np.sum(info['goal_achieved'])
                elif 'success' in info:
                    num_goal_achieved += np.sum(info['success'])
                    
                
                done = np.all(done)
                actual_step_count += 1

            # Record episode results
            final_success = info.get('goal_achieved', info.get('success', False))
            cprint(f"episode{episode_idx+1}: {info['goal_achieved']}", 'green')
            all_success_rates.append(final_success)
            all_goal_achieved.append(num_goal_achieved)
            all_rewards.append(episode_reward)

        # log
        log_data = dict()
        
        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)
        log_data['mean_episode_reward'] = np.mean(all_rewards)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')
        cprint(f"mean_episode_reward: {np.mean(all_rewards)}", 'green')
        cprint(f"success_rate: {np.mean(all_success_rates):.3f}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        # Record video if available
        try:
            videos = env.env.get_video()
            if len(videos.shape) == 5:
                videos = videos[:, 0]  # select first frame
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb
        except:
            # Video recording might not be available
            pass

        # clear out video buffer
        _ = env.reset(episode_idx=0)
        # clear memory
        videos = None
        del env

        return log_data


class RobotwinPointcloudWrapper:
    """
    Point cloud wrapper for RoboTwin environments.
    Similar to MujocoPointcloudWrapperAdroit but adapted for RoboTwin.
    """
    def __init__(self, env, env_name='robotwin', use_point_crop=True, num_points=1024):
        self.env = env
        self.env_name = env_name
        self.use_point_crop = use_point_crop
        self.num_points = num_points
        
    def reset(self, **kwargs):
        obs_dict = self.env.reset(**kwargs)
        # Point cloud is already included in the observation
        return obs_dict
    
    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        # Point cloud is already included in the observation
        return obs_dict, reward, done, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)
