from typing import Dict, List, Optional
import torch
import numpy as np
import os
import copy
import json
import random
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from termcolor import cprint

def add_noise(data, noise_std=0.01, clip_range=0.02):
    """
    为输入添加经过clip的高斯噪声
    
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


class MultiTaskRobotwin2Dataset(BaseDataset):
    """
    Multi-task dataset for RoboTwin2.0 tasks with language instruction support.
    Each episode has its own language instructions stored in separate JSON files.
    """
    
    def __init__(self, 
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.02,
                 max_train_episodes=None,
                 # Multi-task specific parameters
                 data_root=None,
                 multi_task_config=None,
                 use_data_augmentation=False,
                 pc_noise_std=0.002,
                 agent_pos_noise_std=0.0002,
                 task_name=None):
        
        super().__init__()
        self.use_data_augmentation=use_data_augmentation
        self.pc_noise_std=pc_noise_std
        self.agent_pos_noise_std=agent_pos_noise_std
        if use_data_augmentation:
            cprint("--------------------------", "green")
            cprint(f"使用噪声data augmentation: pc_noise_std={pc_noise_std}, agent_pos_noise_std={agent_pos_noise_std}", "green")
            cprint("--------------------------", "green")
        else:
            cprint("--------------------------", "red")
            cprint("数据读入时未使用加噪声的data augmentation", "red")
            cprint("--------------------------", "red")
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        
        # Multi-task configuration
        self.multi_task_config = multi_task_config or {}
        self.data_root = data_root
        
        # RoboTwin2.0 task names (5 tasks)
        self.task_names = [
            "adjust_bottle-demo_clean",
            "beat_block_hammer-demo_clean", 
            "click_alarmclock-demo_clean",
            "click_bell-demo_clean",
            "open_laptop-demo_clean"
        ]
        
        # Initialize
        self._init_multi_task(seed, val_ratio, max_train_episodes)
    
    def _init_multi_task(self, seed, val_ratio, max_train_episodes):
        """Initialize multi-task mode with all RoboTwin2.0 tasks."""
        self.tasks_data = {}
        self.task_samplers = {}
        self.task_episode_instructions = {}  # episode-specific instructions
        self.tasks_train_mask = {}
        
        for task_name in self.task_names:
            # Load zarr data
            zarr_path = os.path.join(self.data_root, f"{task_name}.zarr")
            if os.path.exists(zarr_path):
                replay_buffer = ReplayBuffer.copy_from_path(
                    zarr_path, keys=['state', 'action', 'point_cloud'])
                
                # Create validation mask for this task
                val_mask = get_val_mask(
                    n_episodes=replay_buffer.n_episodes, 
                    val_ratio=val_ratio,
                    seed=seed)
                train_mask = ~val_mask
                train_mask = downsample_mask(
                    mask=train_mask, 
                    max_n=max_train_episodes, 
                    seed=seed)
                
                # Create sampler for this task
                sampler = SequenceSampler(
                    replay_buffer=replay_buffer, 
                    sequence_length=self.horizon,
                    pad_before=self.pad_before, 
                    pad_after=self.pad_after,
                    episode_mask=train_mask)
                
                self.tasks_train_mask[task_name] = train_mask
                self.tasks_data[task_name] = replay_buffer
                self.task_samplers[task_name] = sampler
                
                # Load episode-specific language instructions for this task
                self._load_episode_instructions(task_name)
                
                print(f"Loaded task {task_name}: {len(sampler)} sequences, {replay_buffer.n_episodes} episodes")
        
        # Create unified sample indices for uniform sampling across tasks
        self._create_sample_indices()
    
    def _load_episode_instructions(self, task_name):
        """Load episode-specific language instructions for a specific task."""
        instruction_dir = os.path.join(self.data_root, f"{task_name}-instructions")
        
        episode_instructions = {}
        if os.path.exists(instruction_dir):
            # Get the number of episodes for this task
            replay_buffer = self.tasks_data[task_name]
            n_episodes = replay_buffer.n_episodes
            
            # Load instructions for each episode
            for episode_idx in range(n_episodes):
                instruction_file = os.path.join(instruction_dir, f"episode{episode_idx}.json")
                if os.path.exists(instruction_file):
                    try:
                        with open(instruction_file, 'r') as f:
                            data = json.load(f)
                            # Extract the "seen" instructions
                            instructions = data['seen']
                            episode_instructions[episode_idx] = instructions
                    except Exception as e:
                        print(f"Warning: Failed to load instructions for {task_name} episode {episode_idx}: {e}")
                        # Fallback to empty list
                        episode_instructions[episode_idx] = []
                else:
                    print(f"Warning: Instruction file not found for {task_name} episode {episode_idx}")
                    episode_instructions[episode_idx] = []
        
        self.task_episode_instructions[task_name] = episode_instructions
        print(f"Loaded instructions for {task_name}: {len(episode_instructions)} episodes")
    
    def _create_sample_indices(self):
        """Create unified sample indices for uniform sampling across tasks."""
        self.sample_indices = []
        
        for task_name, sampler in self.task_samplers.items():
            for i in range(len(sampler)):
                self.sample_indices.append((task_name, i))
        
        # Shuffle for random sampling
        random.Random(self.seed).shuffle(self.sample_indices)

    def _get_episode_idx_from_sample_idx(self, task_name, sample_idx):
        """
        Get the episode index for a given sample index using the sampler's indices.
        """
        sampler = self.task_samplers[task_name]
        # Get the buffer indices for this sample
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = sampler.indices[sample_idx]
        
        # Use replay buffer's get_episode_idxs to find which episode this belongs to
        replay_buffer = self.tasks_data[task_name]
        episode_idxs = replay_buffer.get_episode_idxs()
        
        # Get the episode index for the start of this sequence
        episode_idx = episode_idxs[buffer_start_idx]
        return int(episode_idx)

    def get_validation_dataset(self):
        """Get validation dataset."""
        # Create a copy with validation data
        val_dataset = copy.deepcopy(self)
        
        # Update samplers to use validation masks
        for task_name in self.task_names:
            if task_name in self.tasks_data:
                replay_buffer = self.tasks_data[task_name]
                val_mask = ~self.tasks_train_mask[task_name]
                
                val_sampler = SequenceSampler(
                    replay_buffer=replay_buffer, 
                    sequence_length=self.horizon,
                    pad_before=self.pad_before, 
                    pad_after=self.pad_after,
                    episode_mask=val_mask)
                
                val_dataset.task_samplers[task_name] = val_sampler
            
        # Recreate sample indices for validation
        val_dataset._create_sample_indices()
        
        return val_dataset
    
    def get_normalizer(self):
        """Compute normalization statistics across all tasks."""
        # Collect all data for normalization
        all_actions = []
        all_agent_pos = []
        all_point_clouds = []

        for task_name, replay_buffer in self.tasks_data.items():
            all_actions.append(replay_buffer['action'])
            all_agent_pos.append(replay_buffer['state'][...,:])
            all_point_clouds.append(replay_buffer['point_cloud'])
        
        # Concatenate all data
        all_actions = np.concatenate(all_actions, axis=0)
        all_agent_pos = np.concatenate(all_agent_pos, axis=0)
        all_point_clouds = np.concatenate(all_point_clouds, axis=0)
        
        # Create normalizers
        normalizer = LinearNormalizer()
        normalizer.fit({
            'action': all_actions,
            'agent_pos': all_agent_pos,
            'point_cloud': all_point_clouds
        }, last_n_dims=1, mode='limits')
        return normalizer
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sample_indices)
    
    def _sample_to_data(self, sample: dict) -> dict:
        """
        Helper function to format a raw sample from the replay buffer
        into the data structure expected by the model.
        Also handles language instruction.
        """
        agent_pos = sample['state'][:,].astype(np.float32)
        point_cloud = sample['point_cloud'][:,].astype(np.float32)

        data = {
            'obs': {
                'point_cloud': point_cloud,
                'agent_pos': agent_pos
            },
            'action': sample['action'].astype(np.float32)
        }
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        # Multi-task sampling
        task_name, task_idx = self.sample_indices[idx]
        sampler = self.task_samplers[task_name]
        sample = sampler.sample_sequence(task_idx)
        
        # Get the episode index for this sample
        episode_idx = self._get_episode_idx_from_sample_idx(task_name, task_idx)
        
        # Get episode-specific language instructions
        episode_instructions = self.task_episode_instructions[task_name].get(episode_idx, [])
        
        # Choose a random instruction from this episode's instructions
        if episode_instructions:
            language_instruction = random.choice(episode_instructions)
        else:
            # Fallback to a generic instruction if no specific instructions available
            language_instruction = f"Perform the {task_name.replace('-demo_clean', '').replace('_', ' ')} task"
        
        sample['language'] = language_instruction

        data = self._sample_to_data(sample)

        if self.use_data_augmentation:
            if 'point_cloud' in data['obs']:
                # 添加小的高斯噪声，标准差0.01，clip范围0.02
                data['obs']['point_cloud'] = add_noise(
                    data['obs']['point_cloud'], 
                    noise_std=self.pc_noise_std, 
                    clip_range=2*self.pc_noise_std
                )
            if 'agent_pos' in data['obs']:
                # 添加小的高斯噪声，标准差0.01，clip范围0.02
                data['obs']['agent_pos'] = add_noise(
                    data['obs']['agent_pos'], 
                    noise_std=self.agent_pos_noise_std, 
                    clip_range=2*self.agent_pos_noise_std
                )

        
        # Convert to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)

        # Add language instruction
        if 'language' in sample:
            torch_data['obs']['language'] = sample['language'] # Keep as string
        
        return torch_data
