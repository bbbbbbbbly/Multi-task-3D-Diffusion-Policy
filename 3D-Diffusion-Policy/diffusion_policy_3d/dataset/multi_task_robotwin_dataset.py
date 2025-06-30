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


class MultiTaskRobotwinDataset(BaseDataset):
    """
    Multi-task dataset for Robotwin tasks with language instruction support.
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
                 task_name=None):
        
        super().__init__()
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        
        # Multi-task configuration
        self.multi_task_config = multi_task_config or {}
        self.data_root = data_root
        
        # Robotwin task names (17 tasks)
        self.task_names = [
            "block_hammer_beat", "block_handover", "blocks_stack_easy", "blocks_stack_hard",
            "bottle_adjust", "container_place", "diverse_bottles_pick", "dual_bottles_pick_easy", "dual_bottles_pick_hard",
            "dual_shoes_place", "empty_cup_place", "mug_hanging_easy",
            "mug_hanging_hard", "pick_apple_messy", "put_apple_cabinet", "shoe_place",
            "tool_adjust"
        ]
        
        # Initialize
        self._init_multi_task(seed, val_ratio, max_train_episodes)
    
    def _init_multi_task(self, seed, val_ratio, max_train_episodes):
        """Initialize multi-task mode with all Robotwin tasks."""
        self.tasks_data = {}
        self.task_samplers = {}
        self.task_language_instructions = {}
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
                
                # Load language instructions for this task
                self._load_language_instructions(task_name)
                
                print(f"Loaded task {task_name}: {len(sampler)} sequences")
        
        # Create unified sample indices for uniform sampling across tasks
        self._create_sample_indices()
    
    def _load_language_instructions(self, task_name):
        """Load language instructions for a specific task."""
        # Load from JSON file if available
        instruction_file = os.path.join(self.data_root, f"instructions/{task_name}.json")
        with open(instruction_file, 'r') as f:
            data = json.load(f)
            instructions = data['instructions'] # instructions are a list
        
        self.task_language_instructions[task_name] = instructions
    
    def _create_sample_indices(self):
        """Create unified sample indices for uniform sampling across tasks."""
        self.sample_indices = []
        
        for task_name, sampler in self.task_samplers.items():
            for i in range(len(sampler)):
                self.sample_indices.append((task_name, i))
        
        # Shuffle for random sampling
        random.Random(self.seed).shuffle(self.sample_indices)

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
        
        # # Conditionally add language instruction if it exists in the sample
        # if 'language' in sample:
        #     data['obs']['language'] = sample['language'] # Keep as string

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        # Multi-task sampling
        task_name, task_idx = self.sample_indices[idx]
        sampler = self.task_samplers[task_name]
        sample = sampler.sample_sequence(task_idx)
        
        # Add language instruction
        instructions = self.task_language_instructions[task_name]
        language_instruction = random.choice(instructions)
        sample['language'] = language_instruction

        data = self._sample_to_data(sample)
        
        # Convert to torch tensors
        torch_data = dict_apply(data, torch.from_numpy)

        # Conditionally add language instruction if it exists in the sample
        if 'language' in sample:
            torch_data['obs']['language'] = sample['language'] # Keep as string
        
        return torch_data

