from typing import Dict
import torch
import numpy as np
import os
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import pdb
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

class RobotwinDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            use_data_augmentation=False,
            pc_noise_std=0.002,
            agent_pos_noise_std=0.0002,
            task_name=None,
            use_endpose=False,  # 新增：是否使用 endpose 数据
            ):
        super().__init__()
        
        # 根据 use_endpose 修改 zarr_path
        if use_endpose:
            # 将 data/robotwin2/{task}-demo_clean.zarr 
            # 转换为 data/robotwin2_endpose/{task}-demo_clean-100_endpose.zarr
            zarr_path = zarr_path.replace('data/robotwin2/', 'data/robotwin2_endpose/')
            zarr_path = zarr_path.replace('.zarr', '_endpose.zarr')
            cprint("--------------------------", "cyan")
            cprint(f"使用 Endpose 数据: {zarr_path}", "cyan")
            cprint("--------------------------", "cyan")
        else:
            cprint("--------------------------", "yellow")
            cprint(f"使用 Joint Space 数据: {zarr_path}", "yellow")
            cprint("--------------------------", "yellow")
        
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
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud']) # 'img'
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        # import pdb
        # pdb.set_trace()

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

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

