import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import timm
import os
import pdb

from pytorch3d.ops import sample_farthest_points
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x, eval):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x, eval):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class DP3Encoder(nn.Module):
    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 multi_task_config=None,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.language_key = 'language'
        self.n_output_channels = out_channel

        # Multi-task language support
        self.multi_task_enabled = multi_task_config is not None

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        
        # 支持Uni3D编码器
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        elif pointnet_type == "uni3d":
            cprint(f"[DP3Encoder] 使用Uni3D编码器", "green")
            # 为Uni3D编码器设置默认配置
            uni3d_config = {
                'pc_model': 'eva02_large_patch14_448',
                'pc_feat_dim': 1024,
                'embed_dim': out_channel,  # 匹配输出维度
                'group_size': 32,
                'num_group': 512,
                'patch_dropout': 0.5,
                'drop_path_rate': 0.2,
                'pretrained_pc': None,
                'pc_encoder_dim': 512,
                'use_pretrained_weights': False,
                'pretrained_weights_path': None,
            }
            
            # 使用用户提供的配置覆盖默认值
            if pointcloud_encoder_cfg:
                uni3d_config.update(pointcloud_encoder_cfg)
            
            self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
            
            # 调整输出通道数以匹配Uni3D输出
            self.n_output_channels = uni3d_config['embed_dim']
            
        elif pointnet_type == "uni3d_pretrained":
            cprint(f"[DP3Encoder] 使用预训练Uni3D编码器", "green")
            # 为预训练Uni3D编码器设置配置
            uni3d_config = {
                'pc_model': 'eva02_large_patch14_448',
                'pc_feat_dim': 1024,
                'embed_dim': out_channel,  # 匹配输出维度
                'group_size': 32,
                'num_group': 512,
                'patch_dropout': 0.5,
                'drop_path_rate': 0.2,
                'pretrained_pc': None,
                'pc_encoder_dim': 512,
                'use_pretrained_weights': True,
                'pretrained_weights_path': 'Uni3D_large/model.pt',  # 默认路径
            }
            
            # 使用用户提供的配置覆盖默认值
            if pointcloud_encoder_cfg:
                uni3d_config.update(pointcloud_encoder_cfg)
            
            self.extractor = Uni3DPointcloudEncoder(**uni3d_config)
            
            # 调整输出通道数以匹配Uni3D输出
            self.n_output_channels = uni3d_config['embed_dim']
            
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
        # import pdb
        # pdb.set_trace()

        # Initialize language encoder for multi-task mode
        if self.multi_task_enabled:
            self.language_encoder_type = multi_task_config.get('language_encoder_type', 'sentence_transformers')
            language_mlp_dim = multi_task_config.get('language_mlp_dim', 64)

            if self.language_encoder_type == 'sentence_transformers':
                try:
                    from sentence_transformers import SentenceTransformer
                    self.language_encoder = SentenceTransformer(
                        multi_task_config.get('language_encoder', 'sentence-transformers/all-MiniLM-L6-v2')
                    )
                    # Freeze language encoder parameters
                    for param in self.language_encoder.parameters():
                        param.requires_grad = False

                    # Language feature MLP (384 -> language_mlp_dim)
                    self.language_mlp = nn.Sequential(*create_mlp(384, language_mlp_dim, [128], state_mlp_activation_fn))
                    cprint(f"[DP3Encoder] Using SentenceTransformers encoder: {multi_task_config.get('language_encoder')}", "green")

                except ImportError:
                    raise ImportError("sentence-transformers is required for sentence_transformers mode. Install with: pip install sentence-transformers")

            elif self.language_encoder_type == 'clip':
                try:
                    import open_clip
                    clip_model = multi_task_config.get('clip_model', 'EVA02-E-14-plus')
                    clip_pretrained = multi_task_config.get('clip_pretrained', 'laion2b_s9b_b144k')

                    # Initialize CLIP model and tokenizer
                    self.clip_model, _, _ = open_clip.create_model_and_transforms(clip_model, pretrained=clip_pretrained)
                    self.clip_model.eval()
                    self.clip_tokenizer = open_clip.get_tokenizer(clip_model)

                    # Freeze CLIP parameters
                    for param in self.clip_model.parameters():
                        param.requires_grad = False

                    # # Get CLIP text encoder output dimension (typically 1024 for EVA02-E-14-plus)
                    # clip_dim = self.clip_model.text.output_dim if hasattr(self.clip_model.text, 'output_dim') else 1024

                    # Language feature MLP (clip_dim -> language_mlp_dim)
                    self.language_mlp = nn.Linear(1024, language_mlp_dim)
                    cprint(f"[DP3Encoder] Using CLIP encoder: {clip_model} with pretrained: {clip_pretrained}", "green")

                except ImportError:
                    raise ImportError("open_clip is required for CLIP mode. Install with: pip install open_clip_torch")
            else:
                raise ValueError(f"Unsupported language_encoder_type: {self.language_encoder_type}. Supported types: 'sentence_transformers', 'clip'")

            self.n_output_channels += language_mlp_dim
            cprint(f"[DP3Encoder] Multi-task mode enabled with {self.language_encoder_type} encoder", "green")
            cprint(f"[DP3Encoder] Language MLP output dim: {language_mlp_dim}", "green")

        cprint(f"[DP3Encoder] Final output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict, eval=False) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        # 处理不同类型的编码器
        if self.pointnet_type in ["uni3d", "uni3d_pretrained"]:
            # Uni3D编码器需要6通道输入 (xyz + rgb)
            if points.shape[-1] == 3:
                # 如果只有xyz，添加零颜色
                colors = torch.zeros_like(points)
                points = torch.cat([points, colors], dim=-1)
            elif points.shape[-1] > 6:
                # 如果有超过6个通道，只取前6个
                points = points[..., :6]

        # print('============= print Encoder -> points =============\n')
        # print(points)
        # print('============= print Encoder -> points =============\n')
            
        # pdb.set_trace()

        # points: B * N * (3 or 6)
        pn_feat = self.extractor(points, eval)    # B * out_channel
        # print(f"pn_feat has NaNs after self.extractor: {torch.isnan(pn_feat).any()}")

        # print('============= print Encoder -> pn_feat =============\n')
        # print(pn_feat)
        # print('============= print Encoder -> pn_feat =============\n')

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64

        # print('============= print Encoder -> state_feat =============\n')
        # print(state_feat)
        # print('============= print Encoder -> state_feat =============\n')

        # Prepare feature list for concatenation
        features = [pn_feat, state_feat]

        # Add language features for multi-task mode
        if self.multi_task_enabled and self.language_key in observations:
            language_instructions = observations[self.language_key]

            if self.language_encoder_type == 'sentence_transformers':
                # Handle batch of instructions for SentenceTransformers
                if isinstance(language_instructions, (list, tuple)):
                    # Encode batch of instructions
                    with torch.no_grad():
                        language_embeddings = self.language_encoder.encode(
                            language_instructions,
                            convert_to_tensor=True,
                            device=pn_feat.device,
                            show_progress_bar=False
                        )
                else:
                    # Single instruction
                    with torch.no_grad():
                        language_embeddings = self.language_encoder.encode(
                            [language_instructions],
                            convert_to_tensor=True,
                            device=pn_feat.device,
                            show_progress_bar=False
                        )
                        language_embeddings = language_embeddings[0:1]  # Keep batch dimension

            elif self.language_encoder_type == 'clip':
                # Handle batch of instructions for CLIP
                if isinstance(language_instructions, (list, tuple)):
                    instruction_list = language_instructions
                else:
                    instruction_list = [language_instructions]

                with torch.no_grad():
                    # Tokenize instructions
                    tokenized = self.clip_tokenizer(instruction_list).to(pn_feat.device)
                    # Encode with CLIP text encoder
                    language_embeddings = self.clip_model.encode_text(tokenized)

                    # If single instruction, ensure batch dimension
                    if not isinstance(language_instructions, (list, tuple)):
                        language_embeddings = language_embeddings[0:1]

            # Apply language MLP
            language_feat = self.language_mlp(language_embeddings)  # B * language_mlp_dim
            features.append(language_feat)
        
        # print('============= print Encoder -> features =============\n')
        # print(features)
        # print('============= print Encoder -> features =============\n')

        final_feat = torch.cat(features, dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels


# =============================================================================
# Uni3D 编码器相关组件 - 从FP3项目迁移
# =============================================================================

# 需要的辅助函数和依赖项

# except ImportError:
#     # 如果openpoints不可用，提供一个简单的替代实现
#     def furthest_point_sample(xyz, npoint):
#         """
#         简化的FPS实现 - 如果openpoints不可用则使用随机采样
#         """
#         print("Warning: openpoints not available, using random sampling instead of FPS")
#         B, N, _ = xyz.shape
#         idx = torch.randperm(N)[:npoint].unsqueeze(0).expand(B, -1)
#         return idx

# from openpoints.models.layers import furthest_point_sample
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    xyz_coordinates = data[:, :, :3]

    # sample_farthest_points 返回两个值：sampled_xyz (采样的坐标) 和 idx (索引)
    # 我们只需要索引 idx 来从原始数据中 gather 点
    # PyTorch3D 的函数签名是 sample_farthest_points(xyz: Tensor, K: int, ...)
    _, fps_idx = sample_farthest_points(xyz_coordinates, K=number) # K=number 是要采样的点数

    # fps_idx 是 (B, number) 的 LongTensor 索引
    # 使用 gather 从原始数据 data (B, N, C) 中选取点
    # 需要将 fps_idx 扩展到与 data 相同的维度，以便 gather 操作
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    
    return fps_data
    # fps_idx = furthest_point_sample(data[:, :, :3].contiguous(), number)
    # fps_data = torch.gather(
    #     data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    # return fps_data
# from pointnet2_ops import pointnet2_utils
# def fps(data, number):
#     '''
#         data B N 3
#         number int
#     '''
#     fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
#     fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
#     return fps_data

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    B, N, _ = batch_pc.shape
    result = torch.clone(batch_pc)
    for b in range(B):
        dropout_ratio = torch.rand(1).item() * max_dropout_ratio  # 0 ~ 0.875
        drop_idx = torch.where(torch.rand(N) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            result[b, drop_idx, :] = batch_pc[b, 0, :].unsqueeze(0)  # set to the first point
    return result

class PatchDropout(nn.Module):
    """
    Patch dropout for Uni3D
    https://arxiv.org/abs/2212.00794
    """
    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x

class Group(nn.Module):
    """点云分组模块"""
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = torch.cat((neighborhood, neighborhood_color), dim=-1)
        return neighborhood, center, features

class Encoder(nn.Module):
    """Uni3D点云编码器"""
    def __init__(self, encoder_channel, normalization_type="batch_norm"):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.normalization_type = normalization_type

        # Helper function to create normalization layer
        def create_norm_layer(num_features):
            """
            Create normalization layer based on configuration
            Args:
                num_features: Number of features/channels
            """

            cprint(f"[Uni3DPointcloudEncoder] 使用{self.normalization_type}", "yellow")

            if self.normalization_type == "batch_norm":
                return nn.BatchNorm1d(num_features)
            elif self.normalization_type == "layer_norm":
                # For 1D convolutions with LayerNorm, we need a custom wrapper
                # Input shape: (batch_size, channels, sequence_length)
                # We want to normalize over the channel dimension
                class LayerNorm1d(nn.Module):
                    def __init__(self, num_features):
                        super().__init__()
                        self.layer_norm = nn.LayerNorm(num_features)

                    def forward(self, x):
                        # x shape: (batch_size, channels, sequence_length)
                        # Transpose to (batch_size, sequence_length, channels)
                        x = x.transpose(1, 2)
                        # Apply LayerNorm
                        x = self.layer_norm(x)
                        # Transpose back to (batch_size, channels, sequence_length)
                        x = x.transpose(1, 2)
                        return x

                return LayerNorm1d(num_features)
            elif self.normalization_type == "none":
                return nn.Identity()
            else:
                raise ValueError(f"Unsupported normalization type: {self.normalization_type}")

        # First convolution block
        first_conv_layers = [
            nn.Conv1d(6, 128, 1),
        ]
        first_conv_layers.append(create_norm_layer(128))
        first_conv_layers.extend([
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        ])
        self.first_conv = nn.Sequential(*first_conv_layers)

        # Second convolution block
        second_conv_layers = [
            nn.Conv1d(512, 512, 1),
        ]
        second_conv_layers.append(create_norm_layer(512))
        second_conv_layers.extend([
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        ])
        self.second_conv = nn.Sequential(*second_conv_layers)
    
    def forward(self, point_groups):
        '''
            point_groups : B G N 6
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG encoder_channel n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG encoder_channel
        return feature_global.reshape(bs, g, self.encoder_channel)

class Uni3DPointcloudEncoder(nn.Module):
    """
    Uni3D点云编码器 - 从FP3项目迁移到DP3
    支持预训练权重加载和从头训练两种模式
    """
    def __init__(self,
                 pc_model='eva02_large_patch14_448',
                 pc_feat_dim=1024,
                 embed_dim=1024,
                 group_size=32,
                 num_group=512,
                 patch_dropout=0.5,
                 drop_path_rate=0.2,
                 pretrained_pc=None,
                 pc_encoder_dim=512,
                 use_pretrained_weights=False,
                 pretrained_weights_path=None,
                 normalization_type="batch_norm",
                 **kwargs):
        super().__init__()
        
        # 创建point transformer backbone
        point_transformer = timm.create_model(pc_model, checkpoint_path=pretrained_pc, drop_path_rate=drop_path_rate)
        
        self.trans_dim = pc_feat_dim
        self.embed_dim = embed_dim
        self.group_size = group_size
        self.num_group = num_group
        self.use_pretrained_weights = use_pretrained_weights
        
        # 点云分组器
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        
        # 定义编码器
        self.encoder_dim = pc_encoder_dim
        self.encoder = Encoder(encoder_channel=self.encoder_dim, normalization_type=normalization_type)
    
        # 桥接层
        self.encoder2trans = nn.Linear(self.encoder_dim, self.trans_dim)
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)
        
        # Transformer相关参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        
        # Patch dropout
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        
        # Vision transformer组件
        self.visual_pos_drop = point_transformer.pos_drop
        self.visual_blocks = point_transformer.blocks
        self.visual_norm = point_transformer.norm
        self.visual_fc_norm = point_transformer.fc_norm

        # import pdb
        # pdb.set_trace()
        
        # 加载预训练权重（如果指定）
        if use_pretrained_weights: # and pretrained_weights_path is not None:
            self._load_pretrained_weights_selective(pretrained_weights_path, normalization_type)
        else:
            cprint(f"[Uni3DPointcloudEncoder] 使用随机初始化权重（从头训练模式）", "yellow")

    def _load_pretrained_weights_selective(self, pretrained_weights_path, normalization_type):
        """
        选择性加载预训练权重，根据normalization_type决定加载策略

        Args:
            pretrained_weights_path: 预训练权重文件路径
            normalization_type: 归一化类型 ("batch_norm", "layer_norm", "none")
        """
        try:
            # 构建权重文件完整路径
            current_file_path_abs = os.path.abspath(__file__)
            current_directory_os = os.path.dirname(current_file_path_abs)
            cur_dir = os.path.join(current_directory_os, '../../..')
            load_weight_path = os.path.join(cur_dir, pretrained_weights_path)

            if not os.path.exists(load_weight_path):
                cprint(f"[Uni3DPointcloudEncoder] 预训练权重文件不存在: {load_weight_path}", "red")
                return

            # 加载预训练权重
            checkpoint = torch.load(load_weight_path)
            if 'module' in checkpoint:
                state_dict = checkpoint['module']
            else:
                state_dict = checkpoint

            # 处理键名映射
            processed_state_dict = {}
            for key in list(state_dict.keys()):
                new_key = key.replace('point_encoder.', '').replace('visual.', 'visual_')
                processed_state_dict[new_key] = state_dict[key]

            # 获取当前模型的state_dict
            current_state_dict = self.state_dict()

            # 根据normalization_type选择性加载权重
            if normalization_type == "batch_norm":
                # batch_norm模式：完全兼容，加载所有匹配的权重
                filtered_state_dict = {}
                skipped_keys = []
                # loaded_keys = []
                for key, value in processed_state_dict.items():
                    if key in current_state_dict and value.shape == current_state_dict[key].shape:
                        filtered_state_dict[key] = value
                        # loaded_keys.append(key)
                    else:
                        skipped_keys.append(key)


                missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
                cprint(f"[Uni3DPointcloudEncoder] batch_norm模式：完全加载预训练权重", "green")
                cprint(f"  加载的参数: {len(filtered_state_dict)}", "green")
                cprint(f"  跳过的参数: {skipped_keys}, 共 {len(skipped_keys)} 个", "yellow")
                if missing_keys:
                    cprint(f"  缺失的参数: {len(missing_keys)}", "yellow")
                if unexpected_keys:
                    cprint(f"  意外的参数: {len(unexpected_keys)}", "yellow")

            else:
                # layer_norm或none模式：选择性加载，只加载卷积层权重
                filtered_state_dict = {}
                skipped_keys = []
                loaded_keys = []

                for key, value in processed_state_dict.items():
                    if key in current_state_dict:
                        # 检查是否是BatchNorm相关的参数
                        if any(bn_key in key for bn_key in ['_conv.1.weight', '_conv.1.bias', '_conv.1.running_mean', '_conv.1.running_var', '_conv.1.num_batches_tracked']):
                            # 跳过encoder中的BatchNorm参数（但保留visual transformer的norm参数）
                            if 'encoder.' in key:
                                skipped_keys.append(key)
                                continue

                        # 检查形状是否匹配
                        if value.shape == current_state_dict[key].shape:
                            filtered_state_dict[key] = value
                            loaded_keys.append(key)
                        else:
                            skipped_keys.append(key)
                    else:
                        skipped_keys.append(key)

                # 加载过滤后的权重
                missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)

                cprint(f"[Uni3DPointcloudEncoder] {normalization_type}模式：选择性加载预训练权重", "green")
                cprint(f"  成功加载的参数: {len(loaded_keys)}", "green")
                cprint(f"  跳过的参数: {skipped_keys}, 共 {len(skipped_keys)} 个", "yellow")


                if missing_keys:
                    cprint(f"  缺失的参数: {missing_keys}, 共 {len(missing_keys)} 个", "yellow")
                if unexpected_keys:
                    cprint(f"  意外的参数: {len(unexpected_keys)}", "yellow")

                # 提醒用户新的归一化层将从头训练
                if normalization_type == "layer_norm":
                    cprint(f"  LayerNorm层将从随机初始化开始训练", "cyan")
                elif normalization_type == "none":
                    cprint(f"  无归一化层(identity)，卷积层权重已加载", "cyan")

            cprint(f"[Uni3DPointcloudEncoder] 预训练权重加载完成: {load_weight_path}", "green")

        except Exception as e:
            cprint(f"[Uni3DPointcloudEncoder] 加载预训练权重时出错: {str(e)}", "red")
            cprint(f"[Uni3DPointcloudEncoder] 将使用随机初始化权重", "yellow")

    # def _load_pretrained_weights(self, weights_path):
    #     """加载Uni3D预训练权重"""
    #     try:
    #         if os.path.exists(weights_path):
    #             state_dict = torch.load(weights_path, map_location='cpu')
                
    #             # 处理可能的module前缀
    #             if 'module' in state_dict:
    #                 state_dict = state_dict['module']
                
    #             # 处理键名映射
    #             new_state_dict = {}
    #             for key, value in state_dict.items():
    #                 new_key = key.replace('point_encoder.', '').replace('visual.', 'visual_')
    #                 new_state_dict[new_key] = value
                
    #             # 过滤不匹配的键
    #             filtered_state_dict = {}
    #             for key, value in new_state_dict.items():
    #                 if key in self.state_dict():
    #                     filtered_state_dict[key] = value
    #                 else:
    #                     cprint(f"跳过不匹配的键: {key}", "red")
                
    #             # 加载权重
    #             missing_keys, unexpected_keys = self.load_state_dict(filtered_state_dict, strict=False)
    #             if missing_keys:
    #                 cprint(f"缺失的键: {missing_keys}", "yellow")
    #             if unexpected_keys:
    #                 cprint(f"意外的键: {unexpected_keys}", "yellow")
                    
    #         else:
    #             cprint(f"预训练权重文件不存在: {weights_path}", "red")
                
    #     except Exception as e:
    #         cprint(f"加载预训练权重时出错: {e}", "red")

    def forward(self, pcd, eval):
        # 应用点云dropout（数据增强）
        # if self.training:
        if not eval:
            # cprint(f"[Uni3DPointcloudEncoder] 随机drop点云, for data augmentation", "yellow")
            pcd = random_point_dropout(pcd, max_dropout_ratio=0.8)
        # else:
            # cprint(f"[Uni3DPointcloudEncoder] 保持原点云输入, 不随机dop点云", "yellow")
        
        pts = pcd[..., :3].contiguous()
        colors = pcd[..., 3:].contiguous()
        
        # 点云分组
        _, center, features = self.group_divider(pts, colors)

        # 编码输入点云patches
        group_input_tokens = self.encoder(features)  # B G N
        group_input_tokens = self.encoder2trans(group_input_tokens)
        
        # 准备cls token
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        
        # 添加位置嵌入
        pos = self.pos_embed(center)
        
        # 最终输入
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        
        # transformer
        x = x + pos
        
        # patch dropout
        if not eval:
            # cprint(f"[Uni3DPointcloudEncoder] 启用dropout", "yellow")
            x = self.patch_dropout(x)
            x = self.visual_pos_drop(x)

        # 通过visual transformer blocks
        for i, blk in enumerate(self.visual_blocks):
            x = blk(x)
        
        x = self.visual_norm(x[:, 0, :])
        x = self.visual_fc_norm(x)
        x = self.trans2embed(x)
        
        return x