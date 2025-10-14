# Multi-task 3D Diffusion Policy

> We developed a multi-task DP3 with RoboTwin benchmark.

## Installing conda environment

1. create python/pytorch env

```bash
conda deactivate
conda create -n dp3_robotwin python=3.10
conda activate dp3_robotwin
```

---

2. install necessary packages

```bash
cd RoboTwin2.0_3D_policy
pip install -r script/requirements.txt
cd ..
```

---

3. install pytorch3d

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

---

4. install CuRobo

```bash
cd 3D-Diffusion-Policy/diffusion_policy_3d/env/robotwin2/envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../../../../..
```

---

5. install dp3

```bash
# you should be under `YOUR_PATH_TO_CODEBASE/3D-Diffusion-Policy` right now
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e . && cd ..
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
pip install timm sentence-transformers==3.2.1 huggingface_hub==0.23.2
pip install open_clip_torch
```

---

6. adjust code in `mplib`

> You can use `pip show mplib` to find where the `mplib` installed.

```
# mplib.planner (mplib/planner.py) line 807
# remove `or collide`

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}
```

---

7. Download assets

```bash
cd RoboTwin2.0_3D_policy
bash script/_download_assets.sh
```

---

8. Move assets

* copy `RoboTwin2.0_3D_policy/assets/` to `YOUR_PATH_TO_CODEBASE/` and `3D-Diffusion-Policy/diffusion_policy_3d/env/robotwin2`.

---

6. download Uni3D weight
    
* Download [Uni3D Model](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-l/model.pt) into `3D-Diffusion-Policy/Uni3D_large/`.
  * if you want to use "Uni3D Tiny" or other Uni3D models, follow the same way, but remember to modify configs (guidance is in the below).
  
---

## Collect data

* Generate Robotwin 2.0 training data for DP3
     * Please follow the Robotwin 2.0 guidance to generate Robotwin 2.0 data for DP3, the data format should be '.zarr'.
     * Then put the data in `3D-Diffusion-Policy/data/robotwin2`.

---

## Guide

### Training Examples

1. DP3 baseline
```bash
bash scripts/train_robotwin2_single.sh dp3_robotwin2 beat_block_hammer 9999 0 2
```

2. Single-task DP3 with Uni3D Encoder
```bash
bash scripts/train_robotwin2_single.sh dp3_uni3d_pretrained_robotwin2 beat_block_hammer 9999 0 2
bash scripts/train_robotwin2_single.sh dp3_uni3d_scratch_robotwin2 beat_block_hammer 9999 0 2
```
* for Uni3D(scratch), you could modify `3D-Diffusion-Policy/diffusion_policy_3d/config/dp3_uni3d_scratch_robotwin2.yaml` 'pointcloud_encoder_cfg: normalization_type' to 'layer_norm' to get a better performance.
* if you want to change the size of Uni3D Encoder, you need to modify 'pointcloud_encoder_cfg: pc_model & pc_feat_dim' to the config below. And remember to modify 'pretrained_weights_path' if you want to use pretrained Uni3D.
  ```shell
  if [ "$1" = "giant" ]; then
    pc_model="eva_giant_patch14_560"
    pc_feat_dim=1408
  elif [ "$1" = "large" ]; then
      pc_model="eva02_large_patch14_448"
      pc_feat_dim=1024
  elif [ "$1" = "base" ]; then
      pc_model="eva02_base_patch14_448"
      pc_feat_dim=768
  elif [ "$1" = "small" ]; then
      pc_model="eva02_small_patch14_224"
      pc_feat_dim=384
  elif [ "$1" = "tiny" ]; then
      pc_model="eva02_tiny_patch14_224"
      pc_feat_dim=192
  else
      echo "Invalid option"
      exit 1
  fi
  ```

3. Multi-GPU training examples
```bash
bash scripts/train_robotwin2_single.sh dp3_uni3d_scratch_robotwin2 beat_block_hammer 9999 0 0,1
```
* You only need to change GPUID from the single one like "0" to the multi one like "0,1" or "0,1,2"

 <!-- 3. Multi-task DP3 (with Uni3D)
 ```bash
 bash scripts/train_policy.sh dp3_multi_task multi_task_robotwin2 9999 0 2
 bash scripts/train_policy.sh dp3_uni3d_pretrained_multi_task multi_task_robotwin2 9999 0 2
 ``` -->

---

### Evaluating examples

 ```bash
bash scripts/eval_policy.sh dp3_robotwin2 beat_block_hammer 9999 0 2
 ```


