# Multi-task 3D Diffusion Policy

> We developed a multi-task DP3 with Robotwin 1.0 & 2.0 benchmark.

* The language encoder is sentence-transformers/all-MiniLM-L6-v2 and the vision(point cloud) encoder could be either DP3 encoder or Uni3D encoder.

## Installing required conda environment

* We need three conda environments: `dp3_training` for (robotwin 1.0 & 2.0) data training, `robotwin1` for robotwin 1.0 evaluation and `robotwin2` for robotwin 2.0 evaluation.

### Installing dp3_training conda environment

---

1. create python/pytorch env

```bash
conda deactivate
conda create -n dp3 python=3.8
conda activate dp3
```

---

2. install torch

```bash
# if using cuda>=12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# else, 
# just install the torch version that matches your cuda version
```

---

3. install dp3

```bash
cd 3D-Diffusion-Policy && pip install -e . && cd ..
```

---

4. install pytorch3d (a simplified version)

```bash
cd third_party/pytorch3d_simplified && pip install -e . && cd ..
```

---

5. install some necessary packages

```bash
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
pip install timm multimethod shortuuid easydict natsort sentence-transformers==3.2.1 huggingface_hub==0.23.2
conda update -c conda-forge libstdcxx-ng
conda install -c conda-forge gxx_linux-64
conda install -c conda-forge cxx-compiler
```

---

6. download Uni3D weight
    
* Download [Uni3D Model](https://huggingface.co/BAAI/Uni3D/blob/main/modelzoo/uni3d-l/model.pt) into `3D-Diffusion-Policy/Uni3D_large/`.
  * if you want to use "Uni3D Tiny" and other Uni3D models, follow the same way, but remember to modify configs.
---

### Installing robotwin1 conda environment

---

1. create python/pytorch env

```bash
conda deactivate
conda create -n robotwin1 python=3.8
conda activate robotwin1
```

---

2. install necessary packages

```bash
pip install torch==2.4.1 torchvision sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic zarr openai huggingface_hub==0.25.0
cd RoboTwin1.0_3d_policy
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..
```

---

3. Download Assert
```bash
python ./script/download_asset.py
unzip aloha_urdf.zip && unzip main_models.zip
```

---

3. install dp3

```bash
cd 3D-Diffusion-Policy && pip install -e . && cd ..
```

---

4. install pytorch3d (a simplified version)

```bash
cd third_party/pytorch3d_simplified && pip install -e . && cd ..
```

---

5. install some necessary packages

```bash
pip install zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba==0.56.4 moviepy imageio av matplotlib termcolor
pip install timm multimethod shortuuid easydict natsort sentence-transformers==3.2.1 huggingface_hub==0.23.2
conda update -c conda-forge libstdcxx-ng
conda install -c conda-forge gxx_linux-64
conda install -c conda-forge cxx-compiler
```

---

6. Modify `mplib` Library Code

> You can use `pip show mplib` to find where the `mplib` installed.

* Remove `convex=True`

```
# mplib.planner (mplib/planner.py) line 71
# remove `convex=True`

self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            convex=True,
            verbose=False,
        )
=> 
self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            # convex=True,
            verbose=False,
        )
```

* Remove `or collide`
```
# mplib.planner (mplib/planner.py) line 848
# remove `or collide`

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}
```

---

7. Move assets

* move/copy `RoboTwin1.0_3d_policy/aloha_maniskill_sim/` to `third_party/` and `YOUR_PATH_TO_CODEBASE/`, and then move/copy `RoboTwin1.0_3d_policy/models/` to `3D-Diffusion-Policy/` instead of leaving them in `RoboTwin1.0_3d_policy/`.

---

### Installing robotwin2 conda environment

* The Installing below takes [RoboTwin 2.0 Document (Usage - Install & Download)](https://robotwin-platform.github.io/doc/usage/robotwin-install.html) as reference.

---

1. create python/pytorch env

```bash
conda deactivate
conda create -n robotwin2 python=3.10
conda activate robotwin2
```

---

2. install necessary packages

```bash
cd RoboTwin2.0_3D_policy
pip install -r script/requirements.txt
```

---

3. install pytorch3d

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

---

4. install CuRobo

```bash
cd envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../..
```

---

5. install dp3

```bash
cd ..
cd 3D-Diffusion-Policy && pip install -e . && cd ..
pip install numpy==1.25 zarr==2.12.0 wandb ipdb gpustat dm_control omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 moviepy imageio av matplotlib termcolor
pip install timm multimethod shortuuid easydict natsort sentence-transformers==3.2.1 huggingface_hub==0.23.2
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

* move/copy `RoboTwin2.0_3D_policy/assets/` to `YOUR_PATH_TO_CODEBASE/` instead of leaving it in `RoboTwin2.0_3D_policy/`.

---

## Collect data

---

1. Generate Robotwin 1.0 training data for DP3
     * Please follow the Robotwin 1.0 guidance to generate Robotwin 1.0 data for DP3, the data format should be '.zarr'.
     * Then put the 17-task data in `3D-Diffusion-Policy/data/multi-task-data` for multi-task training and put the data in `3D-Diffusion-Policy/data/robotwin` for single task training.

---

2. Generate Robotwin 2.0 training data for DP3
     * Please follow the Robotwin 2.0 guidance to generate Robotwin 2.0 data for DP3, the data format should be '.zarr'.
     * Then put the data in `3D-Diffusion-Policy/data/robotwin2`.

---

3. Training examples
```bash
bash scripts/train_policy.sh dp3_uni3d_pretrained robotwin_block_hammer_beat 0000 0 0
bash scripts/train_policy.sh dp3 robotwin_block_hammer_beat 0000 0 0
bash scripts/train_policy.sh dp3_multi_task multi_task_robotwin 0000 0 0
```

---

4. Evaluating examples
```bash
bash scripts/robotwin_eval.sh dp3 block_hammer_beat 0000 0 0
bash scripts/robotwin_eval.sh dp3_uni3d_scratch block_hammer_beat 0000 0 0
bash scripts/robotwin_multi_task_eval.sh dp3_multi_task multi_task_robotwin 0000 0 0
```


