# RoboTwin1.0
3d policy
# Data Structure
- **observation** (`dict`)
  - **head_camera** (`dict`)
    - **intrinsic_cv**: `ndarray` — Shape `(3, 3)`
    - **extrinsic_cv**: `ndarray` — Shape `(3, 4)`
    - **cam2world_gl**: `ndarray` — Shape `(4, 4)`
    - **rgb**: `ndarray` — Shape `(240, 320, 3)`
    - **depth**: `ndarray` — Shape `(240, 320)`
  - **left_camera** (`dict`)
    - **intrinsic_cv**: `ndarray` — Shape `(3, 3)`
    - **extrinsic_cv**: `ndarray` — Shape `(3, 4)`
    - **cam2world_gl**: `ndarray` — Shape `(4, 4)`
    - **rgb**: `ndarray` — Shape `(240, 320, 3)`
    - **depth**: `ndarray` — Shape `(240, 320)`
  - **right_camera** (`dict`)
    - **intrinsic_cv**: `ndarray` — Shape `(3, 3)`
    - **extrinsic_cv**: `ndarray` — Shape `(3, 4)`
    - **cam2world_gl**: `ndarray` — Shape `(4, 4)`
    - **rgb**: `ndarray` — Shape `(240, 320, 3)`
    - **depth**: `ndarray` — Shape `(240, 320)`
  - **front_camera** (`dict`)
    - **intrinsic_cv**: `ndarray` — Shape `(3, 3)`
    - **extrinsic_cv**: `ndarray` — Shape `(3, 4)`
    - **cam2world_gl**: `ndarray` — Shape `(4, 4)`
    - **rgb**: `ndarray` — Shape `(240, 320, 3)`
    - **depth**: `ndarray` — Shape `(240, 320)`
- **pointcloud**: `ndarray` — Shape `(1024, 6)`
- **joint_action**: `ndarray` — Shape `(14,)`
- **endpose**: `ndarray` — Shape `(14,)`
---
# Data Path
## You should copy these to your local folder.
`/data/sea_disk0/cuihz/code/RoboTwin/data`
`/data/sea_disk0/cuihz/code/RoboTwin/aloha_maniskill_sim`
`/data/sea_disk0/cuihz/code/RoboTwin/models`
---
# Code Example
`test.py` and `script/eval_policy_dp3.py`
