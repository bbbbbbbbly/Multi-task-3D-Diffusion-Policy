#!/bin/bash

# RoboTwin单任务DP3策略评估脚本
# Examples:
# bash scripts/robotwin_eval.sh dp3 block_hammer_beat 9999 0 0


# 默认参数
task_name="block_hammer_beat"
EPOCH=0
NUM_EPISODES=20
EVAL_SEED=1
HEAD_CAMERA_TYPE="D435"
MAX_STEPS=1000

alg_name=${1}
task_name=${2}
addition_info=${3}
training_seed=${4}

gpu_id=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
RUN_DIR="robotwin_${exp_name}_seed${training_seed}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 显示配置信息
echo "=================================="
echo "RoboTwin单任务DP3策略评估"
echo "=================================="
echo "任务名称: $task_name"
echo "Checkpoint epoch: $EPOCH"
echo "评估回合数: $NUM_EPISODES"
echo "EVAL_SEED: $EVAL_SEED"
echo "头部相机类型: $HEAD_CAMERA_TYPE"
echo "最大步数: $MAX_STEPS"
# echo "项目根目录: $PROJECT_ROOT"
echo "视频保存路径: videos/${TASK_NAME}-epoch_${EPOCH}/"
echo "=================================="
echo

# 运行评估
echo "开始运行单任务评估..."
# echo "命令: python test.py --task_name $TASK_NAME --epoch $EPOCH --num_episodes $NUM_EPISODES --seed $SEED --head_camera_type $HEAD_CAMERA_TYPE --max_steps $MAX_STEPS"
echo

python RoboTwin1.0_3d_policy/test.py \
    --task_name "$task_name" \
    --epoch "$EPOCH" \
    --num_episodes "$NUM_EPISODES" \
    --seed "$EVAL_SEED" \
    --head_camera_type "$HEAD_CAMERA_TYPE" \
    --max_steps "$MAX_STEPS" \
    --run_dir "$RUN_DIR" \
    --alg_name "$alg_name"
