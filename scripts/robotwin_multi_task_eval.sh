#!/bin/bash

# RoboTwin多任务DP3策略评估脚本
# Examples:
# bash scripts/robotwin_multi_task_eval.sh dp3_multi_task multi_task_robotwin 9999 0 0


# 默认参数
task_name="multi_task_robotwin"
EPOCH=75
NUM_EPISODES=20
EVAL_SEED=1
HEAD_CAMERA_TYPE="D435"
MAX_STEPS=1000
TASKS=("block_hammer_beat" "block_handover" "dual_bottles_pick_easy" "bottle_adjust" "dual_bottles_pick_hard")
# "block_hammer_beat" "block_handover" "dual_bottles_pick_easy" "bottle_adjust" "dual_bottles_pick_hard"


alg_name=${1}
task_name=${2}
addition_info=${3}
training_seed=${4}

gpu_id=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
RUN_DIR="${exp_name}_seed${training_seed}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export TOKENIZERS_PARALLELISM=false

# 显示配置信息
echo "=================================="
echo "RoboTwin多任务DP3策略评估"
echo "=================================="
echo "任务名称: $task_name"
echo "Checkpoint epoch: $EPOCH"
echo "评估回合数: $NUM_EPISODES"
echo "EVAL_SEED: $EVAL_SEED"
echo "头部相机类型: $HEAD_CAMERA_TYPE"
echo "最大步数: $MAX_STEPS"
echo "视频保存路径: videos/${TASK_NAME}-epoch_${EPOCH}/"
echo "=================================="
echo
echo

python RoboTwin1.0_3d_policy/test_multi_task.py \
    --task_name "$task_name" \
    --epoch "$EPOCH" \
    --num_episodes "$NUM_EPISODES" \
    --seed "$EVAL_SEED" \
    --head_camera_type "$HEAD_CAMERA_TYPE" \
    --max_steps "$MAX_STEPS" \
    --run_dir "$RUN_DIR" \
    --alg_name "$alg_name" \
    --tasks "${TASKS[@]}"
