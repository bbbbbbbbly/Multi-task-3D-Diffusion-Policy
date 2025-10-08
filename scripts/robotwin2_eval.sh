#!/bin/bash

# RoboTwin2.0单任务DP3策略评估脚本
# Examples:
# 单个epoch评估:
#   bash scripts/robotwin2_eval.sh dp3_robotwin2 beat_block_hammer 9999 0 0 [epoch] [action_space]
# 批量epoch评估:
#   bash scripts/robotwin2_eval.sh dp3_robotwin2 beat_block_hammer 9999 0 0 [start_epoch] [action_space] [end_epoch] [epoch_interval] [wandb_mode]

# 默认参数
task_name="beat_block_hammer"
EPOCH=500
NUM_EPISODES=20
EVAL_SEED=1
HEAD_CAMERA_TYPE="D435"
MAX_STEPS=1000
TASK_CONFIG="demo_clean"
INSTRUCTION_TYPE="unseen"
ACTION_SPACE="joint"  # 新增：默认为joint space，可选'ee'
START_EPOCH="50"        # 新增：批量评估起始epoch
END_EPOCH="3000"          # 新增：批量评估结束epoch
EPOCH_INTERVAL=50     # 新增：批量评估epoch间隔
WANDB_MODE="online"   # 新增：wandb模式（online/offline/disabled）
WANDB_PROJECT="RoboTwin2.0-Evaluation"  # 新增：wandb项目名称

alg_name=${1}
task_name=${2}
addition_info=${3}
training_seed=${4}

gpu_id=${5}

# 参数解析
if [ ! -z "${6}" ]; then
    EPOCH=${6}
fi

# 第7个参数：action_space (joint 或 ee)
if [ ! -z "${7}" ]; then
    ACTION_SPACE=${7}
fi

# 第8个参数：如果提供，则为批量评估的end_epoch
if [ ! -z "${8}" ]; then
    START_EPOCH=${EPOCH}
    END_EPOCH=${8}
fi

# 第9个参数：epoch_interval
if [ ! -z "${9}" ]; then
    EPOCH_INTERVAL=${9}
fi

# 第10个参数：wandb_mode
if [ ! -z "${10}" ]; then
    WANDB_MODE=${10}
fi

exp_name=${task_name}-${alg_name}-${TASK_CONFIG}-${addition_info}
RUN_DIR="robotwin2_${exp_name}_seed${training_seed}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 显示配置信息
echo "=================================="
echo "RoboTwin2.0单任务DP3策略评估"
echo "=================================="
echo "任务名称: $task_name"
echo "Action Space: $ACTION_SPACE"
if [ ! -z "$START_EPOCH" ] && [ ! -z "$END_EPOCH" ]; then
    echo "批量评估模式"
    echo "Epoch范围: $START_EPOCH 到 $END_EPOCH"
    echo "Epoch间隔: $EPOCH_INTERVAL"
    echo "Wandb模式: $WANDB_MODE"
else
    echo "单个评估模式"
    echo "Checkpoint epoch: $EPOCH"
fi
echo "评估回合数: $NUM_EPISODES"
echo "EVAL_SEED: $EVAL_SEED"
echo "头部相机类型: $HEAD_CAMERA_TYPE"
echo "最大步数: $MAX_STEPS"
echo "任务配置: $TASK_CONFIG"
echo "指令类型: $INSTRUCTION_TYPE"
# echo "视频保存路径: videos/${task_name}-epoch_${EPOCH}/"
echo "=================================="
echo

# 运行评估
echo "开始运行评估..."
echo

# 构建基础命令
CMD="python RoboTwin2.0_3D_policy/test.py \
    --task_name $task_name \
    --num_episodes $NUM_EPISODES \
    --seed $EVAL_SEED \
    --head_camera_type $HEAD_CAMERA_TYPE \
    --max_steps $MAX_STEPS \
    --run_dir $RUN_DIR \
    --alg_name $alg_name \
    --task_config $TASK_CONFIG \
    --instruction_type $INSTRUCTION_TYPE \
    --action_space $ACTION_SPACE"

# 根据是否批量评估添加不同参数
if [ ! -z "$START_EPOCH" ] && [ ! -z "$END_EPOCH" ]; then
    # 批量评估模式
    CMD="$CMD --start_epoch $START_EPOCH --end_epoch $END_EPOCH --epoch_interval $EPOCH_INTERVAL --wandb_mode $WANDB_MODE --wandb_project $WANDB_PROJECT"
else
    # 单个评估模式
    CMD="$CMD --epoch $EPOCH"
fi

# 执行命令
eval $CMD
