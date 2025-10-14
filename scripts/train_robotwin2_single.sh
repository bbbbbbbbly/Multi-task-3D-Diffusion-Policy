# Examples:
# bash scripts/train_robotwin2_single.sh dp3_robotwin2 beat_block_hammer 9999 0 2
# bash scripts/train_robotwin2_single.sh dp3_robotwin2 beat_block_hammer 9999 0 2,3
# bash scripts/train_robotwin2_single.sh dp3_uni3d_scratch_robotwin2 beat_block_hammer 9999 0 2,3

DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
setting="demo_clean"
# expert_data_num="100"
addition_info=${3}
seed=${4}
exp_name=${task_name}-${alg_name}-${setting}-${addition_info}
run_dir="data/outputs/robotwin2_${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Check if gpu_id contains comma (multi-GPU mode)
if [[ $gpu_id == *","* ]]; then
    # Multi-GPU DDP mode
    echo -e "\033[32mMulti-GPU DDP mode detected!\033[0m"
    IFS=',' read -ra GPU_ARRAY <<< "$gpu_id"
    num_gpus=${#GPU_ARRAY[@]}
    echo -e "\033[32mUsing ${num_gpus} GPUs: ${gpu_id}\033[0m"

    # Set environment variables for DDP
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    export WORLD_SIZE=${num_gpus}
    export MASTER_ADDR="localhost"
    export MASTER_PORT="12355"

    USE_DDP=true
else
    # Single GPU mode
    echo -e "\033[32mSingle GPU mode detected!\033[0m"
    export CUDA_VISIBLE_DEVICES=${gpu_id}
    USE_DDP=false
fi

if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy

run_dir="$(pwd)/data/outputs/robotwin2_${exp_name}_seed${seed}"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# export MUJOCO_GL=osmesa

# unset LD_PRELOAD

if [ $USE_DDP = true ]; then
    # Multi-GPU DDP training
    echo -e "\033[32mStarting DDP training with ${num_gpus} GPUs...\033[0m"
    torchrun \
        --nproc_per_node=${num_gpus} \
        --master_port=12355 \
        train.py --config-name=${config_name}.yaml \
                            task="robotwin2_demo_task" \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda" \
                            training.use_ddp=true \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            setting=${setting}
else
    # Single GPU training
    echo -e "\033[32mStarting single GPU training...\033[0m"
    python train.py --config-name=${config_name}.yaml \
                            task="robotwin2_demo_task" \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            training.use_ddp=false \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            setting=${setting}
fi



