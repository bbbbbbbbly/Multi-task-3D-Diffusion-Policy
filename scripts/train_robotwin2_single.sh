# Examples:
# bash scripts/train_robotwin2_single.sh dp3_robotwin2 beat_block_hammer 9999 0 2



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


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# export MUJOCO_GL=osmesa

# unset LD_PRELOAD

export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task="robotwin2_demo_task" \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            setting=${setting}



