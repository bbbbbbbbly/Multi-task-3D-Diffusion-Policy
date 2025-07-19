import subprocess
task_list = [
    'block_hammer_beat',
    'block_handover',
    'blocks_stack_easy',
    'blocks_stack_hard',
    'bottle_adjust',
    'container_place',
    'diverse_bottles_pick',
    'dual_bottles_pick_easy',
    'dual_bottles_pick_hard',
    'dual_shoes_place',
    'empty_cup_place',
    'mug_hanging_easy',
    'mug_hanging_hard',
    'pick_apple_messy',
    'put_apple_cabinet',
    'shoe_place',
    'tool_adjust',
]
gpu_id=3
for id, task in enumerate(task_list):
    if int(id) % 4 == gpu_id:
        print(f"Running task {task} (index {id}) on GPU {gpu_id}")
        #bash train.sh ${task_name} ${head_camera_type} ${expert_data_num} ${seed} ${gpu_id}
        # subprocess.run(['bash', 'train_rgb.sh', task, 'D435', '100', '0', str(gpu_id)])
        #bash eval.sh ${task_name} ${head_camera_type} ${expert_data_num} ${checkpoint_num} ${seed} ${gpu_id}
        subprocess.run(['bash', 'eval_rgb.sh', task, 'D435', '100', '3000', '0', str(gpu_id)])
    else:
        print(f"Skipping task {task} (index {id}) for GPU {gpu_id}")
print("All tasks submitted.")
