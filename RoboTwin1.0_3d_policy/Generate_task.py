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
def generate_task(task_name,gpu_id):
    command = ['bash', 'run_task.sh', task_name, str(gpu_id)]
    try:
        subprocess.run(command, check=True)
        print(f"Task {task_name} has been generated successfully on GPU {gpu_id}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while generating task {task_name} on GPU {gpu_id}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
if __name__ == "__main__":
    gpu_id = 2  # Set your desired GPU ID here
    for task in task_list:
        generate_task(task, gpu_id)
    print("All tasks have been generated.")