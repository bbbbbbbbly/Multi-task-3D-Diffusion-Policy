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
for task in task_list:
    print(f'Converting {task}...')
    subprocess.run(['python', 'script/pkl2zarr_dp3.py', task, 'D435','100'], check=True)
    print(f'Finished converting {task}.')