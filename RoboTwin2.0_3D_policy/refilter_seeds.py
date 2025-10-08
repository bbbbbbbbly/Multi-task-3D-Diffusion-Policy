import json
import os

def filter_seeds_by_episode_info(file_path):
    """
    读取JSON文件，删除episode_info为"right"的seed，只保留"left"的seed
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 遍历每个主键（如"20000"）
    for main_key in data:
        original_seeds = data[main_key]["seeds"].copy()
        episode_info = data[main_key]["episode_info"]
        
        # 找出所有episode_info为"left"的seed
        left_seeds = []
        for seed in original_seeds:
            seed_str = str(seed)
            if seed_str in episode_info:
                # 检查episode_info中的"{a}"值
                for info in episode_info[seed_str]:
                    if info.get("{a}") == "left":
                        left_seeds.append(seed)
                        break
        
        # 更新seeds列表，只保留left的seeds
        data[main_key]["seeds"] = left_seeds
        
        print(f"文件 {os.path.basename(file_path)}:")
        print(f"  原始seeds数量: {len(original_seeds)}")
        print(f"  过滤后seeds数量: {len(left_seeds)}")
        print(f"  删除的seeds数量: {len(original_seeds) - len(left_seeds)}")
        
        # 显示删除的seeds（用于验证）
        removed_seeds = [seed for seed in original_seeds if seed not in left_seeds]
        print(f"  删除的seeds: {removed_seeds[:10]}{'...' if len(removed_seeds) > 10 else ''}")
    
    # 保存修改后的文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data

def main():
    # 只处理指定的两个文件
    seeds_dir = "/data/sea_disk0/wushr/3D-Policy/DP3Encoder-to-Uni3D/3D-Diffusion-Policy/RoboTwin2.0_3D_policy/seeds_list_left"
    
    target_files = ["stamp_seal.json", "place_fan.json"]
    
    print(f"重新过滤以下文件:")
    for file in target_files:
        print(f"  - {file}")
    print()
    
    # 处理每个JSON文件
    for json_file in target_files:
        file_path = os.path.join(seeds_dir, json_file)
        if os.path.exists(file_path):
            print(f"正在重新处理: {json_file}")
            filter_seeds_by_episode_info(file_path)
            print()
        else:
            print(f"文件不存在: {json_file}")
    
    print("重新过滤完成！")

if __name__ == "__main__":
    main()
