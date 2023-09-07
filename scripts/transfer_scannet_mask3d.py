'''
gather all files of extarcted ScanNet which are required by Mask3D, into a txt file

to transfer: 

rclone copy --progress --fast-list --checkers=128 --transfers=128 --files-from scannet_list.txt /newfoundland/ScanNet liwen:/home/rzhu/Documents/data/ScanNet

'''

from pathlib import Path
import json
from tqdm import tqdm

SCANNET_ROOT_ = Path('/newfoundland/ScanNet')

file_list_all_path = Path('scannet_list.txt')
file_list_all = []

for split in ['val']:
# for split in ['train', 'val', 'test']:
    if split in ['train', 'val']:
        SCANNET_ROOT = SCANNET_ROOT_ / 'scans'
        file_list = [
            '.txt', 
            '_vh_clean_2.ply', 
            '_vh_clean_2.labels.ply', 
            '.aggregation.json', 
            '*[0-9].segs.json', 
        ]
    else:
        SCANNET_ROOT = SCANNET_ROOT_ / 'scans_test'
        file_list = [
            '.txt', 
            '_vh_clean_2.ply', 
        ]

    scene_name_list_path = Path(f'/home/ruizhu/Documents/Projects/ScanNet/Tasks/Benchmark/scannetv2_{split}.txt')
    assert scene_name_list_path.exists()
    with open(scene_name_list_path, 'r') as f:
        scene_name_list = [_.strip() for _ in f.readlines()]
        
    for scene_name in scene_name_list:
        scene_path = SCANNET_ROOT / scene_name
        assert scene_path.exists()
        
        for file_name_ in file_list:
            # file_name = file_name_.replace('*', scene_name)
            file_name = scene_name + file_name_
            # print('====', file_name_)
            file_count_ = 0
            for _ in scene_path.glob(file_name):
                file_count_ += 1
                file_path_rel = _.relative_to(SCANNET_ROOT_)
                file_list_all.append(file_path_rel)
                # print(file_path_rel)
            assert file_count_ == 1
            
with open(file_list_all_path, 'w') as f:
    for _ in file_list_all:
        f.write(str(_) + '\n')
print(f'Write {len(file_list_all)} files to {file_list_all_path}')