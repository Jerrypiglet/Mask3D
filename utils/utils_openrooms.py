from tqdm import tqdm

openrooms_semantics_black_list = [ # scenes whose rendered semantic labels are not consistent with XML scene (e.g. misplacement of objects; missing/adding objects)
    ('main_xml1', 'scene0386_00'), 
    ('main_xml', 'scene0386_00'), 
    ('main_xml', 'scene0608_01'), 
    ('main_xml1', 'scene0608_01'), 
    ('main_xml1', 'scene0211_02'), 
    ('main_xml1', 'scene0126_02'), 
]

def get_im_info_list(frame_list_root, split):
    frame_list_path = frame_list_root / ('%s.txt'%split)
    assert frame_list_path.exists(), frame_list_path
    scene_list = []
    with open(frame_list_path, 'r') as f:
        frame_list = f.read().splitlines()
        # print(len(frame_list), frame_list[0])
        for frame_info in tqdm(frame_list):
            scene_name, frame_id, im_sdr_file, imsemLabel_path = frame_info.split(' ')
            meta_split, scene_name_, im_sdr_name = im_sdr_file.split('/')
            assert scene_name == scene_name_
            assert im_sdr_name.split('.')[0].split('_')[1] == str(frame_id)
            if (meta_split, scene_name) not in scene_list:
                scene_list.append((meta_split, scene_name))
            
    print('Found %d scenes, %d frames in %s'%(len(scene_list), len(frame_list), split))
    print(scene_list[:5])
    return scene_list
