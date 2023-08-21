import re
from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
from fire import Fire
from natsort import natsorted
from loguru import logger
import pickle
from utils.utils_openrooms import get_im_info_list, openrooms_semantics_black_list

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply
from termcolor import colored

def yellow(text):
    coloredd = colored(text, 'blue', 'on_yellow')
    return coloredd

def read_exclude_scenes_list(dump_root: Path, split: str, scene_list: list=[]):
    exclude_scene_list_files = list(dump_root.glob('excluded_scenes_%s*.txt'%split))
    exclude_scene_list = []
    for exclude_scene_list_file_ in exclude_scene_list_files:
        with open(str(exclude_scene_list_file_), 'r') as f:
            lines = f.readlines()
        exclude_scene_list += [tuple(line.strip().split()) for line in lines]
    print(yellow('Excluded scenes:'), len(exclude_scene_list), exclude_scene_list[:5])
    
    if scene_list != []:
        scene_list = [scene for scene in scene_list if scene not in exclude_scene_list]        
        print(yellow('The rest of the scenes:'), len(scene_list))
        
    return exclude_scene_list, scene_list

class OpenroomPublicPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "/data/Mask3D_data/openrooms_public_dump",
        save_dir: str = "./data/processed/openrooms_public_trainval",
        modes: tuple = ("train", "validation"),
        # modes: tuple = ("validation"),
        OR_version: str = 'OR45',
        OR_modality: str = 'instance_seg',
        not_val_labels: list = [255], 
        # modes: tuple = ("validation"),
        n_jobs: int = -1,
        rui_indoorinv_data_repo_path: str = "/home/ruizhu/Documents/Projects/rui-indoorinv-data",
    ):
        if isinstance(modes, str):
            modes = (modes,)
        super().__init__(data_dir, save_dir, modes, n_jobs)

        rui_indoorinv_data_repo_path = Path(rui_indoorinv_data_repo_path)
        
        self.OR_version = OR_version
        assert self.OR_version in ['OR42', 'OR45']
        self.OR_modality = OR_modality
        assert self.OR_modality in ['instance_seg', 'matseg']
        self.not_val_labels = not_val_labels
        
        self.create_label_database(rui_indoorinv_data_repo_path)
        
        # import ipdb; ipdb.set_trace()
        
        self.save_dir = Path(save_dir)
        
        for mode in self.modes:
            
            split = {'train': 'train', 'validation': 'val', 'test': 'test'}[mode]
            
            scene_list = get_im_info_list(rui_indoorinv_data_repo_path / 'files_openrooms' / 'public', split)
            print('[%s] %d scenes'%(split, len(scene_list)))
            scene_list = [scene for scene in scene_list if (scene[0].replace('DiffMat', '').replace('DiffLight', ''), scene[1]) not in openrooms_semantics_black_list]
            print('[%s] after removing known invalid scenes from [openrooms_semantics_black_list]-> %d scenes'%(split, len(scene_list)))

            # excluded_scenes_file = Path(data_dir) / ('excluded_scenes_%s.txt'%split)
            # if excluded_scenes_file.exists():
            #     with open(str(excluded_scenes_file), 'r') as f:
            #         excluded_scenes = f.read().splitlines()
            #     excluded_scenes = [(scene.split(' ')[0].replace('DiffMat', '').replace('DiffLight', ''), scene.split(' ')[1]) for scene in excluded_scenes]
            #     scene_list = [scene for scene in scene_list if (scene[0].replace('DiffMat', '').replace('DiffLight', ''), scene[1]) not in excluded_scenes]
            #     print('[%s] after removing known invalid scenes [from %s] -> %d scenes'%(split, excluded_scenes_file.name, len(scene_list)))
            
            _, scene_list = read_exclude_scenes_list(Path(data_dir), split, scene_list)

            
            # scene_list_dict[split] = scene_list

            filepaths = []
            for (meta_split, scene_name) in scene_list:
                scene_dump_root = Path(data_dir) / meta_split / scene_name
                tsdf_path = scene_dump_root / 'fused_tsdf.ply'
                filepaths.append(tsdf_path)
                
            self.files[mode] = natsorted(filepaths)
            
        # self.compute_color_mean_std()

    def create_label_database(self, rui_indoorinv_data_repo_path):
        '''
        for fused 2D semseg labels: OR45
            255: unlabelled
            0...44: curtain...ceiling
        '''
        
        semantic_labels_root = rui_indoorinv_data_repo_path / 'files_openrooms'
        
        # OR_mapping_obj_cat_str_to_id_file = semantic_labels_root / 'semanticLabelName_OR42.txt'
        # with open(str(OR_mapping_obj_cat_str_to_id_file)) as f:
        #     mylist = f.read().splitlines()
        # OR_mapping_obj_cat_str_to_id42_name_dict = {x.split(' ')[0]: (int(x.split(' ')[1]), x.split(' ')[2]) for x in mylist} # cat id is 0-based (0 being unlabelled)!
        # OR_mapping_obj_cat_str_to_id42_name_dict = {k: (v[0]-1, v[1]) for k, v in OR_mapping_obj_cat_str_to_id42_name_dict.items()}
        
        # OR_mapping_id42_to_name_dict = {v[0]: v[1] for k, v in OR_mapping_obj_cat_str_to_id42_name_dict.items()}
        # OR_mapping_id42_to_name_dict[255] = 'unlabelled'

        '''
        names
        '''
        OR_names45_file = semantic_labels_root / 'colors/openrooms_names.txt'
        with open(str(OR_names45_file)) as f:
            mylist = f.read().splitlines()
        OR_mapping_id45_to_name_dict = {_: '_'.join(x.split('_')[:-1]) for _, x in enumerate(mylist)} # cat id is 0-based (255 being unlabelled)!
        OR_mapping_id45_to_name_dict = {k-1: v for k, v in OR_mapping_id45_to_name_dict.items() if k != 0}
        OR_mapping_id45_to_name_dict[255] = 'unlabelled'
        
        OR_mapping_obj_cat_str_to_id_file = semantic_labels_root / 'semanticLabelName_OR42.txt'
        with open(str(OR_mapping_obj_cat_str_to_id_file)) as f:
            mylist = f.read().splitlines()
        OR_mapping_obj_cat_str_to_id42_name_dict = {x.split(' ')[0]: (int(x.split(' ')[1]), x.split(' ')[2]) for x in mylist} # cat id is 0-based (0 being unlabelled)!
        OR_mapping_obj_cat_str_to_id42_name_dict = {k: (v[0]-1, v[1]) for k, v in OR_mapping_obj_cat_str_to_id42_name_dict.items()} # {'curtain': (0, 'curtain'), '03790512': (1, 'bike'), ...}
        
        OR_mapping_id42_to_name_dict = {v[0]: v[1] for k, v in OR_mapping_obj_cat_str_to_id42_name_dict.items()}
        OR_mapping_id42_to_name_dict[255] = 'unlabelled'

        
        '''
        colors
        '''
        OR_mapping_id45_to_color_file = semantic_labels_root / 'colors/OR4X_mapping_catInt_to_RGB_light.pkl'
        with (open(OR_mapping_id45_to_color_file, "rb")) as f:
            OR4X_mapping_catInt_to_RGB_light = pickle.load(f)
            
        OR_mapping_id45_to_color_dict = OR4X_mapping_catInt_to_RGB_light['OR45']
        OR_mapping_id45_to_color_dict = {k-1: v for k, v in OR_mapping_id45_to_color_dict.items() if k != 0}
        OR_mapping_id45_to_color_dict[255] = (255, 255, 255) # unlabelled
        
        OR_mapping_id42_to_color_dict = OR4X_mapping_catInt_to_RGB_light['OR42']
        OR_mapping_id42_to_color_dict = {k-1: v for k, v in OR_mapping_id42_to_color_dict.items() if k != 0}
        OR_mapping_id42_to_color_dict[255] = (255, 255, 255) # unlabelled

        label_database = {}
        if self.OR_version == 'OR45':
            for class_id in OR_mapping_id45_to_color_dict.keys():
                label_database[class_id] = {
                    "color": OR_mapping_id45_to_color_dict[class_id],
                    "name": OR_mapping_id45_to_name_dict[class_id],
                    # "validation": class_id not in [40, 41, 42, 255], # 40: wall, 41: floor, 42: ceiling
                    # "validation": class_id not in [42, 43, 44, 255], # 42: wall, 43: floor, 44: ceiling
                    "validation": class_id not in self.not_val_labels, 
                }
        elif self.OR_version == 'OR42':
            for class_id in OR_mapping_id42_to_color_dict.keys():
                label_database[class_id] = {
                    "color": OR_mapping_id42_to_color_dict[class_id],
                    "name": OR_mapping_id42_to_name_dict[class_id],
                    "validation": class_id not in self.not_val_labels, 
                }
            
        self._save_yaml(
            self.save_dir / "label_database.yaml", label_database
        )
        return label_database
    
    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        meta_split, scene_name = self._parse_scene_subscene(filepath)
        filebase = {
            "filepath": filepath,
            "scene": '-'.join([meta_split, scene_name]),
            "sub_scene": 0,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        # reading both files and checking that they are fitting
        coords, features, _ = load_ply(filepath)
        normals = np.load(filepath.parent / 'mi_normal.npy')
        features = np.hstack((features, normals))
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features))
        
        vertex_view_count = np.load(filepath.parent / 'vertex_view_count.npy')
        assert len(vertex_view_count) == len(coords), "vertex_view_count doesn't fit"
        vertex_mask = vertex_view_count > 0

        if mode in ["train", "validation"]:
            # getting scene information
            filebase["scene_type"] = ''
            filebase["raw_description_filepath"] = ''

            # getting instance info
            segment_indexes_filepath = next(
                Path(filepath).parent.glob("*[0-9].segs.json")
            )
            assert segment_indexes_filepath is not None, "no segment indexes"
            segments = self._read_json(segment_indexes_filepath)
            segments = np.array(segments["segIndices"])
            
            if len(segments) != len(coords):
                print(self.files[mode].index(filepath), meta_split, scene_name, len(segments), len(coords))
                raise Exception("segmentation doesn't fit")

            assert len(segments) == len(coords), "segmentation doesn't fit"
            filebase["raw_instance_filepath"] = ''
            filebase["raw_segmentation_filepath"] = segment_indexes_filepath

            # add segment id as additional feature
            segment_ids = np.unique(segments, return_inverse=True)[1]
            points = np.hstack((points, segment_ids[..., None]))
            
            # reading semseg labels: 255-unlabelled
            if self.OR_modality == 'instance_seg':
                semseg_filepath = filepath.parent / 'semseg.npy'
            elif self.OR_modality == 'matseg':
                semseg_filepath = filepath.parent / 'matseg_sem.npy'
            assert semseg_filepath.exists(), semseg_filepath
            semseg = np.load(semseg_filepath).astype(np.int64)
            assert len(semseg) == len(coords), "semseg doesn't fit"
            # assert np.amax(semseg[semseg!=255]) < 42 # 0, 1, ..., 41
            if not np.amax(semseg[semseg!=255]) < 45:
                print(np.unique(semseg))
            # semseg += 1
            # semseg[semseg == 256] = 0 # unlabelled
            
            # reading instance seg labels: -1-unlabelled
            if self.OR_modality == 'instance_seg':
                instance_seg_filepath = filepath.parent / 'instance_seg.npy'
            elif self.OR_modality == 'matseg':
                instance_seg_filepath = filepath.parent / 'matseg.npy'
                
            assert instance_seg_filepath.exists(), instance_seg_filepath
            instance_seg = np.load(instance_seg_filepath).astype(np.int64)
            assert len(instance_seg) == len(coords), "instance_seg doesn't fit"
            _num_instance_seg = len(np.unique(instance_seg))
            instance_seg[instance_seg == 255] = -1 # unlabelled
            instance_seg[instance_seg!=-1] = np.unique(instance_seg[instance_seg!=-1], return_inverse=True)[1] # re-index instance_seg ids to -1 (unlabelled), 0, 1, 2, ..., N
            assert len(np.unique(instance_seg)) == _num_instance_seg
            
            labels = np.hstack((semseg[..., None], instance_seg[..., None]))
            
            # adding semantic + instance label
            points = np.hstack((points, labels))

            # gt_data = (points[:, -2] + 1) * 1000 + points[:, -1] + 1
            gt_data = points[:, -2] * 1000 + points[:, -1] + 1
        else:
            assert False, 'test mode not supported for openroo#ms_public'
            
        points = points[vertex_mask]
        gt_data = gt_data[vertex_mask]
        features = features[vertex_mask]

        processed_filepath = (
            self.save_dir / mode / f"{meta_split}-{scene_name}.npy"
        )
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        if mode == "test":
            return filebase

        processed_gt_filepath = (
            self.save_dir
            / "instance_gt"
            / mode
            / f"{meta_split}-{scene_name}.txt"
        )
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((features[:, 0] / 255).mean()),
            float((features[:, 1] / 255).mean()),
            float((features[:, 2] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((features[:, 0] / 255) ** 2).mean()),
            float(((features[:, 1] / 255) ** 2).mean()),
            float(((features[:, 2] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std(
        self,
        train_database_path: str = "./data/processed/openrooms_public_trainval/train_database.yaml",
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    # @logger.catch
    # def fix_bugs_in_labels(self):
    #     if not self.scannet200:
    #         logger.add(self.save_dir / "fixed_bugs_in_labels.log")
    #         found_wrong_labels = {
    #             tuple([270, 0]): 50,
    #             tuple([270, 2]): 50,
    #             tuple([384, 0]): 149,
    #         }
    #         for scene, wrong_label in found_wrong_labels.items():
    #             scene, sub_scene = scene
    #             bug_file = (
    #                 self.save_dir / "train" / f"{meta_split}-{scene_name}.npy"
    #             )
    #             points = np.load(bug_file)
    #             bug_mask = points[:, -1] != wrong_label
    #             points = points[bug_mask]
    #             np.save(bug_file, points)
    #             logger.info(f"Fixed {bug_file}")

    def _parse_scene_subscene(self, name):
        meta_split, scene_name = str(name).split('/')[-3:-1]
        # scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        # return int(scene_match.group(1)), int(scene_match.group(2))
        return meta_split, scene_name


if __name__ == "__main__":
    Fire(OpenroomPublicPreprocessing)
