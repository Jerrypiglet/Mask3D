import re
from pathlib import Path
import numpy as np
import pandas as pd
from fire import Fire
from natsort import natsorted
from loguru import logger

from datasets.preprocessing.base_preprocessing import BasePreprocessing
from utils.point_cloud_utils import load_ply_with_normals

from datasets.scannet200.scannet200_constants import (
    VALID_CLASS_IDS_200,
    SCANNET_COLOR_MAP_200,
    CLASS_LABELS_200,
)


class ScannetPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/raw/scannet/scannet",
        save_dir: str = "./data/processed/scannet",
        # modes: tuple = ("train", "validation", "test"),
        # modes: tuple = ("validation", "test"),
        modes: tuple = ("validation",),
        # modes: tuple = ("test",),
        n_jobs: int = -1,
        git_repo: str = "./data/raw/scannet/ScanNet",
        scannet200: bool = False,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)
        
        assert scannet200 in ['false', 'False', 'true', 'True']
        self.scannet200 = scannet200 in ['true', 'True']

        if self.scannet200:
            self.labels_pd = pd.read_csv(
                self.data_dir / "scannetv2-labels.combined.tsv",
                sep="\t",
                header=0,
            )

        git_repo = Path(git_repo)
        self.label_database = self.create_label_database(git_repo)
        self.valid_class_ids = list(self.label_database.keys())
        self.valid_class_ids.pop(self.valid_class_ids.index(255))
        
        for mode in self.modes:
            # if mode in ['train', 'validation']: continue
            trainval_split_dir = git_repo / "Tasks" / "Benchmark"
            scannet_special_mode = "val" if mode == "validation" else mode
            with open(
                trainval_split_dir / (f"scannetv2_{scannet_special_mode}.txt")
            ) as f:
                # -1 because the last one is always empty
                split_file = f.read().split("\n")[:-1]

            scans_folder = "scans_test" if mode == "test" else "scans"
            filepaths = []
            for scene in split_file:
                filepaths.append(
                    self.data_dir
                    / scans_folder
                    / scene
                    / (scene + "_vh_clean_2.ply")
                )
            self.files[mode] = natsorted(filepaths)
            
        # self.compute_color_mean_std()

    def create_label_database(self, git_repo):
        # '''
        # [TODO] there is a bug... the label names in label_database.yaml are shifted by 1. should be: 0: unlabelled, 1: wall, 2: floor, etc.
        # '''
        # assert False, 'fix the bug above first'
        
        if self.scannet200:
            label_database = {}
            for row_id, class_id in enumerate(VALID_CLASS_IDS_200):
                label_database[class_id] = {
                    "color": SCANNET_COLOR_MAP_200[class_id],
                    "name": CLASS_LABELS_200[row_id],
                    "validation": True,
                }
            self._save_yaml(
                self.save_dir / "label_database.yaml", label_database
            )
            return label_database
        else:
            if (self.save_dir / "label_database.yaml").exists():
                print("label database existed; not creating a new one. " + str(self.save_dir / "label_database.yaml"))
                return self._load_yaml(self.save_dir / "label_database.yaml")
            df = pd.read_csv(
                self.data_dir / "scannetv2-labels.combined.tsv", sep="\t"
            )
            df = (
                df[~df[["nyu40class", "nyu40id"]].duplicated()][
                    ["nyu40class", "nyu40id"]
                ]
                .set_index("nyu40id")
                .sort_index()[["nyu40class"]]
                .rename(columns={"nyu40class": "name"})
                .replace(" ", "_", regex=True)
            )
            # df = pd.DataFrame([{"name": "empty"}]).append(df)
            df = pd.concat([df, pd.DataFrame([{"name": "empty"}])], ignore_index=True)
            df["validation"] = False

            with open(
                git_repo
                / "Tasks"
                / "Benchmark"
                / "classes_SemVoxLabel-nyu40id.txt"
            ) as f:
                for_validation = f.read().split("\n")
                
            for category in for_validation:
                index = int(re.split(" +", category)[0]) - 1 # for_validation is 1-based while df is 0-based
                df.loc[index, "validation"] = True

            # doing this hack because otherwise I will have to install imageio
            with open(git_repo / "BenchmarkScripts" / "util.py") as f:
                util = f.read()
                color_list = eval("[" + util.split("return [\n")[1])

            df["color"] = color_list

            label_database = df.to_dict("index") # 0-based
            
            unlabelled_id = len(df)-1
            assert label_database[unlabelled_id]['name'] == 'empty'
            label_database[255] = label_database[unlabelled_id] # set the unlabelled id to 255
            del label_database[unlabelled_id]
            
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
        scene, sub_scene = self._parse_scene_subscene(filepath.name)
        filebase = {
            "filepath": filepath,
            "scene": scene,
            "sub_scene": sub_scene,
            "raw_filepath": str(filepath),
            "file_len": -1,
        }
        # reading both files and checking that they are fitting
        coords, features, _ = load_ply_with_normals(filepath)
        file_len = len(coords)
        filebase["file_len"] = file_len
        points = np.hstack((coords, features))

        # if mode in ["train", "validation"]:
        if mode in ["train", "validation", "test"]:
            # getting scene information
            description_filepath = Path(
                filepath
            ).parent / filepath.name.replace("_vh_clean_2.ply", ".txt")
            with open(description_filepath) as f:
                scene_type = f.read().split("\n")[:-1]
            scene_type = scene_type[-1].split(" = ")[1]
            filebase["scene_type"] = scene_type
            filebase["raw_description_filepath"] = description_filepath

            # getting instance info
            instance_info_filepath = next(
                Path(filepath).parent.glob("*.aggregation.json")
            )
            segment_indexes_filepath = next(
                Path(filepath).parent.glob("*[0-9].segs.json")
            )
            instance_db = self._read_json(instance_info_filepath)
            print('====', instance_info_filepath)
            segments = self._read_json(segment_indexes_filepath)
            segments = np.array(segments["segIndices"])
            filebase["raw_instance_filepath"] = instance_info_filepath
            filebase["raw_segmentation_filepath"] = segment_indexes_filepath

            # add segment id as additional feature
            segment_ids = np.unique(segments, return_inverse=True)[1]
            points = np.hstack((points, segment_ids[..., None]))

            # reading semantic labels file
            label_filepath = filepath.parent / filepath.name.replace(
                ".ply", ".labels.ply"
            )
            filebase["raw_label_filepath"] = label_filepath
            label_coords, label_colors, labels = load_ply_with_normals(
                label_filepath
            ) # labels: semantic segmentation labels
            assert np.all([_ in self.valid_class_ids for _ in np.unique(labels)])
            
            if not np.allclose(coords, label_coords):
                raise ValueError("files doesn't have same coordinates")

            # adding instance label
            labels = labels[:, np.newaxis]
            empty_instance_label = np.full(labels.shape, -1)
            labels = np.hstack((labels, empty_instance_label))
            for instance in instance_db["segGroups"]:
                segments_occupied = np.array(instance["segments"])
                occupied_indices = np.isin(segments, segments_occupied)
                labels[occupied_indices, 1] = instance["id"]

                if self.scannet200:
                    label200 = instance["label"]
                    # Map the category name to id
                    label_ids = self.labels_pd[
                        self.labels_pd["raw_category"] == label200
                    ]["id"]
                    label_id = (
                        int(label_ids.iloc[0]) if len(label_ids) > 0 else 0
                    )
                    labels[occupied_indices, 0] = label_id
            points = np.hstack((points, labels))

            # gt_data = (points[:, -2] + 1) * 1000 + points[:, -1] + 1
            gt_data = points[:, -2] * 1000 + points[:, -1] + 1
        else:
            segments_test = "../../data/raw/scannet_test_segments"
            segment_indexes_filepath = filepath.name.replace(
                ".ply", ".0.010000.segs.json"
            )
            segments = self._read_json(
                f"{segments_test}/{segment_indexes_filepath}"
            )
            segments = np.array(segments["segIndices"])
            # add segment id as additional feature
            segment_ids = np.unique(segments, return_inverse=True)[1]
            points = np.hstack((points, segment_ids[..., None]))

        processed_filepath = (
            self.save_dir / mode / f"{scene:04}_{sub_scene:02}.npy"
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
            / f"scene{scene:04}_{sub_scene:02}.txt"
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
        train_database_path: str = "./data/processed/scannet/train_database.yaml",
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

    @logger.catch
    def fix_bugs_in_labels(self):
        if not self.scannet200:
            logger.add(self.save_dir / "fixed_bugs_in_labels.log")
            found_wrong_labels = {
                tuple([270, 0]): 50,
                tuple([270, 2]): 50,
                tuple([384, 0]): 149,
            }
            for scene, wrong_label in found_wrong_labels.items():
                scene, sub_scene = scene
                bug_file = (
                    self.save_dir / "train" / f"{scene:04}_{sub_scene:02}.npy"
                )
                points = np.load(bug_file)
                bug_mask = points[:, -1] != wrong_label
                points = points[bug_mask]
                np.save(bug_file, points)
                logger.info(f"Fixed {bug_file}")

    def _parse_scene_subscene(self, name):
        scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        return int(scene_match.group(1)), int(scene_match.group(2))


if __name__ == "__main__":
    Fire(ScannetPreprocessing)
