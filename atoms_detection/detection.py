from typing import Tuple, List

import os
from hashlib import sha1

import numpy as np
from PIL import Image
from scipy.ndimage import label

from utils.constants import Split
from utils.paths import PREDS_PATH
from atoms_detection.dataset import ImageDataset


class Detection:
    def __init__(self, dataset_csv: str, threshold: float, detections_path: str, inference_cache_path: str):
        self.image_dataset = ImageDataset(dataset_csv)
        self.threshold = threshold
        self.detections_path = detections_path
        self.inference_cache_path = inference_cache_path
        if not os.path.exists(self.detections_path):
            os.makedirs(self.detections_path)
        if not os.path.exists(self.inference_cache_path):
            os.makedirs(self.inference_cache_path)

    def image_to_pred_map(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pred_map_to_atoms(self, pred_map: np.ndarray) -> Tuple[List[Tuple[int, int]], List[float]]:
        pred_mask = pred_map > self.threshold
        labeled_array, num_features = label(pred_mask)

        # Convert labelled_array to indexes
        center_coords_list = []
        likelihood_list = []
        for label_idx in range(num_features+1):
            if label_idx == 0:
                continue
            label_mask = np.where(labeled_array == label_idx)
            likelihood = np.max(pred_map[label_mask])
            likelihood_list.append(likelihood)
            # label_size = len(label_mask[0])
            # print(f"\t\tAtom {label_idx}: {label_size}")
            atom_bbox = (label_mask[0].min(), label_mask[0].max(), label_mask[1].min(), label_mask[1].max())
            center_coord = self.bbox_to_center_coords(atom_bbox)
            center_coords_list.append(center_coord)

        return center_coords_list, likelihood_list

    def detect_atoms(self, img_filename: str) -> Tuple[List[Tuple[int, int]], List[float]]:
        img_hash = sha1(img_filename.encode()).hexdigest()
        prediciton_cache = os.path.join(self.inference_cache_path, f"{img_hash}.npy")
        if not os.path.exists(prediciton_cache):
            img = self.open_image(img_filename)
            pred_map = self.image_to_pred_map(img)
            np.save(prediciton_cache, pred_map)
        else:
            pred_map = np.load(prediciton_cache)
        center_coords_list, likelihood_list = self.pred_map_to_atoms(pred_map)
        return center_coords_list, likelihood_list

    @staticmethod
    def bbox_to_center_coords(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x_center = (bbox[0] + bbox[1]) // 2
        y_center = (bbox[2] + bbox[3]) // 2
        return x_center, y_center

    @staticmethod
    def open_image(img_filename: str):
        img = Image.open(img_filename)
        np_img = np.asarray(img).astype(np.float32)
        return np_img

    def run(self):
        if not os.path.exists(self.detections_path):
            os.makedirs(self.detections_path)

        for image_path in self.image_dataset.iterate_data(Split.TEST):
            print(f"Running detection on {os.path.basename(image_path)}")
            center_coords_list, likelihood_list = self.detect_atoms(image_path)

            image_filename = os.path.basename(image_path)
            img_name = os.path.splitext(image_filename)[0]
            detection_csv = os.path.join(self.detections_path, f"{img_name}.csv")
            with open(detection_csv, "w") as _csv:
                _csv.write("Filename,x,y,Likelihood\n")
                for (x, y), likelihood in zip(center_coords_list, likelihood_list):
                    _csv.write(f"{image_filename},{x},{y},{likelihood}\n")
