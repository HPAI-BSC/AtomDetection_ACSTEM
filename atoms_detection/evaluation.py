from typing import Optional, Tuple, List

import os

import numpy as np
import scipy.optimize
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import patches

from utils.constants import Split
from atoms_detection.dataset import CoordinatesDataset


def bbox_iou(bb1, bb2):
    assert bb1[0] <= bb1[1]
    assert bb1[2] <= bb1[3]
    assert bb2[0] <= bb2[1]
    assert bb2[2] <= bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def match_bboxes(iou_matrix, IOU_THRESH=0.5):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.

    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true, n_pred = iou_matrix.shape
    MIN_IOU = 0.0
    MAX_DIST = 1.0

    if n_pred > n_true:
        # there are more predictions than ground-truth - add dummy rows
        diff = n_pred - n_true
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((diff, n_pred), MIN_IOU)),
                                    axis=0)

    if n_true > n_pred:
        # more ground-truth than predictions - add dummy columns
        diff = n_true - n_pred
        iou_matrix = np.concatenate((iou_matrix,
                                     np.full((n_true, diff), MIN_IOU)),
                                    axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred < n_pred
    idx_pred_actual = idxs_pred[sel_pred]
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label


class Evaluation:
    def __init__(self, coords_csv: str, predictions_path: str, logging_filename: str):
        self.coordinates_dataset = CoordinatesDataset(coords_csv)
        self.predictions_path = predictions_path
        self.logging_filename = logging_filename
        if not os.path.exists(os.path.dirname(self.logging_filename)):
            os.makedirs(os.path.dirname(self.logging_filename))
        self.logs_df = pd.DataFrame(columns=["Filename", "Precision", "Recall", "F1Score"])
        self.threshold = 0.5

    def get_predictions_dict(self, image_filename: str) -> List[Tuple[int, int]]:
        img_name = os.path.splitext(os.path.basename(image_filename))[0]
        preds_csv = os.path.join(self.predictions_path, f"{img_name}.csv")
        df = pd.read_csv(preds_csv)
        pred_coords_list = []
        for idx, row in df.iterrows():
            pred_coords_list.append((row["x"], row["y"]))
        return pred_coords_list

    @staticmethod
    def center_coords_to_bbox(gt_coord: Tuple[int, int]) -> Tuple[int, int, int, int]:
        box_rwidth, box_rheight = 10, 10
        gt_bbox = (
            gt_coord[0] - box_rwidth,
            gt_coord[0] + box_rwidth + 1,
            gt_coord[1] - box_rheight,
            gt_coord[1] + box_rheight + 1
        )
        return gt_bbox

    def eval_matches(
            self,
            gt_bboxes_dict: List[Tuple[int, int, int, int]],
            atoms_bbox_dict: List[Tuple[int, int, int, int]],
            iou_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        iou_matrix = np.zeros((len(gt_bboxes_dict), len(atoms_bbox_dict))).astype(np.float)

        for gt_idx, gt_bbox in enumerate(gt_bboxes_dict):
            for atom_idx, atom_bbox in enumerate(atoms_bbox_dict):
                iou = bbox_iou(gt_bbox, atom_bbox)
                iou_matrix[gt_idx, atom_idx] = iou
        idxs_true, idxs_pred, ious, labels = match_bboxes(iou_matrix, IOU_THRESH=iou_threshold)
        return idxs_true, idxs_pred, ious, labels

    @staticmethod
    def eval_metrics(n_matches: int, n_gt: int, n_pred: int) -> Tuple[float, float]:
        precision = n_matches / n_pred if n_pred > 0 else 0.0
        if n_gt == 0:
            raise RuntimeError("No ground truth atoms???")
        recall = n_matches / n_gt
        return precision, recall

    def atom_bboxes_to_fixed_bboxes(self, atoms_bboxes_dict: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
        atom_bboxes_dict = []
        for atom_center in atoms_bboxes_dict:
            atom_fixed_bbox = self.center_coords_to_bbox(atom_center)
            atom_bboxes_dict.append(atom_fixed_bbox)
        return atom_bboxes_dict

    def gt_coord_to_bboxes(self, gt_coordinates_dict: List[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
        gt_bboxes_list = []
        for gt_coord in gt_coordinates_dict:
            gt_bbox = self.center_coords_to_bbox(gt_coord)
            gt_bboxes_list.append(gt_bbox)
        return gt_bboxes_list

    @staticmethod
    def open_image(img_filename: str):
        img = Image.open(img_filename)
        np_img = np.asarray(img).astype(np.float32)
        return np_img

    def run(self, plot=False):
        for image_path, coordinates_path in self.coordinates_dataset.iterate_data(Split.TEST):
            img = self.open_image(image_path)

            center_coords_dict = self.get_predictions_dict(image_path)
            atoms_fixed_bboxes_dict = self.atom_bboxes_to_fixed_bboxes(center_coords_dict)

            gt_coordinates = self.coordinates_dataset.load_coordinates(coordinates_path)
            gt_bboxes_dict = self.gt_coord_to_bboxes(gt_coordinates)

            # VISUALILZE gt & pred bboxes!
            if plot:
                plt.figure(figsize=(20, 20))
                ax = plt.gca()
                ax.imshow(img)
                for gt_bbox in gt_bboxes_dict:
                    xy = (gt_bbox[0], gt_bbox[2])
                    width = gt_bbox[1] - gt_bbox[0]
                    height = gt_bbox[3] - gt_bbox[2]
                    rect = patches.Rectangle(xy, width, height, linewidth=3, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                for atom_bbox in atoms_fixed_bboxes_dict:
                    xy = (atom_bbox[0], atom_bbox[2])
                    width = atom_bbox[1] - atom_bbox[0]
                    height = atom_bbox[3] - atom_bbox[2]
                    rect = patches.Rectangle(xy, width, height, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect)
                plt.tight_layout()
                plt.show()

            idxs_true, idxs_pred, ious, labels = self.eval_matches(gt_bboxes_dict, atoms_fixed_bboxes_dict)
            precision, recall = self.eval_metrics(n_matches=len(idxs_pred), n_gt=len(gt_coordinates), n_pred=len(atoms_fixed_bboxes_dict))
            f1_score = 2*(precision*recall)/(precision+recall) if precision+recall > 0 else 0
            if self.logging_filename:
                self.logs_df = self.logs_df.append({
                    "Filename": os.path.basename(image_path),
                    "Precision": precision,
                    "Recall": recall,
                    "F1Score": f1_score
                }, ignore_index=True)
            format_args = (os.path.basename(image_path), precision, recall, f1_score)
            print("{}: F1Score: {}, Precision: {}, Recall: {}".format(*format_args))

        if self.logging_filename:
            mean_precision = self.logs_df["Precision"].mean()
            mean_recall = self.logs_df["Recall"].mean()
            mean_f1_score = self.logs_df["F1Score"].mean()
            std_precision = self.logs_df["Precision"].std()
            std_recall = self.logs_df["Recall"].std()
            std_f1_score = self.logs_df["F1Score"].std()
            print(f"F1Score: {mean_f1_score}, Precision: {mean_precision}, Recall: {mean_recall}")
            self.logs_df = self.logs_df.append({
                "Filename": "Mean",
                "Precision": mean_precision,
                "Recall": mean_recall,
                "F1Score": mean_f1_score
            }, ignore_index=True)
            self.logs_df = self.logs_df.append({
                "Filename": "Std",
                "Precision": std_precision,
                "Recall": std_recall,
                "F1Score": std_f1_score
            }, ignore_index=True)
            self.logs_df.to_csv(self.logging_filename, index=False, float_format='%.4f')
