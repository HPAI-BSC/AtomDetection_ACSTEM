from typing import List

import argparse
import os

from atoms_detection.cv_detection import CVDetection
from atoms_detection.evaluation import Evaluation
from utils.paths import LOGS_PATH, DETECTION_PATH, PREDS_PATH


def cv_full_pipeline(
        extension_name: str,
        coords_csv: str,
        thresholds_list: List[float],
        force: bool = False
):

    # DL Detection & Evaluation
    for threshold in thresholds_list:
        inference_cache_path = os.path.join(PREDS_PATH, f"cv_detection_{extension_name}")
        detections_path = os.path.join(DETECTION_PATH, f"cv_detection_{extension_name}_{threshold}")
        if force or not os.path.exists(detections_path):
            print(f"Detecting atoms on test data with threshold={threshold}...")
            detection = CVDetection(
                dataset_csv=coords_csv,
                threshold=threshold,
                detections_path=detections_path,
                inference_cache_path=inference_cache_path
            )
            detection.run()

        logging_filename = os.path.join(LOGS_PATH, f"cv_detection_{extension_name}_{threshold}.csv")
        if force or not os.path.exists(logging_filename):
            evaluation = Evaluation(
                coords_csv=coords_csv,
                predictions_path=detections_path,
                logging_filename=logging_filename
            )
            evaluation.run()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "extension_name",
        type=str,
        help="Experiment extension name"
    )
    parser.add_argument(
        "coords_csv",
        type=str,
        help="Coordinates CSV file to use as input"
    )
    parser.add_argument(
        "-t"
        "--thresholds",
        nargs="+",
        type=float,
        help="Coordinates CSV file to use as input"
    )
    parser.add_argument(
        "--force",
        action="store_true"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    cv_full_pipeline(args.extension_name, args.coords_csv, args.t__thresholds, args.force)
