from typing import List

import argparse
import os

from atoms_detection.create_crop_dataset import create_crops_dataset
from atoms_detection.dl_detection import DLDetection
from atoms_detection.evaluation import Evaluation
from atoms_detection.training_model import train_model
from utils.paths import CROPS_PATH, CROPS_DATASET, MODELS_PATH, LOGS_PATH, DETECTION_PATH, PREDS_PATH
from utils.constants import ModelArgs


def dl_full_pipeline(
        extension_name: str,
        architecture: ModelArgs,
        coords_csv: str,
        thresholds_list: List[float],
        force: bool = False
):
    # Create crops data
    crops_folder = CROPS_PATH + f"_{extension_name}"
    crops_dataset = CROPS_DATASET.replace(".csv", f"_{extension_name}.csv")
    if force or not os.path.exists(crops_dataset):
        print("Creating crops dataset...")
        create_crops_dataset(crops_folder, coords_csv, crops_dataset)

    # training DL model
    ckpt_filename = os.path.join(MODELS_PATH, f"model_{extension_name}.ckpt")
    if force or not os.path.exists(ckpt_filename):
        print("Training DL crops model...")
        train_model(architecture, crops_dataset, crops_folder, ckpt_filename)

    # DL Detection & Evaluation
    for threshold in thresholds_list:
        inference_cache_path = os.path.join(PREDS_PATH, f"dl_detection_{extension_name}")
        detections_path = os.path.join(DETECTION_PATH, f"dl_detection_{extension_name}", f"dl_detection_{extension_name}_{threshold}")
        if force or not os.path.exists(detections_path):
            print(f"Detecting atoms on test data with threshold={threshold}...")
            detection = DLDetection(
                model_name=architecture,
                ckpt_filename=ckpt_filename,
                dataset_csv=coords_csv,
                threshold=threshold,
                detections_path=detections_path,
                inference_cache_path=inference_cache_path
            )
            detection.run()

        logging_filename = os.path.join(LOGS_PATH, f"dl_evaluation_{extension_name}", f"dl_evaluation_{extension_name}_{threshold}.csv")
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
        "architecture",
        type=ModelArgs,
        choices=ModelArgs,
        help="Architecture name"
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
    dl_full_pipeline(args.extension_name, args.architecture, args.coords_csv, args.t__thresholds, args.force)
