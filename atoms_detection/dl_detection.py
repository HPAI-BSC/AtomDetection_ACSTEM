import os
from typing import Tuple, List
import argparse

import torch
import numpy as np
import torch.nn
import torch.nn.functional

from atoms_detection.detection import Detection
from atoms_detection.training_model import model_pipeline
from atoms_detection.image_preprocessing import prepro_image
from utils.constants import ModelArgs
from utils.paths import PREDS_PATH, MODELS_PATH, DETECTION_PATH


class DLDetection(Detection):
    def __init__(self,
                 model_name: ModelArgs,
                 ckpt_filename: str,
                 dataset_csv: str,
                 threshold: float,
                 detections_path: str,
                 inference_cache_path: str):
        self.model_name = model_name
        self.ckpt_filename = ckpt_filename
        self.device = self.get_torch_device()
        self.batch_size = 64

        self.stride = 1
        self.padding = 10
        self.window_size = (21, 21)

        super().__init__(dataset_csv, threshold, detections_path, inference_cache_path)

    @staticmethod
    def get_torch_device():
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device

    def sliding_window(self, image: np.ndarray, padding: int = 0) -> Tuple[int, int, np.ndarray]:
        # slide a window across the image
        x_to_center = self.window_size[0] // 2 - 1 if self.window_size[0] % 2 == 0 else self.window_size[0] // 2
        y_to_center = self.window_size[1] // 2 - 1 if self.window_size[1] % 2 == 0 else self.window_size[1] // 2

        for y in range(0, image.shape[0] - self.window_size[1]+1, self.stride):
            for x in range(0, image.shape[1] - self.window_size[0]+1, self.stride):
                # yield the current window
                center_x = x + x_to_center
                center_y = y + y_to_center
                yield center_x-padding, center_y-padding, image[y:y + self.window_size[1], x:x + self.window_size[0]]

    def batch_sliding_window(self, image: np.ndarray, padding: int = 0) -> Tuple[List[int], List[int], List[np.ndarray]]:
        x_idx_list = []
        y_idx_list = []
        images_list = []
        count = 0
        for _x, _y, _img in self.sliding_window(image, padding=padding):
            x_idx_list.append(_x)
            y_idx_list.append(_y)
            images_list.append(_img)
            count += 1
            if count == self.batch_size:
                yield x_idx_list, y_idx_list, images_list
                x_idx_list = []
                y_idx_list = []
                images_list = []
                count = 0
        if count != 0:
            yield x_idx_list, y_idx_list, images_list

    def padding_image(self, img: np.ndarray) -> np.ndarray:
        image_padded = np.zeros((img.shape[0] + self.padding*2, img.shape[1] + self.padding*2))
        image_padded[self.padding:-self.padding, self.padding:-self.padding] = img
        return image_padded

    def load_model(self) -> torch.nn.Module:
        checkpoint = torch.load(self.ckpt_filename, map_location=self.device)

        model = model_pipeline[self.model_name](num_classes=2).to(self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def images_to_torch_input(self, images_list: List[np.ndarray]) -> torch.Tensor:
        expanded_img = np.expand_dims(images_list, axis=1)
        input_tensor = torch.from_numpy(expanded_img).float()
        input_tensor = input_tensor.to(self.device)
        return input_tensor

    def get_prediction_map(self, padded_image: np.ndarray) -> np.ndarray:
        _shape = padded_image.shape
        pred_map = np.zeros((_shape[0] - self.padding*2, _shape[1] - self.padding*2))
        model = self.load_model()
        for x_idxs, y_idxs, image_crops in self.batch_sliding_window(padded_image, padding=self.padding):
            torch_input = self.images_to_torch_input(image_crops)
            output = model(torch_input)
            pred_prob = torch.nn.functional.softmax(output, 1)
            pred_prob = pred_prob.detach().cpu().numpy()[:, 1]
            pred_map[np.array(y_idxs), np.array(x_idxs)] = pred_prob
        return pred_map

    def image_to_pred_map(self, img: np.ndarray) -> np.ndarray:
        preprocessed_img = prepro_image(img)
        padded_image = self.padding_image(preprocessed_img)
        pred_map = self.get_prediction_map(padded_image)
        return pred_map


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_csv",
        type=str,
        help="Dataset CSV file containing all images to test"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    dataset_name = os.path.splitext(os.path.basename(args.dataset_csv))[0]
    inference_cache_path = os.path.join(PREDS_PATH, f"dl_cache_{dataset_name}")
    detections_path = os.path.join(DETECTION_PATH, f"dl_detection_{dataset_name}_0.89")

    detection = DLDetection(
        model_name=ModelArgs.BASICCNN,
        ckpt_filename=os.path.join(MODELS_PATH, "model_existing.ckpt"),
        dataset_csv=args.dataset_csv,
        threshold=0.89,
        detections_path=detections_path,
        inference_cache_path=inference_cache_path
    )
    detection.run()
