from typing import List, Tuple
import os
import glob

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage.filters import gaussian_filter, median_filter, rank_filter
from torch.utils.data import Dataset
from torchvision import transforms

from utils.constants import Split, Columns, CropsColumns, ProbsColumns
from utils.paths import CROPS_DATASET, CROPS_PATH, COORDS_PATH, TIF_PATH, PROBS_DATASET, PROBS_PATH, HAADF_DATASET, COORDS_DATASET


class ImageClassificationDataset(Dataset):

    def __init__(self, image_paths, image_labels, include_filename=False):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.include_filename = include_filename
        self.transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_n_labels(self):
        return len(set(self.image_labels))

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def load_image(img_filename):
        img = Image.open(img_filename)
        np_img = np.asarray(img).astype(np.float32)
        np_bg = median_filter(np_img, size=(40, 40))
        np_clean = np_img - np_bg
        np_normed = (np_clean - np_clean.min()) / (np_clean.max() - np_clean.min())
        return np_normed

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self.load_image(img_path)
        image = self.transform(image)
        label = self.image_labels[idx]

        if self.include_filename:
            return image, label, os.path.basename(img_path)
        else:
            return image, label

    @staticmethod
    def get_filenames_labels(split: Split) -> Tuple[List[str], List[int]]:
        raise NotImplementedError

    @classmethod
    def train_dataset(cls, **kwargs):
        filenames, labels = cls.get_filenames_labels(Split.TRAIN)
        return cls(filenames, labels, **kwargs)

    @classmethod
    def val_dataset(cls, **kwargs):
        filenames, labels = cls.get_filenames_labels(Split.VAL)
        return cls(filenames, labels, **kwargs)

    @classmethod
    def test_dataset(cls, **kwargs):
        filenames, labels = cls.get_filenames_labels(Split.TEST)
        return cls(filenames, labels, **kwargs)


class HaadfDataset(ImageClassificationDataset):
    @staticmethod
    def get_filenames_labels(split: Split) -> Tuple[List[str], List[int]]:
        df = pd.read_csv(HAADF_DATASET)
        split_df = df[df[Columns.SPLIT] == split]
        filenames = (TIF_PATH + os.sep + split_df[Columns.FILENAME]).to_list()
        labels = (split_df[Columns.LABEL]).to_list()
        return filenames, labels


class ImageDataset:
    FILENAME_COL = "Filename"
    SPLIT_COL = "Split"

    def __init__(self, dataset_csv: str):
        self.df = pd.read_csv(dataset_csv)

    def iterate_data(self, split: Split):
        df = self.df[self.df[self.SPLIT_COL] == split]
        for idx, row in df.iterrows():
            image_filename = os.path.join(TIF_PATH, row[self.FILENAME_COL])
            yield image_filename


class CoordinatesDataset:
    FILENAME_COL = "Filename"
    COORDS_COL = "Coords"
    SPLIT_COL = "Split"

    def __init__(self, coord_image_csv: str):
        self.df = pd.read_csv(coord_image_csv)

    def iterate_data(self, split: Split):
        df = self.df[self.df[self.SPLIT_COL] == split]
        for idx, row in df.iterrows():
            image_filename = os.path.join(TIF_PATH, row[self.FILENAME_COL])
            if isinstance(row[self.COORDS_COL], str):
                coords_filename = os.path.join(COORDS_PATH, row[self.COORDS_COL])
            else:
                coords_filename = None
            yield image_filename, coords_filename

    @staticmethod
    def load_coordinates(label_filename: str) -> List[Tuple[int, int]]:
        atom_coordinates = pd.read_csv(label_filename)
        return list(zip(atom_coordinates['X'], atom_coordinates['Y']))

    def split_length(self, split: Split):
        df = self.df[self.df[self.SPLIT_COL] == split]
        return len(df)


class HaadfCoordinates(CoordinatesDataset):
    def __init__(self):
        super().__init__(coord_image_csv=COORDS_DATASET)


class CropsDataset(ImageClassificationDataset):
    @staticmethod
    def get_filenames_labels(split: Split):
        df = pd.read_csv(CROPS_DATASET)
        split_df = df[df[CropsColumns.SPLIT] == split]
        filenames = (CROPS_PATH + os.sep + split_df[CropsColumns.FILENAME]).to_list()
        labels = (split_df[CropsColumns.LABEL]).to_list()
        return filenames, labels


class CropsCustomDataset(ImageClassificationDataset):

    @staticmethod
    def get_filenames_labels(split: Split, crops_dataset: str, crops_path: str):
        df = pd.read_csv(crops_dataset)
        split_df = df[df[CropsColumns.SPLIT] == split]
        filenames = (crops_path + os.sep + split_df[CropsColumns.FILENAME]).to_list()
        labels = (split_df[CropsColumns.LABEL]).to_list()
        return filenames, labels


class ProbsDataset(ImageClassificationDataset):
    @staticmethod
    def get_filenames_labels(split: Split):
        df = pd.read_csv(PROBS_DATASET)
        split_df = df[df[ProbsColumns.SPLIT] == split]
        filenames = (PROBS_PATH + os.sep + split_df[ProbsColumns.FILENAME]).to_list()
        labels = (split_df[ProbsColumns.LABEL]).to_list()
        return filenames, labels


class SlidingCropDataset(Dataset):

    def __init__(self, tif_filename, include_coords=True):
        self.filename = tif_filename
        self.include_coords = include_coords
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.n_labels = 2
        self.step_size = 2
        self.window_size = (21, 21)
        self.loaded_crops = []
        self.loaded_coords = []
        self.load_crops()

    def sliding_window(self, image):
        # slide a window across the image
        for x in range(0, image.shape[0] - self.window_size[0], self.step_size):
            for y in range(0, image.shape[1] - self.window_size[1], self.step_size):
                # yield the current window
                center_x = x + ((self.window_size[0] - 1) // 2)
                center_y = y + ((self.window_size[1] - 1) // 2)
                yield center_x, center_y, image[x:x + self.window_size[0], y:y + self.window_size[1]]

    @staticmethod
    def load_image(img_filename):
        img = Image.open(img_filename)
        np_img = np.asarray(img).astype(np.float32)
        np_bg = median_filter(np_img, size=(40, 40))
        np_clean = np_img - np_bg
        np_normed = (np_clean - np_clean.min()) / (np_clean.max() - np_clean.min())
        return np_normed

    def load_crops(self):
        img = self.load_image(self.filename)
        for x_center, y_center, img_crop in self.sliding_window(img):
            self.loaded_crops.append(img_crop)
            self.loaded_coords.append((x_center, y_center))

    def get_n_labels(self):
        return self.n_labels

    def __len__(self):
        return len(self.loaded_crops)

    def __getitem__(self, idx):
        crop = self.loaded_crops[idx]
        x, y = self.loaded_coords[idx]
        crop = self.transform(crop)

        return crop, x, y


def get_image_path_without_coords(split: str or None = None):
    coords_prefix_set = set()
    for coords_name in os.listdir(COORDS_PATH):
        coord_prefix = coords_name.split('_')[0]
        coords_prefix_set.add(coord_prefix)

    all_prefixes_set = set()
    for tif_name in os.listdir(TIF_PATH):
        coord_prefix = tif_name.split('_')[0]
        all_prefixes_set.add(coord_prefix)

    if split == Split.TRAIN:
        missing_prefixes = coords_prefix_set
    elif split == Split.TEST:
        missing_prefixes = all_prefixes_set - coords_prefix_set
    elif split is None:
        missing_prefixes = all_prefixes_set
    else:
        raise ValueError
    tif_filenames_list = []
    labels_list = []
    for prefix in missing_prefixes:
        filename_matches = glob.glob(os.path.join(TIF_PATH, f'{prefix}_HAADF*NC*'))
        if len(filename_matches) == 0:
            continue
        pos_filenames = [filename for filename in filename_matches if '_PtNC' in filename]
        neg_filenames = [filename for filename in filename_matches if '_NC' in filename]

        if len(pos_filenames) > 0:
            pos_filename = sorted(pos_filenames)[-1]
            tif_filenames_list.append(pos_filename)
            labels_list.append(1)
        if len(neg_filenames) > 0:
            neg_filename = sorted(neg_filenames)[-1]
            tif_filenames_list.append(neg_filename)
            labels_list.append(0)

    return tif_filenames_list, labels_list


if __name__ == "__main__":
    filenames_list = get_image_path_without_coords()
    filename = filenames_list[0]
    dataset = SlidingCropDataset(filename)
