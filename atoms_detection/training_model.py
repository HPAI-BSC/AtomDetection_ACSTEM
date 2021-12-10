import os
import argparse
import random

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from utils.paths import MODELS_PATH, CROPS_PATH, CROPS_DATASET
from utils.constants import ModelArgs, Split, CropsColumns
from atoms_detection.training import train_epoch, val_epoch
from atoms_detection.dataset import ImageClassificationDataset
from atoms_detection.model import BasicCNN


torch.manual_seed(777)
random.seed(777)
np.random.seed(777)


def get_basic_cnn(*args, **kwargs):
    model = BasicCNN(*args, **kwargs)
    return model


def get_resnet(*args, **kwargs):
    model = resnet18(*args, **kwargs)
    model.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    return model


model_pipeline = {
    ModelArgs.BASICCNN: get_basic_cnn,
    ModelArgs.RESNET18: get_resnet
}

epochs_pipeline = {
    ModelArgs.BASICCNN: 12,
    ModelArgs.RESNET18: 3
}


def train_model(model_arg: ModelArgs, crops_dataset: str, crops_path: str, ckpt_filename: str):

    class CropsDataset(ImageClassificationDataset):
        @staticmethod
        def get_filenames_labels(split: Split):
            df = pd.read_csv(crops_dataset)
            split_df = df[df[CropsColumns.SPLIT] == split]
            filenames = (crops_path + os.sep + split_df[CropsColumns.FILENAME]).to_list()
            labels = (split_df[CropsColumns.LABEL]).to_list()
            return filenames, labels


    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset = CropsDataset.train_dataset()
    val_dataset = CropsDataset.val_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64)
    model = model_pipeline[model_arg](num_classes=train_dataset.get_n_labels()).to(device)

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    loss_function = torch.nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)

    epoch = 0
    for epoch in range(epochs_pipeline[model_arg]):
        train_epoch(train_dataloader, model, loss_function, optimizer, device, epoch)
        val_epoch(val_dataloader, model, loss_function, device, epoch)

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, ckpt_filename)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "model",
        type=ModelArgs,
        help="model architecture",
        choices=list(ModelArgs)
    )
    return parser.parse_args()


if __name__ == "__main__":
    extension_name = "replicate"
    ckpt_filename = os.path.join(MODELS_PATH, "basic_replicate2.ckpt")
    crops_folder = CROPS_PATH + f"_{extension_name}"
    train_model(ModelArgs.BASICCNN, CROPS_DATASET, CROPS_PATH, ckpt_filename)
