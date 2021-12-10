import os

import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from atoms_detection.training_model import model_pipeline, get_args
from atoms_detection.dataset import CropsDataset
from atoms_detection.training import test_epoch
from utils.cf_matrix import make_confusion_matrix
from utils.paths import MODELS_PATH, CM_VIS_PATH


def main(args):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_dataset = CropsDataset.test_dataset()
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    ckpt_filename = os.path.join(MODELS_PATH, f'{args.experiment_name}.ckpt')
    checkpoint = torch.load(ckpt_filename, map_location=device)

    model = model_pipeline[args.model](num_classes=test_dataset.get_n_labels()).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    loss_function = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    y_true, y_pred = test_epoch(test_dataloader, model, loss_function, device)

    cm = confusion_matrix(y_true, y_pred)
    labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
    make_confusion_matrix(cm, group_names=labels, cbar_range=(0, 110))
    if not os.path.exists(CM_VIS_PATH):
        os.makedirs(CM_VIS_PATH)
    plt.savefig(os.path.join(CM_VIS_PATH, f"cm_{args.experiment_name}.jpg"))
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    with open(os.path.join(CM_VIS_PATH, f"metrics_{args.experiment_name}.txt"), 'w') as _log:
        _log.write(f"F1_score: {f1}\nACCURACY: {acc}\n")
    print(f"F1_score: {f1}")
    print(f"ACCURACY: {acc}")


if __name__ == "__main__":
    main(get_args())
