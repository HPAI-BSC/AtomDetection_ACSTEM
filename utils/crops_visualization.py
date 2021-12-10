import os

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from utils.constants import CropsColumns
from utils.paths import CROPS_DATASET, CROPS_PATH, CROPS_VIS_PATH


if not os.path.exists(CROPS_VIS_PATH):
    os.makedirs(CROPS_VIS_PATH)


dataset_df = pd.read_csv(CROPS_DATASET)
for tif_name in dataset_df[CropsColumns.FILENAME]:
    tif_filename = os.path.join(CROPS_PATH, tif_name)
    img = Image.open(tif_filename)
    img = np.array(img).astype(np.float32)
    img = (img - img.min()) / img.max()
    plt.tight_layout()
    plt.imshow(img)
    vis_name = "{}.jpg".format(os.path.splitext(tif_name)[0])
    vis_filename = os.path.join(CROPS_VIS_PATH, vis_name)
    plt.savefig(vis_filename)
