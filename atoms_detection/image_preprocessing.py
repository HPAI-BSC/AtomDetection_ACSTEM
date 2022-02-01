import numpy as np

from scipy.ndimage.filters import gaussian_filter, median_filter


def prepro_image(np_img: np.ndarray) -> np.ndarray:
    np_bg = median_filter(np_img, size=(40, 40))
    np_clean = np_img - np_bg
    np_clean[np_clean < 0] = 0
    np_normed = (np_clean - np_clean.min()) / (np_clean.max() - np_clean.min())
    return np_normed
