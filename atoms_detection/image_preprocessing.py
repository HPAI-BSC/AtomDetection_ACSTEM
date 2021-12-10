import numpy as np

from scipy.ndimage.filters import gaussian_filter, median_filter


def dl_prepro_image(np_img: np.ndarray) -> np.ndarray:
    # np_bg = gaussian_filter(np_img, sigma=20)
    np_bg = median_filter(np_img, size=(40, 40))
    np_clean = np_img - np_bg
    np_clean[np_clean < 0] = 0
    np_normed = (np_clean - np_clean.min()) / (np_clean.max() - np_clean.min())
    # np_normed = (np_img - np_img.min()) / (np_img.max() - np_img.min())
    return np_normed


def cv_prepro_image(img: np.ndarray) -> np.ndarray:
    bg_img = gaussian_filter(img, sigma=10)
    clean_img = img - bg_img
    normed_img = (clean_img - clean_img.min()) / (clean_img.max() - clean_img.min())
    return normed_img
