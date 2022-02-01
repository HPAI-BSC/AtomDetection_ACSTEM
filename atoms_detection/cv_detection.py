import cv2
import numpy as np

from atoms_detection.image_preprocessing import prepro_image
from atoms_detection.detection import Detection


class CVDetection(Detection):

    @staticmethod
    def get_gaussian_kernel(size=21, mean=0, sigma=0.22, offset=0.0):
        # Initializing value of x-axis and y-axis
        # in the range -1 to 1
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        dst = np.sqrt(x * x + y * y)
        # Calculating Gaussian array
        kernel = np.exp(-((dst - mean) ** 2 / (2.0 * sigma ** 2))) - offset
        return kernel

    def filter_image(self, img_arr: np.ndarray, **kwargs):
        gauss_kernel = self.get_gaussian_kernel(**kwargs)
        max_kernel_value = gauss_kernel.flatten().sum()
        filtered_img = cv2.filter2D(img_arr, -1, gauss_kernel)
        filtered_img /= max_kernel_value
        return filtered_img

    def image_to_pred_map(self, img: np.ndarray) -> np.ndarray:
        prepro_img = prepro_image(img)
        filtered_img = self.filter_image(prepro_img)
        filtered_img = filtered_img.transpose()
        return filtered_img
