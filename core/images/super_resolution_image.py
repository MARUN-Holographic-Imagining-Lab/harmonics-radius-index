"""
This class is used to create a super-resolution image object.
"""
import cv2
import numpy
from core.images.base_image import BaseImage


class SuperResolutionImage(BaseImage):
    """
    This class is used to create a bicubic image object.
    """
    def __init__(self, original_image_path: str):
        self.image = self._preprocess(original_image_path)

    @staticmethod
    def _preprocess(image_path_name: str) -> numpy.ndarray:
        """
        The main algorithm to prepare the new image from original.
        """
        return cv2.imread(f"{image_path_name.split('.')[0]}_sr.png", cv2.IMREAD_COLOR)
        