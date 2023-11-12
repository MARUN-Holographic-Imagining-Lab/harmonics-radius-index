"""
This class is used to create a small image object.
"""
import cv2
import numpy
from core.images.base_image import BaseImage


class SmallImage(BaseImage):
    """
    This class is used to create a small image object.
    """
    def __init__(self, original_image: numpy.ndarray, scale_factor: int):
        super().__init__(original_image, scale_factor)
        self._small_image = self._preprocess(original_image, scale_factor)
        self.image = self._small_image

    @staticmethod
    def _preprocess(image: numpy.ndarray, scale_factor: int) -> numpy.ndarray:
        """
        The main algorithm to prepare the new image from original.
        """
        return SmallImage._shrink_to(image, scale_factor, scale_factor)

    @staticmethod
    def _shrink_to(image: numpy.ndarray, new_size_x: int, new_size_y: int):
        """
        Shrinks the image to the new size.
        """
        # Get the small image size.
        small_image_size = (
            new_size_x,
            new_size_y,
        )
        # Get the small image using linear interpolation.
        small_image = cv2.resize(
            image, small_image_size, interpolation=cv2.INTER_LINEAR
        )
        return small_image
