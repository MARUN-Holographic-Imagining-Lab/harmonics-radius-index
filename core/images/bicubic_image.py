"""
This class is used to create a bicubic interpolation image object.
"""
import cv2
import numpy
from core.images.base_image import BaseImage


class BicubicImage(BaseImage):
    """
    This class is used to create a bicubic image object.
    """

    @staticmethod
    def _preprocess(image: numpy.ndarray, scale_factor: int) -> numpy.ndarray:
        """
        The main algorithm to prepare the new image from original.
        """
        # Get the large image using bicubic interpolation.
        large_image = cv2.resize(
            image,
            (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        return large_image
