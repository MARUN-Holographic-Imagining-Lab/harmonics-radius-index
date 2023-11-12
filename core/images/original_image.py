"""
This class is used to create a original image object.
"""
import numpy
from core.images.base_image import BaseImage


class OriginalImage(BaseImage):
    """
    This class is used to create a original image object.
    """
    def __init__(self, original_image: numpy.ndarray):
        super().__init__(original_image, 1)
        self._original_image = original_image
        self.image = self._original_image

    @staticmethod
    def _preprocess(image: numpy.ndarray, scale_factor: int) -> numpy.ndarray:
        """
        The main algorithm to prepare the new image from original.
        """
        return None
