"""
The base class for representing an image.
"""

from abc import ABCMeta, abstractstaticmethod
import numpy
import matplotlib.pyplot as plt

from core.transform_strategies.base_transform import BaseTransformStrategy


class BaseImage(metaclass=ABCMeta):
    """
    The base class for representing an image.
    """

    def __init__(self, original_image: numpy.ndarray, scale_factor: int):
        self.original_image = original_image
        self.scale_factor = scale_factor
        self.image = self._preprocess(self.original_image, scale_factor)

    @abstractstaticmethod
    def _preprocess(image: numpy.ndarray, scale_factor: int) -> numpy.ndarray:
        """The main algorithm to prepare the new image from original."""
        raise NotImplementedError

    def get_image(self):
        """
        Return the image.
        """
        return self.image

    def get_original_image(self):
        """
        Return the original image.
        """
        return self.original_image

    def show_image(self, title: str = "Derived image",
                   transform_strategy: BaseTransformStrategy | None = None):
        """
        Show the image.
        """
        if transform_strategy is not None:
            self.show_transform_image(transform_strategy, title)
        else:
            plt.imshow(self.image)
            plt.title(title)
            plt.show()

    def show_original_image(self):
        """
        Show the original image.
        """
        plt.imshow(self.original_image)
        plt.title("Original image")
        plt.show()

    def apply_transform(self, transform_strategy: BaseTransformStrategy) -> numpy.ndarray:
        """
        Return the transform of the image.
        """
        return transform_strategy.apply_transform(self.image)

    def show_transform_image(self, 
                             transform_strategy: BaseTransformStrategy,
                             title: str = "Transformed Image") -> None:
        """
        Show the transformed image.
        """
        transformed_image = self.apply_transform(transform_strategy)
        plt.imshow(transformed_image)
        plt.title(title)
        plt.show()
