"""
This class is used to store base transform startegy.
"""
from abc import ABCMeta, abstractmethod
import numpy


class BaseTransformStrategy(metaclass=ABCMeta):
    """
    This class is used to store base transform startegy.
    """

    @staticmethod
    @abstractmethod
    def apply_transform(image: numpy.ndarray) -> numpy.ndarray:
        """
        Return the transformed image.
        """
        raise NotImplementedError
