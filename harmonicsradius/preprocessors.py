"""
This module defines a couple of most used preprocessor functions.
"""

import cv2
from numpy import ndarray


def shrink_to(image: ndarray, width: int, height: int) -> ndarray:
    """
    Shrink the image to the given width and height.

    :param image: The image to be shrinked.
    :param width: The width of the new image.
    :param height: The height of the new image.
    :return: The shrinked image.
    """
    return cv2.resize(image, (width, height),
                      interpolation=cv2.INTER_AREA)  # pylint: disable=no-member


def linear_upscale(image: ndarray, factor: int) -> ndarray:
    """
    Upscale the image using linear interpolation.

    :param image: The image to be upscaled.
    :param factor: The factor to upscale the image.
    :return: The upscaled image.
    """
    return cv2.resize(image, (0, 0), fx=factor, fy=factor,
                      interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member


def bicubic_upscale(image: ndarray, factor: int) -> ndarray:
    """
    Upscale the image using bicubic interpolation.

    :param image: The image to be upscaled.
    :param factor: The factor to upscale the image.
    :return: The upscaled image.
    """
    return cv2.resize(image, (0, 0), fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)  # pylint: disable=no-member


def nearest_upscale(image: ndarray, factor: int) -> ndarray:
    """
    Upscale the image using nearest neighbour interpolation.

    :param image: The image to be upscaled.
    :param factor: The factor to upscale the image.
    :return: The upscaled image.
    """
    return cv2.resize(image, (0, 0), fx=factor, fy=factor,
                      interpolation=cv2.INTER_NEAREST)  # pylint: disable=no-member
