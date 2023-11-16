"""
This module defines a couple of most used preprocessor functions.
"""

import cv2


def shrink_to(image, width, height):
    """
    Shrink the image to the given width and height.
    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)  # pylint: disable=no-member

def linear_upscale(image, factor):
    """
    Upscale the image using linear interpolation.
    """
    return cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)  # pylint: disable=no-member

def bicubic_upscale(image, factor):
    """
    Upscale the image using bicubic interpolation.
    """
    return cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)  # pylint: disable=no-member

def nearest_upscale(image, factor):
    """
    Upscale the image using nearest neighbour interpolation.
    """
    return cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)  # pylint: disable=no-member
