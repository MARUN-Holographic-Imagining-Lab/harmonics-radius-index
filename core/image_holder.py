"""
This module contains the image holder class.
"""
from enum import Enum, auto, unique
import cv2
import numpy
from matplotlib import pyplot as plt
from core.images.base_image import BaseImage
from core.transform_strategies.base_transform import BaseTransformStrategy


@unique
class ImagePossibleTypes(Enum):
    """
    This class contains the possible types of the image.
    """
    ORIGINAL = auto()
    SMALL = auto()
    ZERO_ORDER = auto()
    LINEAR = auto()
    BICUBIC = auto()
    SUPER_RESOLUTION = auto()


class IdGenerator:
    """Holds the id of the image in a plot."""
    def __init__(self, col_count: int, row_count: int):
        self._col_id = 0
        self._row_id = 0
        self._col_count = col_count
        self._row_count = row_count

    def get_id(self):
        """Gets the id of the image in subplot."""
        # Check bounds.
        if self._col_id == self._col_count + 1:
            self._col_id = 0
            self._row_id += 1
            if self._row_id == self._row_count + 1:
                self._row_id = 0

        # Get the cursor.
        cursor_col = self._col_id
        cursor_row = self._row_id

        # Update col id.
        self._col_id += 1

        return cursor_row, cursor_col


class ImageHolder:
    """
    It holds the images.
    """
    def __init__(self):
        self._images = {}

    def add_image(self, image: BaseImage, image_type: ImagePossibleTypes):
        """
        Adds the image to the holder.
        """
        self._images[image_type] = image

    def get_image(self, image_type: ImagePossibleTypes) -> BaseImage:
        """
        Returns the image from the holder.
        """
        return self._images[image_type]

    def show_images(self, transform_strategy: BaseTransformStrategy | None = None) -> None:
        """
        Shows the images.
        """
        # Travel through all the images added to the list.
        fig, axis = plt.subplots(2, 3)
        fig.suptitle("Images")

        id_generator = IdGenerator(2, 3)
        for image_type in ImagePossibleTypes:
            # Check if the image fits [0, 255] or [0, 1] range.
            if transform_strategy is None:
                image_show = cv2.cvtColor(self._images[image_type].get_image(), cv2.COLOR_BGR2RGB)
            else:
                image_show = transform_strategy.apply_transform(
                    self._images[image_type].get_image()
                )

            if image_show.max() <= 1.0:
                image_show = (image_show * 255).astype(numpy.uint8)

            row_id, col_id = id_generator.get_id()
            axis[row_id, col_id].imshow(image_show)
            axis[row_id, col_id].set_title(image_type.name)
            axis[row_id, col_id].axis("off")
        plt.show()
        