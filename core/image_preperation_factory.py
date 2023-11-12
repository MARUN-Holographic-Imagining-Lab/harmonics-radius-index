"""
The main logic behind the image generation is implemented here.
"""

import cv2

from core.image_holder import ImageHolder, ImagePossibleTypes
from core.images.original_image import OriginalImage
from core.images.small_image import SmallImage
from core.images.zero_order_image import ZeroOrderImage
from core.images.linear_image import LinearImage
from core.images.bicubic_image import BicubicImage
from core.images.super_resolution_image import SuperResolutionImage


class ImagePreperationFactory:
    """It provides a functionality to generate images."""

    @staticmethod
    def generate_images(original_image_path: str, scale_factor: int = 8) -> ImageHolder:
        """
        Generates the images.
        """
        # Read the image.
        original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)

        # Create the image holder.
        holder = ImageHolder()

        # Add the images into the holder.
        holder.add_image(OriginalImage(original_image),
                         ImagePossibleTypes.ORIGINAL)
        small_image = SmallImage(original_image, 64)
        holder.add_image(small_image, ImagePossibleTypes.SMALL)
        holder.add_image(ZeroOrderImage(small_image.get_image(), scale_factor),
                         ImagePossibleTypes.ZERO_ORDER)
        holder.add_image(LinearImage(small_image.get_image(), scale_factor),
                         ImagePossibleTypes.LINEAR)
        holder.add_image(BicubicImage(small_image.get_image(), scale_factor),
                         ImagePossibleTypes.BICUBIC)
        holder.add_image(SuperResolutionImage(original_image_path),
                         ImagePossibleTypes.SUPER_RESOLUTION)

        return holder
