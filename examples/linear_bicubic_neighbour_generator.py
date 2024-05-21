"""
This script generates the linear, bicubic and nearest neighbour upscaled images
from the low resolution images.
"""

from harmonicsradius.image import Image
from harmonicsradius.preprocessors import (
    linear_upscale,
    bicubic_upscale,
    nearest_upscale,
)


if __name__ == "__main__":

    SCALE_FACTOR = 2

    for IMAGE_NO in range(1, 6):
        LOW_RES_IMAGE = "low_image.png"

        low_resolution_image = Image(LOW_RES_IMAGE, name="low_resolution")
        zero_order_image = Image(low_resolution_image, name="nearest",
                                 preprocess=lambda img: nearest_upscale(img, SCALE_FACTOR))
        linear_image = Image(low_resolution_image, name="linear",
                             preprocess=lambda img: linear_upscale(img, SCALE_FACTOR))
        bicubic_image = Image(low_resolution_image, name="bicubic",
                              preprocess=lambda img: bicubic_upscale(img, SCALE_FACTOR))

        # Save images.
        zero_order_image.save_image("zero_order.png")
        linear_image.save_image("linear_image.png")
        bicubic_image.save_image("bicubic_image.png")
