"""
Main application script
"""
from core.settings import SRAnalyzerSettings
from core.image import Image
from core.sr_analyzer import SRAnalyzer
from core.preprocessors import (
    linear_upscale,
    bicubic_upscale,
    nearest_upscale,
)


if __name__ == "__main__":

    SCALE_FACTOR = 2

    for IMAGE_NO in range(1, 6):
        LOW_RES_IMAGE = f"datasets/Set5/image_SRF_{SCALE_FACTOR}/img_00{IMAGE_NO}_SRF_{SCALE_FACTOR}_LR.png"

        low_resolution_image    = Image(LOW_RES_IMAGE, name="low_resolution")
        zero_order_image        = Image(low_resolution_image, name="nearest",
                                        preprocess=lambda img: nearest_upscale(img, SCALE_FACTOR))
        linear_image            = Image(low_resolution_image, name="linear",
                                        preprocess=lambda img: linear_upscale(img, SCALE_FACTOR))
        bicubic_image           = Image(low_resolution_image, name="bicubic",
                                        preprocess=lambda img: bicubic_upscale(img, SCALE_FACTOR))

        # Save images.
        zero_order_image.save_image(f"datasets/nearest_neighbour_results/image_{IMAGE_NO}_x{SCALE_FACTOR}.png")
        linear_image.save_image(f"datasets/linear_results/image_{IMAGE_NO}_x{SCALE_FACTOR}.png")
        bicubic_image.save_image(f"datasets/bicubic_results/image_{IMAGE_NO}_x{SCALE_FACTOR}.png")
