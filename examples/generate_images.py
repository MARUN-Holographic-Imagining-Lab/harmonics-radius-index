"""
A script to generate HR images from LR image.
"""
from harmonicsradius.image import Image
from harmonicsradius.preprocessors import (
    shrink_to,
    linear_upscale,
    bicubic_upscale,
    nearest_upscale,
)
from harmonicsradius.utils import save_image

if __name__ == "__main__":
    # Add images.
    HR_IMAGE_PATH = "high_res_image.png"

    high_resolution_image = Image(HR_IMAGE_PATH, name="high_resolution")
    low_resolution_image = Image(HR_IMAGE_PATH, name="low_resolution",
                                 preprocess=lambda img: shrink_to(img, 128, 128))
    zero_order_image = Image(low_resolution_image, name="zero_order_upscaled",
                             preprocess=lambda img: nearest_upscale(img, 4))
    linear_image = Image(low_resolution_image, name="linear_upscaled",
                         preprocess=lambda img: linear_upscale(img, 4))
    bicubic_image = Image(low_resolution_image, name="bicubic_upscaled",
                          preprocess=lambda img: bicubic_upscale(img, 4))

    # Get all the images.
    hr_image = high_resolution_image.get_image()
    lr_image = low_resolution_image.get_image()
    zo_image = zero_order_image.get_image()
    li_image = linear_image.get_image()
    bi_image = bicubic_image.get_image()

    # Save the images into a directory with custom names.
    save_image(hr_image, "outputs/hr_image.png")
    save_image(lr_image, "outputs/lr_image.png")
    save_image(zo_image, "outputs/zo_image.png")
    save_image(li_image, "outputs/li_image.png")
    save_image(bi_image, "outputs/bi_image.png")
