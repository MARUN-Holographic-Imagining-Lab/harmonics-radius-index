"""
This script is used to show the drawback of SSIM.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity

from harmonicsradius.preprocessors import shrink_to, linear_upscale, nearest_upscale

if __name__ == "__main__":

    # Read the image
    image = cv2.imread("datasets/lenna.png", cv2.IMREAD_GRAYSCALE)

    # Shrinking the image to 128x128.
    image_shrink = shrink_to(image, 128, 128)

    # Expand the image to 512x512 with linear.
    image_linear = linear_upscale(image_shrink, 4)
    image_zeroed = nearest_upscale(image_shrink, 4)

    # Save image
    cv2.imwrite("outputs/ssim_test_image_zeroed.png", image_zeroed)

    # Mirror in x and y axis.
    image_linear_mirror = np.flip(image_linear, axis=0)
    image_linear_mirror = np.flip(image_linear_mirror, axis=1)

    # Save image
    cv2.imwrite("outputs/ssim_test_linear_mirror.png", image_linear_mirror)
    cv2.imwrite("outputs/ssim_test_linear.png", image_linear)

    # Calculate the SSIM.
    calculated_zero_ssim = structural_similarity(
        image,
        image_zeroed,
    )

    calculated_linear_ssim = structural_similarity(
        image,
        image_linear,
        data_range=image.max() - image_linear_mirror.min(),
    )
    calculated_linear_ssim_mirror = structural_similarity(
        image,
        image_linear_mirror,
        data_range=image.max() - image_linear_mirror.min(),
    )

    print("SSIM of nearest image: ", calculated_zero_ssim)
    print("SSIM of linear image: ", calculated_linear_ssim)
    print("SSIM of linear image mirror: ", calculated_linear_ssim_mirror)
