"""
This example demonstrates how to calculate the Fourier Magnitude of an image
and apply thresholding to the Fourier Magnitude.
"""
import numpy as np
import matplotlib.pyplot as plt
from harmonicsradius.image import Image
from harmonicsradius.preprocessors import shrink_to, bicubic_upscale
from harmonicsradius.utils import get_fft_of_image

# Constants
IMAGES: dict[str, str] = {
    "true": "0_5_hr.png",
    "sr": "0_5_sr.png",
}
DOWNSCALE_SIZE: tuple[int] = (64, 64)

if __name__ == "__main__":
    # Read images.
    true_image = Image(IMAGES["true"], name="true_image")
    sr_image = Image(IMAGES["sr"], name="sr_image")

    # Shrink the true image to the downscale size.
    def shrinker(img): return shrink_to(img, *DOWNSCALE_SIZE)
    lr_image = Image(true_image,
                     name="low_resolution",
                     preprocess=shrinker)

    scale_factor = true_image.get_shape()[0] // lr_image.get_shape()[0]
    def scaler(img): return bicubic_upscale(img, scale_factor)
    bicubic_image = Image(lr_image,
                          "bicubic_upscaled",
                          preprocess=scaler)

    # Calculate the Fourier Magnitude of the images.
    true_fft_mag = get_fft_of_image(true_image.get_image())
    bicubic_fft_mag = get_fft_of_image(bicubic_image.get_image())
    sr_fft_mag = get_fft_of_image(sr_image.get_image())

    def fft_scale_uint8(fft): return (fft / np.max(fft) * 255).astype(np.uint8)
    true_fft_mag = fft_scale_uint8(true_fft_mag)
    bicubic_fft_mag = fft_scale_uint8(bicubic_fft_mag)
    sr_fft_mag = fft_scale_uint8(sr_fft_mag)

    # Apply thresholding to the Fourier Magnitude.
    # Only keep the values that are greater than the threshold.
    THREASHOLD_PERCENTAGE = 50

    # Calculate the maximum value of the Fourier Magnitude.
    # Use this value to calculate the threshold where
    # threshold = round(max(fft_image_mag) * THRESHOLD_PERCENTAGE / 100).
    # Change the values under the threshold to 0.
    def apply_threshold(fft_magnitudes):
        max_value = fft_magnitudes.max()
        threshold = round(max_value * THREASHOLD_PERCENTAGE / 100)
        fft_magnitudes[fft_magnitudes < threshold] = 0
        return fft_magnitudes

    true_fft_thres = apply_threshold(true_fft_mag)
    bicubic_fft_thres = apply_threshold(bicubic_fft_mag)
    sr_fft_thres = apply_threshold(sr_fft_mag)

    # Create a 3x2 grid of images using matplotlib.
    fig, axs = plt.subplots(2, 3)

    # Plot the images.
    def bgr_to_rgb(img): return img[:, :, ::-1]
    axs[0, 0].imshow(bgr_to_rgb(true_image.get_image()))
    axs[0, 0].axis("off")
    axs[0, 1].imshow(bgr_to_rgb(bicubic_image.get_image()))
    axs[0, 1].axis("off")
    axs[0, 2].imshow(bgr_to_rgb(sr_image.get_image()))
    axs[0, 2].axis("off")
    axs[1, 0].imshow(true_fft_thres, cmap="gray")
    axs[1, 0].axis("off")
    axs[1, 0].set_title("Ground Truth")
    axs[1, 1].imshow(bicubic_fft_thres, cmap="gray")
    axs[1, 1].set_title("Bicubic")
    axs[1, 1].axis("off")
    axs[1, 2].imshow(sr_fft_thres, cmap="gray")
    axs[1, 2].set_title("Super-Resolution")
    axs[1, 2].axis("off")

    # Save the figure.
    fig.tight_layout()
    plt.savefig("figure.png",
                transparent=True,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.0)
