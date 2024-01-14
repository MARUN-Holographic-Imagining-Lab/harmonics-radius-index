"""
This script generates phase components of two image,
and then compares them using SSIM, MSE and PSNR metrics.
"""
import cv2
import numpy
from matplotlib import pyplot
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

def get_harmonics_phases(image: numpy.ndarray) -> numpy.ndarray:
    """
    Returns phase components of image.
    :param image: Image as ndarray.
    :return: Phase components as ndarray.
    """
    # Get the FFT of the image with shifted to center.
    fft_image = numpy.fft.fftshift(numpy.fft.fft2(image))
    # Get the phases of the FFT image in radians.
    return numpy.angle(fft_image, deg=False)

def show_phases(phases_original: numpy.ndarray,
                phases_generated: numpy.ndarray,
                output_location: str | None = None) -> None:
    """
    Shows the phases.
    :param phases_original: Phases of the original image.
    :param phases_generated: Phases of the generated image.
    """

    # Plot the phases.
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(phases_original, cmap="gray")
    pyplot.title("Original Phases")
    pyplot.axis("off")

    pyplot.subplot(1, 2, 2)
    pyplot.imshow(phases_generated, cmap="gray")
    pyplot.title("Generated Phases")
    pyplot.axis("off")

    # Save the plot.
    pyplot.tight_layout()
    if output_location is not None:
        pyplot.savefig(output_location,
                transparent=True,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.0)
    pyplot.show()


if __name__ == "__main__":
    HIGH_RES_PATH = "datasets/Set5/image_SRF_2/img_005_SRF_2_HR.png"
    LINEAR_RES_PATH = "datasets/linear_results/image_5_x2.png"
    BICUBIC_RES_PATH = "datasets/bicubic_results/image_5_x2.png"
    HAT_RES_PATH = "datasets/hat_results/img_005_out.png"
    BLACKED_RES_PATH = "datasets/hat_results/image_5_x2-random_blacked.png"

    original_image = cv2.imread(HIGH_RES_PATH, cv2.IMREAD_GRAYSCALE)  # pylint: disable=no-member
    bicubic_image = cv2.imread(BICUBIC_RES_PATH, cv2.IMREAD_GRAYSCALE)  # pylint: disable=no-member
    hat_image = cv2.imread(HAT_RES_PATH, cv2.IMREAD_GRAYSCALE)  # pylint: disable=no-member
    blacked_image = cv2.imread(BLACKED_RES_PATH, cv2.IMREAD_GRAYSCALE)  # pylint: disable=no-member

    # Get the phases of images.
    original_pha = get_harmonics_phases(original_image)
    bicubic_pha = get_harmonics_phases(bicubic_image)
    hat_pha = get_harmonics_phases(hat_image)
    blacked_pha = get_harmonics_phases(blacked_image)

    # Show the phases.
    # show_phases(original_pha, bicubic_pha, "phases_org_vs_bicubic.png")
    # show_phases(original_pha, hat_pha, "phases_org_vs_hat.png")

    # Calculate the metrics.
    radians_data_range = original_pha.max() - original_pha.min()
    bicubic_ssim = ssim(original_pha, bicubic_pha, data_range=radians_data_range)
    bicubic_psnr = psnr(original_pha, bicubic_pha, data_range=radians_data_range)

    hat_ssim = ssim(original_pha, hat_pha, data_range=radians_data_range)
    hat_psnr = psnr(original_pha, hat_pha, data_range=radians_data_range)

    blacked_ssim = ssim(original_pha, blacked_pha, data_range=radians_data_range)
    blacked_psnr = psnr(original_pha, blacked_pha, data_range=radians_data_range)

    original_ssim = ssim(original_pha, original_pha, data_range=radians_data_range)
    original_psnr = psnr(original_pha, original_pha, data_range=radians_data_range)

    # Print the metrics.
    print("Bicubic SSIM: ", bicubic_ssim)
    print("HAT SSIM: ", hat_ssim)
    print("Blacked SSIM: ", blacked_ssim)

    print("Original SSIM: ", original_ssim)
