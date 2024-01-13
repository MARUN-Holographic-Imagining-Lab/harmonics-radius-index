"""
This script generates the FFT of an image 
and plots the spatial domain image, FFT
magnitudes and FFT phases in log domain.
"""
import cv2
import numpy
from matplotlib import pyplot as plt


if __name__ == "__main__":
    IMAGE_PATH = "image.png"

    # Read the image.
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)  # pylint: disable=no-member
    # Show the image.
    cv2.imshow("Image", image)

    # Get the FFT of the image.
    fft_image = numpy.fft.fftshift(numpy.fft.fft2(image))

    # Get the magnitudes and phases of the FFT image.
    fft_mags = numpy.log(1 + numpy.abs(fft_image))
    fft_phases = numpy.angle(fft_image)

    # Apply hamming window to the FFT magnitudes.
    fft_pha_lp = fft_phases.copy()
    fft_pha_lp = fft_pha_lp * numpy.hamming(fft_pha_lp.shape[0])

    # Get the inverse FFT of the image.
    combined_new = numpy.exp(fft_mags) * numpy.exp(1j * fft_pha_lp)
    ifft_image = numpy.fft.ifft2(numpy.fft.ifftshift(combined_new)).real

    # Plot the images.
    plt.subplot(2, 2, 3)
    plt.imshow(image, cmap="gray")
    plt.title("Spatial Domain")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(ifft_image, cmap="gray")
    plt.title("Spatial Domain After Filtering")
    plt.axis("off")

    plt.subplot(2, 2, 1)
    plt.imshow(fft_phases, cmap="gray")
    plt.title("Phases")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(fft_pha_lp, cmap="gray")
    plt.title("Phases After Filtering")
    plt.axis("off")

    # Save the plot.
    plt.tight_layout()
    plt.savefig("save_here.png",
                transparent=True,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.0)
