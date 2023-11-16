"""
Holds the utility functions to be used in the analisys.
"""
import cv2
import numpy
from matplotlib import pyplot as plt

def read_image(image_path: str) -> numpy.ndarray:
    """Read the image from the path."""
    return cv2.imread(image_path)


def show_image(image: numpy.ndarray, title: str = "Image") -> None:
    """Show the image."""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_fft_of_image(image: numpy.ndarray, scale_log: bool = True) -> numpy.ndarray:
    """Get the FFT of the image."""
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
    fft_image = numpy.fft.fftshift(numpy.fft.fft2(grey_image))

    if scale_log:
        fft_image = numpy.log(1 + numpy.abs(fft_image))
    return fft_image

def show_fft_image(fft_image: numpy.ndarray, title: str = "FFT Image") -> None:
    """Show the FFT image."""
    fft_uint8 = (fft_image / numpy.max(fft_image) * 255).astype(numpy.uint8)
    plt.imshow(fft_uint8, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def draw_square_from_center(image: numpy.ndarray, center: tuple[int, int], half_size: int,
                               color: tuple[int, int, int] = (0, 0, 255)) -> None:
    """Draw a rectangle from the center point."""
    # Find the coordinates of the rectangle.
    x_center, y_center = center
    x_start = x_center - half_size
    y_start = y_center - half_size
    x_end = x_center + half_size
    y_end = y_center + half_size

    # Copy the image and convert it to RGB.
    resulted_image = image.copy()
    resulted_image = (resulted_image / numpy.max(resulted_image) * 255).astype(numpy.uint8)
    resulted_image_rgb = cv2.cvtColor(resulted_image, cv2.COLOR_GRAY2BGR)  # pylint: disable=no-member

    # Draw the rectangle.
    cv2.rectangle(resulted_image_rgb, (x_start, y_start), (x_end, y_end), color, 2)  # pylint: disable=no-member
    # Now draw a circle in the center with radius half_size.
    cv2.circle(resulted_image_rgb, center, half_size, color, 2)  # pylint: disable=no-member
    show_image(resulted_image_rgb)
