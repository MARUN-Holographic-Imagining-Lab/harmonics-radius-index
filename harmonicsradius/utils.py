"""
Holds the utility functions to be used in the analisys.
"""
import cv2
from numpy import ndarray, fft
from numpy import log as np_log
from numpy import abs as np_abs
from numpy import max as np_max
from numpy import uint8 as np_uint8


def read_image(image_path: str) -> ndarray:
    """Read the image from the path.

    :param image_path: The path of the image.
    :return: The image as a numpy array.
    """
    return cv2.imread(image_path)  # pylint: disable=no-member


def show_image(image: ndarray, title: str = "Image") -> None:
    """Show the image.

    :param image: The image to be shown.
    :param title: The title of the image.
    """
    cv2.imshow(title, image)    # pylint: disable=no-member
    cv2.waitKey(0)              # pylint: disable=no-member
    cv2.destroyAllWindows()     # pylint: disable=no-member


def save_image(image: ndarray, image_path: str) -> None:
    """Save the image to the path.

    :param image: The image to be saved.
    :param image_path: The path of the image.
    """
    cv2.imwrite(image_path, image)  # pylint: disable=no-member


def get_fft_of_image(image: ndarray, scale_log: bool = True) -> ndarray:
    """Get the FFT of the image.

    :param image: The image to get the FFT.
    :param scale_log: If the FFT should be scaled logarithmically.
    :return: The FFT of the image.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY)  # pylint: disable=no-member
    fft_image = fft.fftshift(fft.fft2(image))

    if scale_log:
        fft_image = np_log(1 + np_abs(fft_image))
    else:
        fft_image = np_abs(fft_image)
    return fft_image


def show_fft_image(fft_image: ndarray, title: str = "FFT Image") -> None:
    """Show the FFT image.

    :param fft_image: The FFT image to be shown.
    :param title: The title of the image.
    """
    # Importing here to avoid waiting time for matplotlib load.
    from matplotlib import pyplot as plt

    fft_uint8 = (fft_image / np_max(fft_image) * 255).astype(np_uint8)
    plt.imshow(fft_uint8, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


def draw_square_from_center(image: ndarray,
                            center: tuple[int, int],
                            half_size: int,
                            color: tuple[int, int, int] = (0, 0, 255)) -> None:
    """Draw a rectangle from the center point.

    :param image: The image to draw the rectangle.
    :param center: The center of the rectangle.
    :param half_size: The half size of the rectangle.
    :param color: The color of the rectangle in (red, green, blue).
    """
    # Find the coordinates of the rectangle.
    x_center, y_center = center
    x_start = x_center - half_size
    y_start = y_center - half_size
    x_end = x_center + half_size
    y_end = y_center + half_size

    # Copy the image and convert it to RGB.
    resulted_image = image.copy()
    resulted_image = (resulted_image / np_max(resulted_image)
                      * 255).astype(np_uint8)
    resulted_image_rgb = cv2.cvtColor(
        resulted_image, cv2.COLOR_GRAY2BGR)  # pylint: disable=no-member

    # Draw the rectangle.
    cv2.rectangle(resulted_image_rgb, (y_start, x_start),
                  (y_end, x_end), color, 2)  # pylint: disable=no-member
    show_image(resulted_image_rgb)
