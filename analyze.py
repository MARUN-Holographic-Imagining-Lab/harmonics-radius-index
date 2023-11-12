'''
This script provides a functionality to analyze the super-resolution
images' algorithms. It uses following steps:
1. Reads the original small image.
2. Uses zero-order interpolation to get the large image.
3. Uses linear interpolation to get the large image.
4. Uses bicubic interpolation to get the large image.
5. Uses the super-resolution algorithm to get the large image.
6. Calculates the PSNR and SSIM for each image. -> A metric for comparison.
7. Calculates the Fourier of the large images found.
8. Gets the amplitude of the harmonics.
9. Plots the amplitudes-harmonics frequency graph on a log-linear scale.
10. Draw a circle from the amplitude magnitude equals to 10^-5.
11. Return the radius of the circle.
'''
from enum import Enum, unique, auto

import cv2
import numpy
import matplotlib.pyplot as plt


@unique
class ImageType(Enum):
    """It defines the type of the image."""
    ORIGINAL_IMAGE = auto()
    SMALL_IMAGE = auto()
    ZERO_ORDER_INTERPOLATION = auto()
    LINEAR_INTERPOLATION = auto()
    BICUBIC_INTERPOLATION = auto()
    SUPER_RESOLUTION = auto()


class ImageGenerator:
    """It provides a functionality to generate images."""
    @staticmethod
    def generate_small_sized_image(image: numpy.ndarray, scale_factor: int) -> numpy.ndarray:
        """It creates a new function from the original image with scale factor."""
        # Get the small image size.
        small_image_size = (image.shape[1] // scale_factor, image.shape[0] // scale_factor)
        # Get the small image using linear interpolation.
        small_image = cv2.resize(image, small_image_size, interpolation=cv2.INTER_LINEAR)
        return small_image

    @staticmethod
    def generate_image_zero_order_interpolation(image: numpy.ndarray, scale_factor: int
                                                ) -> numpy.ndarray:
        """It creates a new function from the original image with scale factor."""
        # Get the large image using zero-order interpolation.
        large_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                 fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        return large_image

    @staticmethod
    def generate_image_linear_interpolation(image: numpy.ndarray, scale_factor: int
                                            ) -> numpy.ndarray:
        """It creates a new function from the original image with scale factor."""
        # Get the large image using linear interpolation.
        large_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                 fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        return large_image

    @staticmethod
    def generate_image_bicubic_interpolation(image: numpy.ndarray, scale_factor: int
                                             ) -> numpy.ndarray:
        """It creates a new function from the original image with scale factor."""
        # Get the large image using bicubic interpolation.
        large_image = cv2.resize(image, (0, 0), fx=scale_factor,
                                 fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        return large_image


class SuperResolutionAnalyzer:
    """
    It provides a functionality to analyze the super-resolution
    images' algorithms.
    """
    def __init__(self, original_image_path: str, scale: int):
        # Read the original RGB image.
        self._image_path = original_image_path
        self._image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
        if self._image is None:
            raise ValueError("The image file path is invalid.")
        # Get the scale factor to be used.
        if scale < 1:
            raise ValueError("The scale must be greater than 1.")
        self._scale = scale

        # Generator functions.
        self._generators = {
            ImageType.SMALL_IMAGE: \
                ImageGenerator.generate_small_sized_image,
            ImageType.ZERO_ORDER_INTERPOLATION: \
                ImageGenerator.generate_image_zero_order_interpolation,
            ImageType.LINEAR_INTERPOLATION: \
                ImageGenerator.generate_image_linear_interpolation,
            ImageType.BICUBIC_INTERPOLATION: \
                ImageGenerator.generate_image_bicubic_interpolation,
        }

        # Stores the images.
        self._images = {
            ImageType.ORIGINAL_IMAGE: self._image,
            ImageType.SMALL_IMAGE: None,
            ImageType.ZERO_ORDER_INTERPOLATION: None,
            ImageType.LINEAR_INTERPOLATION: None,
            ImageType.BICUBIC_INTERPOLATION: None,
            ImageType.SUPER_RESOLUTION: None
        }

        # Stores the Fourier amplitudes.
        self._fourier_amplitudes = {
            ImageType.ORIGINAL_IMAGE: None,
            ImageType.ZERO_ORDER_INTERPOLATION: None,
            ImageType.LINEAR_INTERPOLATION: None,
            ImageType.BICUBIC_INTERPOLATION: None,
            ImageType.SUPER_RESOLUTION: None
        }


    def show_image(self, image_type: ImageType | str, domain: str = "time") -> None:
        """Show the image on the screen."""
        if domain == "time":
            images_dict = self._images
        elif domain == "frequency":
            images_dict = self._fourier_amplitudes

        # If one image is asked.
        if isinstance(image_type, ImageType):
            image = images_dict[image_type]
            if image is None:
                raise RuntimeError("The image is not calculated yet. Call prepare_images() method.")
            # Show the image using matplotlib.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_image)
            plt.show()

        # If all the images are asked.
        elif isinstance(image_type, str) and image_type == "all":
            # Show all the images in one figure.
            fig, axs = plt.subplots(2, 3)
            fig.suptitle("Image Comparison - Scale: " + str(self._scale))
            # Show the original image.
            rgb_image = cv2.cvtColor(images_dict[ImageType.ORIGINAL_IMAGE],
                                     cv2.COLOR_BGR2RGB)
            axs[0, 0].imshow(rgb_image)
            axs[0, 0].set_title("Original image")
            axs[0, 0].axis("off")
            # Show the small image.
            rgb_image = cv2.cvtColor(images_dict[ImageType.SMALL_IMAGE],
                                     cv2.COLOR_BGR2RGB)
            axs[0, 1].imshow(rgb_image)
            axs[0, 1].set_title("Small image")
            axs[0, 1].axis("off")
            # Show the zero-order interpolation image.
            rgb_image = cv2.cvtColor(images_dict[ImageType.ZERO_ORDER_INTERPOLATION],
                                    cv2.COLOR_BGR2RGB)
            axs[0, 2].imshow(rgb_image)
            axs[0, 2].set_title("Zero-order interpolation")
            axs[0, 2].axis("off")
            # Show the linear interpolation image.
            rgb_image = cv2.cvtColor(images_dict[ImageType.LINEAR_INTERPOLATION],
                                     cv2.COLOR_BGR2RGB)
            axs[1, 0].imshow(rgb_image)
            axs[1, 0].set_title("Linear interpolation")
            axs[1, 0].axis("off")
            # Show the bicubic interpolation image.
            rgb_image = cv2.cvtColor(images_dict[ImageType.BICUBIC_INTERPOLATION],
                                     cv2.COLOR_BGR2RGB)
            axs[1, 1].imshow(rgb_image)
            axs[1, 1].set_title("Bicubic interpolation")
            axs[1, 1].axis("off")
            # Show the super-resolution image.
            rgb_image = cv2.cvtColor(images_dict[ImageType.SUPER_RESOLUTION],
                                     cv2.COLOR_BGR2RGB)
            axs[1, 2].imshow(rgb_image)
            axs[1, 2].set_title("Super-resolution")
            axs[1, 2].axis("off")
            # Show the figure.
            plt.show()

    def prepare_images(self) -> None:
        """Prepare the images to be analyzed using generators."""
        # Prepare the small image.
        self._images[ImageType.ORIGINAL_IMAGE] = self._image
        self._images[ImageType.SMALL_IMAGE] = \
            self._generators[ImageType.SMALL_IMAGE](self._image, self._scale)
        # Prepare the large images.
        for name, generator in self._generators.items():
            # Skip the small image.
            if name == ImageType.SMALL_IMAGE:
                continue
            # Prepare the large image using interpolations.
            self._images[name] = generator(self._images[ImageType.SMALL_IMAGE], self._scale)
        # Get the SR image using the file_name_sr.png.
        self._images[ImageType.SUPER_RESOLUTION] = self._get_super_resolution_image()

    def calculate_fourier_amplitudes(self) -> None:
        """This method calculates all the Fourier amplitudes for each image."""
        # Calculate the Fourier amplitudes for each image.
        for name, image in self._images.items():
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Calculate the DFT of the images.
            fourier_image= cv2.dft(numpy.float32(grayscale_image), flags=cv2.DFT_COMPLEX_OUTPUT)

            # Shift the zero-frequency component to the center of the spectrum.
            fourier_shifted = numpy.fft.fftshift(fourier_image)

            # Calculate the magnitude of the Fourier Transform.
            print(cv2.magnitude(fourier_shifted[:,:,0], fourier_shifted[:,:,1]))
            fourier_magnitude = 20 * numpy.log(
                cv2.magnitude(fourier_shifted[:,:,0], fourier_shifted[:,:,1])
            )

            # Scale the magnitude for display
            fourier_normalized = cv2.normalize(fourier_magnitude, None, 0, 255,
                                               cv2.NORM_MINMAX, cv2.CV_8UC1)

            # Store the Fourier amplitude.
            self._fourier_amplitudes[name] = fourier_normalized

    def _get_super_resolution_image(self) -> numpy.ndarray:
        """It creates a new function from the original image with scale factor."""
        # Get the SR image using the file_name_sr.png.
        sr_image_path = self._image_path.replace(".png", "_sr.png")
        sr_image = cv2.imread(sr_image_path, cv2.IMREAD_COLOR)
        if sr_image is None:
            raise ValueError("The SR image file path is invalid.")
        return sr_image
