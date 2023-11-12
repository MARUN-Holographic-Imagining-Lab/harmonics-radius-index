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
from core.image_preperation_factory import ImagePreperationFactory
from core.image_holder import ImagePossibleTypes
from core.transform_strategies.fourier_transform import FourierTransformStrategy


class SuperResolutionAnalyzer:
    """
    It provides a functionality to analyze the super-resolution
    images' algorithms.
    """
    def __init__(self, original_image_path: str, scale: int):
        self._image_path = original_image_path
        self._scale = scale
        # Generate the images.
        self._image_holder = ImagePreperationFactory.generate_images(
            original_image_path, scale_factor=scale
        )

    def show_image(self, image_type: ImagePossibleTypes | str, domain: str = "spatial") -> None:
        """Show the image on the screen."""
        transform_strategy = None if domain == "spatial" else FourierTransformStrategy()

        # If one image is asked.
        if isinstance(image_type, ImagePossibleTypes):
            self._image_holder.get_image(image_type).show(transform_strategy)

        # If all the images are asked.
        elif isinstance(image_type, str) and image_type == "all":
            self._image_holder.show_images(transform_strategy)
