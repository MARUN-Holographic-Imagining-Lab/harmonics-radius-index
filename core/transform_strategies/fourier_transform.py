"""
This class is used to store fourier transform startegy.
"""
import cv2
import numpy
from core.transform_strategies.base_transform import BaseTransformStrategy


class FourierTransformStrategy(BaseTransformStrategy):
    """
    This class is used to store base transform startegy.
    """

    @staticmethod
    def apply_transform(image: numpy.ndarray) -> numpy.ndarray:
        """
        Return the transformed image.
        """
        # Convert the image to grayscale since 2D DFT has to be
        # performed on one dimension.
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the optimal DFT size, and pad the image if needed.
        optimal_rows = cv2.getOptimalDFTSize(grayscale_image.shape[0])
        optimal_cols = cv2.getOptimalDFTSize(grayscale_image.shape[1])
        padded_image = cv2.copyMakeBorder(
            src=grayscale_image,
            top=0,
            bottom=optimal_rows - grayscale_image.shape[0],
            left=0,
            right=optimal_cols - grayscale_image.shape[1],
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Create a two-channel image for storing the real and imaginary parts.
        # The first channel will have the real part of the result and the second
        # channel will have the imaginary part of the result.
        planes = [numpy.float32(padded_image), numpy.zeros(padded_image.shape, dtype=numpy.float32)]
        complex_image = cv2.merge(planes)

        # Compute the DFT and save the result ontop.
        cv2.dft(src=complex_image, dst=complex_image)

        # Split the DFT result into two channels, and save them on planes.
        # The first channel will have the real part of the result and the second
        # channel will have the imaginary part of the result.
        cv2.split(complex_image, planes)

        # Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2).
        # Save it to the first channel of the planes.
        cv2.magnitude(x=planes[0], y=planes[1], magnitude=planes[0])
        magnitude_image = planes[0]

        # Switch to logarithmic scale using ln(1 + Mag).
        mat_of_ones = numpy.ones(magnitude_image.shape, dtype=magnitude_image.dtype)
        cv2.add(src1=mat_of_ones, src2=magnitude_image, dst=magnitude_image)
        cv2.log(src=magnitude_image, dst=magnitude_image)

        # Rearrange the quadrants of Fourier image so that the origin is at
        # the image center.
        magnitude_image_rows, magnitude_image_cols = magnitude_image.shape
        magnitude_image = magnitude_image[0:(magnitude_image_rows & -2),
                                            0:(magnitude_image_cols & -2)]
        cx = int(magnitude_image_rows/2)
        cy = int(magnitude_image_cols/2)
        q0 = magnitude_image[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
        q1 = magnitude_image[cx:cx+cx, 0:cy]     # Top-Right
        q2 = magnitude_image[0:cx, cy:cy+cy]     # Bottom-Left
        q3 = magnitude_image[cx:cx+cx, cy:cy+cy] # Bottom-Right
        # Swap quadrants (Top-Left with Bottom-Right)
        tmp = numpy.copy(q0)
        magnitude_image[0:cx, 0:cy] = q3
        magnitude_image[cx:cx + cx, cy:cy + cy] = tmp
        # Swap quadrant (Top-Right with Bottom-Left)
        tmp = numpy.copy(q1)
        magnitude_image[cx:cx + cx, 0:cy] = q2
        magnitude_image[0:cx, cy:cy + cy] = tmp

        # Store the Fourier amplitude.
        return magnitude_image
