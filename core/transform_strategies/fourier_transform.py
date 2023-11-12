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
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_rows, image_cols = grayscale_image.shape
        m_rows = cv2.getOptimalDFTSize(image_rows)
        n_cols = cv2.getOptimalDFTSize(image_cols)
        padded_image = cv2.copyMakeBorder(
            src=grayscale_image,
            top=0,
            bottom=m_rows - image_rows,
            left=0,
            right=n_cols - image_cols,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        planes = [numpy.float32(padded_image), numpy.zeros(padded_image.shape, numpy.float32)]
        complex_image = cv2.merge(planes)
        cv2.dft(complex_image, complex_image)
        # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
        cv2.split(complex_image, planes)
        # planes[0] = magnitude
        cv2.magnitude(planes[0], planes[1], planes[0])
        magnitude_image = planes[0]
        # Switch to logarithmic scale
        mat_of_ones = numpy.ones(magnitude_image.shape, dtype=magnitude_image.dtype)
        cv2.add(mat_of_ones, magnitude_image, magnitude_image)
        cv2.log(magnitude_image, magnitude_image)
        # Crop the spectrum
        magnitude_image_rows, magnitude_image_cols = magnitude_image.shape
        # crop the spectrum, if it has an odd number of rows or columns
        magnitude_image = magnitude_image[0:(magnitude_image_rows & -2),
                                            0:(magnitude_image_cols & -2)]
        cx = int(magnitude_image_rows/2)
        cy = int(magnitude_image_cols/2)
        q0 = magnitude_image[0:cx, 0:cy]         # Top-Left - Create a ROI per quadrant
        q1 = magnitude_image[cx:cx+cx, 0:cy]     # Top-Right
        q2 = magnitude_image[0:cx, cy:cy+cy]     # Bottom-Left
        q3 = magnitude_image[cx:cx+cx, cy:cy+cy] # Bottom-Right
        # swap quadrants (Top-Left with Bottom-Right)
        tmp = numpy.copy(q0)
        magnitude_image[0:cx, 0:cy] = q3
        magnitude_image[cx:cx + cx, cy:cy + cy] = tmp
        # swap quadrant (Top-Right with Bottom-Left)
        tmp = numpy.copy(q1)
        magnitude_image[cx:cx + cx, 0:cy] = q2
        magnitude_image[0:cx, cy:cy + cy] = tmp
        # Normalize the magnitude image for the display
        cv2.normalize(magnitude_image, magnitude_image, 0, 255, cv2.NORM_MINMAX)
        # Store the Fourier amplitude.
        return magnitude_image
