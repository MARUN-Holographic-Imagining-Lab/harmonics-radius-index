"""
MSE implementation as a metric.
"""
import numpy
import cv2
from matplotlib import pyplot as plt

from core.image import Image
from core.metrics.interface_metric import InterfaceMetric, MetricResult
from core.utils import get_fft_of_image, draw_square_from_center


class HarmonicsRadius(InterfaceMetric):
    """The Harmonics Radius' metric."""

    @property
    def keywords_needed(self) -> dict[str, type]:
        """The keywords needed to calculate the metric."""
        return {"y_true": Image, "y_pred": Image}

    def calculate(self, **kwargs) -> MetricResult:
        """Calculate the Harmonics Radius' metric."""
        # Check keywords.
        if not set(self.keywords_needed).issubset(kwargs):
            raise ValueError("Missing keywords needed to calculate the metric.")

        # Get the parameters.
        y_true = kwargs["y_true"]
        y_pred = kwargs["y_pred"]

        # Check the types of the parameters.
        if not isinstance(y_true, Image):
            raise TypeError("y_true must be an Image.")
        if not isinstance(y_pred, Image):
            raise TypeError("y_pred must be an Image.")

        # Check the shapes of the parameters.
        if y_true.get_shape() != y_pred.get_shape():
            raise ValueError("y_true and y_pred must have the same shape.")

        # Convert images to greyscale.
        fft_of_true: numpy.ndarray = get_fft_of_image(y_true.get_image(), scale_log=True)
        fft_of_pred: numpy.ndarray = get_fft_of_image(y_pred.get_image(), scale_log=True)

        # Find the peak of the FFT.
        peak_mag_true = numpy.max(fft_of_true)

        # Start from a 3x3 grid in the center point of the fft_of_pred.
        # Check the average magnitude of the 3x3 grid.
        # If it is greater than the 75% of the peak_mag_true, then return the radius.
        # Else, increase the grid size by 2 and repeat.
        pred_x_center = fft_of_pred.shape[0] // 2
        pred_y_center = fft_of_pred.shape[1] // 2

        grid_size = fft_of_pred.shape[0] // 2
        while True:
            # Get the grid.
            grid = fft_of_pred[
                pred_x_center - grid_size//2 : pred_x_center + grid_size//2,
                pred_y_center - grid_size//2 : pred_y_center + grid_size//2,
            ]

            # Check the average magnitude of the grid.
            avg_mag = numpy.average(grid)
            if avg_mag > 0.60 * peak_mag_true:
                # The radius is the half of the grid size.
                radius = grid_size // 2

                draw_square_from_center(fft_of_pred, (pred_x_center, pred_y_center), radius)
                return MetricResult(
                    metric_name="harmonics_radius",
                    metric_value=radius,
                    metric_unit="px"
                )

            # Increase the grid size by 2.
            grid_size -= 2

            # A stop condition.
            if grid_size < 5:
                break

        # If the radius is not found, return the maximum radius.
        return MetricResult(
            metric_name="harmonics_radius",
            metric_value=fft_of_pred.shape[0] // 2,
            metric_unit="px"
        )
