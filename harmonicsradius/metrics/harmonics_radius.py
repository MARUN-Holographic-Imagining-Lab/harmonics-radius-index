"""
Harmonics Radius implementation with SSIM as a metric.
"""
import numpy
from skimage.metrics import structural_similarity

from harmonicsradius.image import Image
from harmonicsradius.metrics.interface_metric import InterfaceMetric, MetricResult
from harmonicsradius.utils import get_fft_of_image
#  from harmonicsradius.utils import draw_square_from_center


class HarmonicsRadius(InterfaceMetric):
    """The HRI95 metric."""
    THRESHOLD: float = 0.95
    METRIC_NAME: str = f"harmonics_radius_{int(THRESHOLD*100)}"
    METRIC_UNIT: str = "%"

    @property
    def keywords_needed(self) -> dict[str, type]:
        """The keywords needed to calculate the metric.

        :return: The keywords needed.
        """
        return {"y_true": Image, "y_pred": Image}

    def calculate(self, **kwargs) -> MetricResult:
        """Calculate the HRI95.

        :param kwargs: The keywords needed to calculate the metric.
        Check the keywords_needed property.

        :return: The HRI95 in a MetricResult object.
        """
        # Check keywords.
        if not set(self.keywords_needed).issubset(kwargs):
            raise ValueError(
                "Missing keywords needed to calculate the metric.")

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
        fft_of_true: numpy.ndarray = get_fft_of_image(
            y_true.get_image(), scale_log=True)
        fft_of_pred: numpy.ndarray = get_fft_of_image(
            y_pred.get_image(), scale_log=True)

        # Find the center point.
        pred_x_center = fft_of_pred.shape[0] // 2
        pred_y_center = fft_of_pred.shape[1] // 2

        # Check the smallest side of the image.
        grid_size = self.get_smallest_side_size(fft_of_pred)

        while True:
            # Get the grid.
            grid_pred = fft_of_pred[
                pred_x_center - grid_size//2: pred_x_center + grid_size//2,
                pred_y_center - grid_size//2: pred_y_center + grid_size//2,
            ]
            grid_true = fft_of_true[
                pred_x_center - grid_size//2: pred_x_center + grid_size//2,
                pred_y_center - grid_size//2: pred_y_center + grid_size//2,
            ]

            # Check the SSIM of the grid.
            ssim_result = structural_similarity(
                grid_true,
                grid_pred,
                data_range=fft_of_true.max() - fft_of_true.min(),
                multichannel=False,
            )

            # draw_square_from_center(
            #     fft_of_pred,
            #     (pred_x_center,
            #      pred_y_center),
            #     grid_size//2,
            # )
            if ssim_result > self.THRESHOLD:
                # The radius is the half of the grid size.
                found_radius: int = grid_size / 2
                image_radius: int = self.get_smallest_side_size(
                    fft_of_pred) / 2
                metric_value = (found_radius / image_radius) * 100

                return MetricResult(
                    metric_name=self.METRIC_NAME,
                    metric_value=metric_value,
                    metric_unit=self.METRIC_UNIT
                )

            # Increase the grid size by 2.
            grid_size -= 2

            # A stop condition.
            if grid_size < 7:
                break

        # If the radius is not found, return the maximum radius.
        return MetricResult(
            metric_name=self.METRIC_NAME,
            metric_value=0,
            metric_unit=self.METRIC_UNIT
        )

    @staticmethod
    def get_smallest_side_size(image: numpy.ndarray) -> int:
        """Get the smallest side size of the image.

        :param image: The image.

        :return: The smallest side size of the image.
        """
        return image.shape[0] \
            if image.shape[0] < image.shape[1] \
            else image.shape[1]
