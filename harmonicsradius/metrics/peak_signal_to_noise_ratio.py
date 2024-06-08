"""
PSNR implementation as a metric.
"""
import numpy
from skimage.metrics import peak_signal_noise_ratio

from harmonicsradius.image import Image
from harmonicsradius.metrics.interface_metric import InterfaceMetric, MetricResult


class PeakSignalToNoiseRatio(InterfaceMetric):
    """The PSNR metric."""

    @property
    def keywords_needed(self) -> dict[str, type]:
        """The keywords needed to calculate the metric.

        :return: The keywords needed.
        """
        return {"y_true": Image, "y_pred": Image}

    def calculate(self, **kwargs) -> MetricResult:
        """Calculate the PSNR metric.

        :param kwargs: The keywords needed to calculate the metric.
        Check the keywords_needed property.
        :return: The PSNR metric in a MetricResult object.
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

        # Calculate the MSE.
        y_true_array: numpy.ndarray = y_true.get_image()
        y_pred_array: numpy.ndarray = y_pred.get_image()
        calculated_psnr = peak_signal_noise_ratio(
            y_true_array,
            y_pred_array,
            data_range=y_true_array.max() - y_true_array.min(),
        )

        return MetricResult(metric_name="PSNR", metric_value=calculated_psnr, metric_unit="dB")
