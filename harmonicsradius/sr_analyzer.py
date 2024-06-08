"""
The main super resolution analyzer class.
"""
from harmonicsradius.metrics.interface_metric import InterfaceMetric, MetricResult
from harmonicsradius.image import Image
from harmonicsradius.settings import SRAnalyzerSettings


class SRAnalyzer:
    """The main super resolution analyzer class."""

    def __init__(self, settings: SRAnalyzerSettings = SRAnalyzerSettings()):
        """Initialize the SRAnalyzer class.

        :param settings: The settings of the analyzer.
        """
        self._settings = settings
        self._metrics: list[InterfaceMetric] = []
        self._reference: Image = None
        self._images: list[Image] = []
        self._is_done = False

    def add_metric(self, metric: InterfaceMetric) -> None:
        """Add a metric to the analyzer.

        :param metric: The metric to be added.
        """
        self._metrics.append(metric)

    def add_reference_image(self, image: Image) -> None:
        """Add a reference image to the analyzer.

        :param image: The reference image.
        """
        self._reference = image

    def add_image(self, image: Image) -> None:
        """Add an image to the analyzer.

        :param image: The image to be added.
        """
        self._images.append(image)

    def calculate(self) -> list[MetricResult]:
        """Calculate the metrics.

        :return: The calculated metrics.
        """
        # Check if the analyzer is done.
        if self._is_done:
            raise RuntimeError("The analyzer is already done.")

        if self._reference is None:
            raise RuntimeError("The reference image is not set.")

        if len(self._images) == 0:
            raise RuntimeError("There is no image to calculate.")

        # Calculate the metrics.
        results: list[MetricResult] = []
        for metric in self._metrics:
            for image in self._images:
                keyword_args = {
                    "y_true": self._reference,
                    "y_pred": image
                }
                metric_result = metric.calculate(**keyword_args)
                metric_result.register_image_names(
                    reference_image_name=self._reference.get_name(),
                    image_name=image.get_name()
                )
                results.append(metric_result)

        # Set true if finished.
        self._is_done = True

        return results
