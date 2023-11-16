"""
The interface for all metrics.
"""

from abc import ABCMeta, abstractmethod
from typing import Any


class MetricResult:
    """Hold the result of a metric."""
    def __init__(self, metric_name: str, metric_value: Any, metric_unit: str):
        self.name = metric_name
        self.value = metric_value
        self.unit = metric_unit

        self.reference_image_name: str = ""
        self.image_name: str = ""

    def __str__(self) -> str:
        """Return the string representation of the metric."""
        return f"{self.name}: {self.value} {self.unit} " \
               f"(ref: {self.reference_image_name}, img: {self.image_name})"

    def to_dict(self) -> dict[str, Any]:
        """Return the dictionary representation of the metric."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "referance": self.reference_image_name,
            "image": self.image_name
        }

    def register_image_names(self, reference_image_name: str, image_name: str) -> None:
        """Register the image names."""
        self.reference_image_name = reference_image_name
        self.image_name = image_name



class InterfaceMetric(metaclass=ABCMeta):
    """The interface for all metrics."""

    @abstractmethod
    def calculate(self, **kwargs) -> MetricResult:
        """
        The function to calculate the metric.

        Returns:
            The calculated metric as MetricResult.
        """

    @property
    @abstractmethod
    def keywords_needed(self) -> dict[str, type]:
        """
        The keywords needed to calculate the metric.

        Returns:
            The keywords needed to calculate the metric.
            {
                "keyword_name_1": keyword_type,
                "keyword_name_2": keyword_type,
                (...)
            }
        """
