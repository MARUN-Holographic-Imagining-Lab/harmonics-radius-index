"""
This script compares the results of linear and bicubic interpolation
with the high resolution image.
"""

from harmonicsradius.metrics import (
    HarmonicsRadius,
    MeanSquaredError,
    StructuralSimilarityIndex,
    PeakSignalToNoiseRatio
)

from harmonicsradius.settings import SRAnalyzerSettings
from harmonicsradius.image import Image
from harmonicsradius.sr_analyzer import SRAnalyzer
from harmonicsradius.preprocessors import (
    linear_upscale,
    bicubic_upscale,
    nearest_upscale,
)


if __name__ == "__main__":
    TRUE_IMAGE_PATH = "true_image.png"
    LOW_PATH = "low_image.png"
    REAL_ESRGAN_PATH = "esrgan_image.png"
    HAT_PATH = "hat_image.png"

    high_resolution_image = Image(TRUE_IMAGE_PATH, name="high_resolution")
    low_resolution_image = Image(LOW_PATH, name="low_resolution")
    zero_order_image = Image(low_resolution_image, name="neighbour",
                             preprocess=lambda img: nearest_upscale(img, 2))
    linear_image = Image(low_resolution_image, name="linear",
                         preprocess=lambda img: linear_upscale(img, 2))
    bicubic_image = Image(low_resolution_image, name="bicubic",
                          preprocess=lambda img: bicubic_upscale(img, 2))
    hat_image = Image(HAT_PATH, name="HAT")
    real_esrgan_image = Image(REAL_ESRGAN_PATH, name="Real-ESRGAN")

    analyzer = SRAnalyzer(
        SRAnalyzerSettings(name="SuperResolution Example")
    )

    analyzer.add_metric(MeanSquaredError())
    analyzer.add_metric(PeakSignalToNoiseRatio())
    analyzer.add_metric(StructuralSimilarityIndex())
    analyzer.add_metric(HarmonicsRadius())
    analyzer.add_reference_image(high_resolution_image)
    analyzer.add_image(zero_order_image)
    analyzer.add_image(linear_image)
    analyzer.add_image(bicubic_image)
    analyzer.add_image(real_esrgan_image)
    analyzer.add_image(hat_image)

    # Calculate the metrics.
    results = analyzer.calculate()
    for result in results:
        print(result)
