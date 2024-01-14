"""
Main application script
"""

from core.metrics import (
    MeanSquaredError,
    HarmonicsRadius,
    StructuralSimilarityIndex,
    PeakSignalToNoiseRatio
)

from core.settings import SRAnalyzerSettings
from core.image import Image
from core.sr_analyzer import SRAnalyzer
from core.preprocessors import (
    shrink_to,
    linear_upscale,
    bicubic_upscale,
    nearest_upscale,
)


if __name__ == "__main__":
    # Add images.
    TRUE_IMAGE_PATH = "true_image.png"
    ESRGAN_RESULT = "esrgan_result.png"
    HAT_RESULT = "hat_result.png"

    high_resolution_image   = Image(TRUE_IMAGE_PATH, name="high_resolution")
    shapes = high_resolution_image.get_shape()

    low_resolution_image    = Image(TRUE_IMAGE_PATH, name="low_resolution",
                                    preprocess=lambda img: shrink_to(img,
                                                                     shapes[1] // 2,
                                                                     shapes[0] // 2))
    zero_order_image        = Image(low_resolution_image, name="zero_order_upscaled",
                                    preprocess=lambda img: nearest_upscale(img, 2))
    linear_image            = Image(low_resolution_image, name="linear_upscaled",
                                    preprocess=lambda img: linear_upscale(img, 2))
    bicubic_image           = Image(low_resolution_image, name="bicubic_upscaled",
                                    preprocess=lambda img: bicubic_upscale(img, 2))
    hat                     = Image(HAT_RESULT, name="hat")
    esrgan                  = Image(ESRGAN_RESULT, name="esrgan")
    high_res                = Image(TRUE_IMAGE_PATH, name="high_res")

    # Create the analyzer.
    analyzer = SRAnalyzer(
        SRAnalyzerSettings(name="SuperResolution Example")
    )

    # Add metrics.
    analyzer.add_metric(HarmonicsRadius())
    analyzer.add_metric(MeanSquaredError())
    analyzer.add_metric(StructuralSimilarityIndex())
    analyzer.add_metric(PeakSignalToNoiseRatio())

    # Add images.
    analyzer.add_reference_image(high_resolution_image)
    analyzer.add_image(zero_order_image)
    analyzer.add_image(linear_image)
    analyzer.add_image(bicubic_image)
    analyzer.add_image(hat)
    analyzer.add_image(esrgan)
    analyzer.add_image(high_res)

    # Calculate the metrics.
    results = analyzer.calculate()
    for result in results:
        print(result)
