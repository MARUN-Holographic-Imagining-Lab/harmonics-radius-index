"""
Main application script
"""

from core.metrics import (
    HarmonicsRadius,
    MeanSquaredError,
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
    IMAGE_NUMBER = 5
    SCALE_FACTOR = 2

    TRUE_IMAGE_PATH = f"datasets/Set5/image_SRF_{SCALE_FACTOR}/img_00{IMAGE_NUMBER}_SRF_{SCALE_FACTOR}_HR.png"
    LOW_PATH = f"datasets/Set5/image_SRF_{SCALE_FACTOR}/img_00{IMAGE_NUMBER}_SRF_{SCALE_FACTOR}_LR.png"
    REAL_ESRGAN_PATH = f"datasets/model_results/Real-ESRGAN/img_00{IMAGE_NUMBER}_out.png"
    HAT_PATH = f"datasets/model_results/HAT/img_00{IMAGE_NUMBER}_out.png"

    high_resolution_image   = Image(TRUE_IMAGE_PATH, name="high_resolution")
    low_resolution_image    = Image(LOW_PATH, name="low_resolution")
    zero_order_image        = Image(low_resolution_image, name="neighbour",
                                    preprocess=lambda img: nearest_upscale(img, 2))
    linear_image            = Image(low_resolution_image, name="linear",
                                    preprocess=lambda img: linear_upscale(img, 2))
    bicubic_image           = Image(low_resolution_image, name="bicubic",
                                    preprocess=lambda img: bicubic_upscale(img, 2))
    hat_image               = Image(HAT_PATH, name="HAT")
    real_esrgan_image       = Image(REAL_ESRGAN_PATH, name="Real-ESRGAN")

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
    print("Results:")
    for result in results:
        print(result)
