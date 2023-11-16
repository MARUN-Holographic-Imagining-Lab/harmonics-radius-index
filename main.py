"""
Main application script
"""

from core.metrics import MeanSquaredError, HarmonicsRadius
from core.settings import SRAnalyzerSettings
from core.image import Image
from core.sr_analyzer import SRAnalyzer
from core.preprocessors import shrink_to, linear_upscale, bicubic_upscale, nearest_upscale


if __name__ == "__main__":
    # Add images.
    HR_IMAGE_PATH = "test_images/gokhan_hr.png"
    SR_IMAGE_PATH = "test_images/gokhan_sr.png"
    high_resolution_image   = Image(HR_IMAGE_PATH, name="high_resolution")
    low_resolution_image    = Image(HR_IMAGE_PATH, name="low_resolution",
                                    preprocess=lambda img: shrink_to(img, 64, 64))
    zero_order_image        = Image(low_resolution_image, name="zero_order_upscaled",
                                    preprocess=lambda img: nearest_upscale(img, 8))
    linear_image            = Image(low_resolution_image, name="linear_upscaled",
                                    preprocess=lambda img: linear_upscale(img, 8))
    bicubic_image           = Image(low_resolution_image, name="bicubic_upscaled",
                                    preprocess=lambda img: bicubic_upscale(img, 8))
    super_image             = Image(SR_IMAGE_PATH, name="super_upscaled")

    # Create the analyzer.
    analyzer = SRAnalyzer(SRAnalyzerSettings(name="MSE_HR_Calculator"))

    # Add metrics and images.
    analyzer.add_metric(MeanSquaredError())
    analyzer.add_metric(HarmonicsRadius())
    analyzer.add_reference_image(high_resolution_image)
    analyzer.add_image(zero_order_image)
    analyzer.add_image(linear_image)
    analyzer.add_image(bicubic_image)
    analyzer.add_image(super_image)

    # Calculate the metrics.
    results = analyzer.calculate()
    for result in results:
        print(result)
