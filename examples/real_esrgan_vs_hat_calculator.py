"""
This script compares the results of linear and bicubic interpolation
with the high resolution image.
"""

import argparse
from harmonicsradius.metrics import HarmonicsRadius

from harmonicsradius.settings import SRAnalyzerSettings
from harmonicsradius.image import Image
from harmonicsradius.sr_analyzer import SRAnalyzer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate HRI95 metric for given image pairs."
    )
    parser.add_argument(
        "--image_no",
        "-i",
        type=str,
        required=True,
        dest="image_no",
    )
    arguments = parser.parse_args()

    # Paths for the images.
    hr = f"images/image_SRF_2/img_00{arguments.image_no}_SRF_2_HR.png"
    hat = f"images/HAT/img_00{arguments.image_no}_HAT_SRx2_ImageNet-pretrain.png"
    esrgan = f"images/Real-ESRGAN/img_00{arguments.image_no}_out.png"

    # Read the images.
    true_image = Image(hr, name="true_image")
    hat_image = Image(hat, name="hat_image")
    esrgan_image = Image(esrgan, name="esrgan_image")

    # Find the scale factor by comparing the shapes of the images.
    scale_factor = "2"

    # Create the analyzer instance.
    analyzer = SRAnalyzer(
        SRAnalyzerSettings(name="HAT RealESRGAN HRI Calculator")
    )
    analyzer.add_metric(HarmonicsRadius())
    analyzer.add_reference_image(true_image)
    analyzer.add_image(esrgan_image)
    analyzer.add_image(hat_image)

    # Calculate the metrics.
    results = analyzer.calculate()

    # Print the results.
    print(f"\n\tImage No: {arguments.image_no}")
    print(f"\tScale Factor: x{scale_factor}")
    print()
    for result in results:
        print(result)
