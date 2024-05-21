"""
This script compares the results of linear and bicubic interpolation
with the high resolution image.
"""

import argparse
from harmonicsradius.metrics import HarmonicsRadius

from harmonicsradius.settings import SRAnalyzerSettings
from harmonicsradius.image import Image
from harmonicsradius.sr_analyzer import SRAnalyzer
from harmonicsradius.preprocessors import (
    linear_upscale,
    bicubic_upscale,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate HRI95 metric for given image pairs."
    )
    parser.add_argument(
        "--true",
        "-t",
        type=str,
        help="Path to the true image",
        required=True,
        dest="true_image",
    )
    parser.add_argument(
        "--low_quality",
        "-l",
        type=str,
        help="Path to the low quality image",
        required=True,
        dest="low_quality_image",
    )
    arguments = parser.parse_args()

    # Read the images.
    true_image = Image(arguments.true_image, name="true_image")
    low_image = Image(arguments.low_quality_image, name="low_image")

    # Find the scale factor by comparing the shapes of the images.
    scale_factor = true_image.get_shape()[0] // low_image.get_shape()[0]

    # Generate the upscaled images.
    linear_image = Image(low_image,
                         name="linear",
                         preprocess=lambda img:
                             linear_upscale(img, scale_factor)
                         )
    bicubic_image = Image(low_image,
                          name="bicubic",
                          preprocess=lambda img:
                              bicubic_upscale(img, scale_factor)
                          )

    #  Create the analyzer instance.
    analyzer = SRAnalyzer(
        SRAnalyzerSettings(name="Linear Bicubic HRI Calculator")
    )
    analyzer.add_metric(HarmonicsRadius())
    analyzer.add_reference_image(true_image)
    analyzer.add_image(linear_image)
    analyzer.add_image(bicubic_image)

    # Calculate the metrics.
    results = analyzer.calculate()

    # Print the results.
    print(f"\n\tImage Name: {arguments.true_image}")
    print(f"\tScale Factor: x{scale_factor}")
    print()
    for result in results:
        print(result)
