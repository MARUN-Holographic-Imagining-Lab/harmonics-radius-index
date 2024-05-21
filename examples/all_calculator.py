"""
This script compares the results of linear and bicubic interpolation
with the high resolution image.
"""

import argparse
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
    nearest_upscale,
    linear_upscale,
    bicubic_upscale,
)

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
    parser.add_argument(
        "--hri95",
        "-hr",
        action="store_true",
        dest="metric_hri",
    )
    parser.add_argument(
        "--mse",
        "-m",
        action="store_true",
        dest="metric_mse",
    )
    parser.add_argument(
        "--ssim",
        "-s",
        action="store_true",
        dest="metric_ssim",
    )
    parser.add_argument(
        "--psnr",
        "-p",
        action="store_true",
        dest="metric_psnr",
    )
    arguments = parser.parse_args()

    # Paths for the images.
    hr = f"images/image_SRF_2/img_00{arguments.image_no}_SRF_2_HR.png"
    lr = f"images/image_SRF_2/img_00{arguments.image_no}_SRF_2_LR.png"
    hat = \
        f"images/HAT/img_00{arguments.image_no}_HAT_SRx2_ImageNet-pretrain.png"
    esrgan = f"images/Real-ESRGAN/img_00{arguments.image_no}_out.png"

    # Read the images.
    true_image = Image(hr, name="true_image")
    low_image = Image(lr, name="low_image")

    # Find the scale factor by comparing the shapes of the images.
    scale_factor = true_image.get_shape()[0] // low_image.get_shape()[0]

    #  Create the images.
    hat_image = Image(hat, name="hat_image")
    esrgan_image = Image(esrgan, name="esrgan_image")
    zero_image = Image(low_image,
                       name="neighbour",
                       preprocess=lambda img:
                       nearest_upscale(img, scale_factor)
                       )
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
    analyzer = SRAnalyzer(SRAnalyzerSettings(
        name="All Calculator"
    ))
    analyzer.add_reference_image(true_image)
    analyzer.add_image(true_image)
    analyzer.add_image(zero_image)
    analyzer.add_image(linear_image)
    analyzer.add_image(bicubic_image)
    analyzer.add_image(esrgan_image)
    analyzer.add_image(hat_image)

    # Add the metrics.
    if arguments.metric_mse:
        analyzer.add_metric(MeanSquaredError())
    if arguments.metric_psnr:
        analyzer.add_metric(PeakSignalToNoiseRatio())
    if arguments.metric_ssim:
        analyzer.add_metric(StructuralSimilarityIndex())
    if arguments.metric_hri:
        analyzer.add_metric(HarmonicsRadius())

    # Calculate the metrics.
    results = analyzer.calculate()

    # Print the results.
    print(f"\n\tImage Name: {arguments.image_no}")
    print(f"\tScale Factor: x{scale_factor}")
    print()
    for result in results:
        print(result)
