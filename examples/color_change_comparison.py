"""
This script shows how to use the SRAnalyzer class to compare the results of
different super resolution algorithms with high-resolution one color changed.
"""
from harmonicsradius.image import Image
from harmonicsradius.sr_analyzer import SRAnalyzer, SRAnalyzerSettings
from harmonicsradius.metrics import StructuralSimilarityIndex, HarmonicsRadius


def circler_colours(image):
    """
    Change the color of the image.
    """
    temp = image[:, :, 0]
    image[:, :, 0] = image[:, :, 1]
    image[:, :, 1] = image[:, :, 2]
    image[:, :, 2] = temp
    return image


if __name__ == "__main__":
    # Settings
    IMAGE_NO = 5

    # Get file paths.
    ORIGINAL_IMAGE = "original_image.png"
    IMAGE_HAT = "hat_image.png"
    IMAGE_LINEAR = "linear_image.png"

    # Read images.
    original = Image(ORIGINAL_IMAGE, name="original")
    hat = Image(IMAGE_HAT, name="hat", preprocess=circler_colours)
    linear = Image(IMAGE_LINEAR, name="linear")

    #  Save the image.
    hat.save_image("save_here.png")

    # Create the analyzer.
    analyzer = SRAnalyzer(
        SRAnalyzerSettings(name="Color Change Example")
    )

    # Add images.
    analyzer.add_reference_image(original)
    analyzer.add_image(hat)
    analyzer.add_image(linear)

    # Add metrics.
    analyzer.add_metric(HarmonicsRadius())
    analyzer.add_metric(StructuralSimilarityIndex())

    # Analyze.
    results = analyzer.calculate()
    for result in results:
        print(result)
