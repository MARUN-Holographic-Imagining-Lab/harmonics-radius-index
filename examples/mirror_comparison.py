"""
This script compares the results of the HAT algorithm with the linear
interpolation algorithm when the HAT result is mirrored.
"""
from harmonicsradius.image import Image
from harmonicsradius.sr_analyzer import SRAnalyzer, SRAnalyzerSettings
from harmonicsradius.metrics import StructuralSimilarityIndex, HarmonicsRadius


def mirror_image_in_xy(image):
    """
    Mirror the image in the x and y axis.
    """
    return image[::-1, ::-1, :]


if __name__ == "__main__":
    # Settings
    IMAGE_NO = 5

    # Get file paths.
    IMAGE_HAT = f"datasets/hat_results/img_00{IMAGE_NO}_out.png"
    IMAGE_LINEAR = f"datasets/linear_results/image_{IMAGE_NO}_x2.png"

    # Read images.
    original = Image(
        f"datasets/Set5/image_SRF_2/img_00{IMAGE_NO}_SRF_2_HR.png", name="original")
    hat = Image(IMAGE_HAT, name="hat", preprocess=mirror_image_in_xy)
    linear = Image(IMAGE_LINEAR, name="linear")

    #  Save the image.
    hat.save_image(f"datasets/hat_results/image_{IMAGE_NO}_x2-mirror-xy.png")

    # Create the analyzer.
    analyzer = SRAnalyzer(
        SRAnalyzerSettings(name="Mirror Example")
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
