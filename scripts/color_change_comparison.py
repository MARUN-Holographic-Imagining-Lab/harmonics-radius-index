"""
Main application script
"""
from core.image import Image
from core.sr_analyzer import SRAnalyzer, SRAnalyzerSettings
from core.metrics import StructuralSimilarityIndex, HarmonicsRadius

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
    IMAGE_HAT = f"datasets/hat_results/img_00{IMAGE_NO}_out.png"
    IMAGE_LINEAR = f"datasets/linear_results/image_{IMAGE_NO}_x2.png"

    # Read images.
    original    = Image(f"datasets/Set5/image_SRF_2/img_00{IMAGE_NO}_SRF_2_HR.png", name="original")
    hat         = Image(IMAGE_HAT, name="hat", preprocess=circler_colours)
    linear      = Image(IMAGE_LINEAR, name="linear")

    # Save the image.
    hat.save_image(f"datasets/hat_results/image_{IMAGE_NO}_x2-color_change.png")

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

