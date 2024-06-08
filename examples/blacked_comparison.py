"""
Main application script
"""
import numpy as np
import random

from harmonicsradius.image import Image
from harmonicsradius.sr_analyzer import SRAnalyzer, SRAnalyzerSettings
from harmonicsradius.metrics import StructuralSimilarityIndex, HarmonicsRadius


def randomly_blacked(image):
    """
    Randomly black out parts of the image.
    """
    BLACK_SIZE = 40
    for _ in range(3):
        x = random.randint(0, image.shape[0] - (1 + BLACK_SIZE))
        y = random.randint(0, image.shape[1] - (1 + BLACK_SIZE))
        image[x:(x + BLACK_SIZE), y:(y + BLACK_SIZE),
              :] = np.zeros((BLACK_SIZE, BLACK_SIZE, 3))
    return image


if __name__ == "__main__":
    # Settings
    IMAGE_NO = 5

    # Get file paths.
    IMAGE_HAT = f"datasets/hat_results/img_00{IMAGE_NO}_out.png"
    IMAGE_LINEAR = f"datasets/linear_results/image_{IMAGE_NO}_x2.png"

    # Read images.
    original = Image(
        f"datasets/Set5/image_SRF_2/img_00{IMAGE_NO}_SRF_2_HR.png", name="original")
    hat = Image(IMAGE_HAT, name="hat", preprocess=randomly_blacked)
    linear = Image(IMAGE_LINEAR, name="linear")

    #  Save the image.
    hat.save_image(
        f"datasets/hat_results/image_{IMAGE_NO}_x2-random_blacked.png")

    # Create the analyzer.
    analyzer = SRAnalyzer(
        SRAnalyzerSettings(name="Randomly Blacked Example")
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
