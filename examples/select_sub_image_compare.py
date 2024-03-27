"""
1. Compute bicubic interpolation of 2x upscale.
2. Select a subset of HR image.
3. Select 15px left from the previous selected subset of HAT and Bicubic.
3. Compute the metrics comparingly original one.
"""

from core.image import Image
from core.preprocessors import bicubic_upscale
from core.sr_analyzer import SRAnalyzer, SRAnalyzerSettings
from core.metrics import HarmonicsRadius, StructuralSimilarityIndex

# Shift amount
SHIFT_AMOUNT = 15

if __name__ == "__main__":
    # Return the file paths of the images.
    images = {
        "hr": "images/hr.png",
        "hat": "images/hat.png",
        "lr": "images/lr.png"
    }

    # Read the images.
    hr_image = Image(images['hr'], name="hr_image")
    hat_image = Image(images['hat'], name="hat_image")
    lr_image = Image(images['lr'], name="lr_image")
    bicubic_image = Image(lr_image, name="bicubic",
                          preprocess=lambda img: bicubic_upscale(img, 2)
                          )

    # Select a subset of HR image.
    hr_image_subset = Image(hr_image.get_image()[0:120, 0:120],
                            name="hr_image_subset")
    hat_image = Image(hat_image.get_image()[SHIFT_AMOUNT:120+SHIFT_AMOUNT,
                                            SHIFT_AMOUNT:120+SHIFT_AMOUNT],
                      name="hat_image_subset")
    bicubic_image = Image(bicubic_image.get_image()[SHIFT_AMOUNT:120+SHIFT_AMOUNT,
                                                    SHIFT_AMOUNT:120+SHIFT_AMOUNT],
                          name="bicubic_image_subset")

    # Calculate the metrics.
    analyzer = SRAnalyzer(
        SRAnalyzerSettings(name="Check Subimage Shift")
    )

    # Add images.
    analyzer.add_reference_image(hr_image_subset)
    analyzer.add_image(hat_image)
    analyzer.add_image(bicubic_image)

    # Add metrics.
    analyzer.add_metric(HarmonicsRadius())
    analyzer.add_metric(StructuralSimilarityIndex())

    # Analyze.
    results = analyzer.calculate()
    for result in results:
        print(result)
