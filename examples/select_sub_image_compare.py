"""
1. Compute bicubic interpolation of 2x upscale.
2. Select a subset of HR image.
3. Select 15px left from the previous selected subset of HAT and Bicubic.
3. Compute the metrics comparingly original one.
"""

from core.image import Image
from core.preprocessors import bicubic_upscale
from core.metrics import HarmonicsRadius, StructuralSimilarityIndex

# Shift amount
SHIFT_AMOUNT = 10
HRI_THRESHOLD = 0.95

if __name__ == "__main__":
    # Return the file paths of the images.
    images = {
        "hr": "images/hr.png",
        "hat": "images/hat.png",
        "lr": "images/lr.png"
    }

    # Select a subset of HR image.
    hr_image_subset = Image(
        Image(images['hr'], name="hr_image")
        .get_image()[0:288, 0:(288-SHIFT_AMOUNT)],
        name="hr_image_subset"
    )
    hat_image = Image(
        Image(images['hat'], name="hat_image")
        .get_image()[0:288, SHIFT_AMOUNT:288],
        name="hat_image_subset"
    )
    bicubic_image = Image(
        Image(
            Image(images['lr'], name="lr_image"),
            name="bicubic",
            preprocess=lambda img: bicubic_upscale(img, 2)
        )
        .get_image()[0:288, SHIFT_AMOUNT:288],
        name="bicubic_image_subset")

    # Analyze.
    hri85_hat = HarmonicsRadius()\
        .calculate(y_true=hr_image_subset,
                   y_pred=hat_image,
                   custom_threshold=HRI_THRESHOLD)
    hri85_bicubic = HarmonicsRadius()\
        .calculate(y_true=hr_image_subset,
                   y_pred=bicubic_image,
                   custom_threshold=HRI_THRESHOLD)
    ssim_hat = StructuralSimilarityIndex()\
        .calculate(y_true=hr_image_subset,
                   y_pred=hat_image)
    ssim_bicubic = StructuralSimilarityIndex()\
        .calculate(y_true=hr_image_subset,
                   y_pred=bicubic_image)

    print(f"HRI85(HAT): {hri85_hat.value}")
    print(f"HRI85(Bicubic): {hri85_bicubic.value}")
    print(f"SSIM(HAT): {ssim_hat.value}")
    print(f"SSIM(Bicubic): {ssim_bicubic.value}")

    # Check the differences in percentage.
    hri85_diff = (hri85_hat.value - hri85_bicubic.value) / \
        hri85_hat.value * 100
    ssim_diff = abs(ssim_hat.value - ssim_bicubic.value) / \
        ssim_hat.value * 100

    print(f"HRI85 Diff: {hri85_diff:.2f}%")
    print(f"SSIM Diff: {ssim_diff:.2f}%")
