"""
1. Compute bicubic interpolation of 2x upscale.
2. Select a subset of HR image.
3. Select 15px left from the previous selected subset of hr and Bicubic.
3. Compute the metrics comparingly original one.
"""

from core.image import Image
from core.metrics import HarmonicsRadius, StructuralSimilarityIndex

# Shift amount
SHIFT_AMOUNT_PERC = 4
REDUCE_SHIFT_AMOUNT_PERC = 1
HRI_THRESHOLD = 0.95
IMAGE_SIZE = 288


while SHIFT_AMOUNT_PERC > 0:
    print("####")
    print(f"PIXEL SHIFT AMOUNT: {SHIFT_AMOUNT_PERC}%")

    # Return the file paths of the images.
    images = {
        "hr": "images/hr.png",
    }

    # Shift amount in pixels.
    SHIFT_AMOUNT = int(288 * SHIFT_AMOUNT_PERC / 100)
    print(f"PIXEL SHIFT AMOUNT: {SHIFT_AMOUNT}px")

    # Select a subset of HR image.
    hr_image_shifted = Image(
        Image(images['hr'], name="hr_image_shifted")
        .get_image()[0:288, SHIFT_AMOUNT:288],
        name="hr_image_shifted"
    )
    hr_image_normal = Image(
        Image(images['hr'], name="hr_image_normal")
        .get_image()[0:288, 0:(288-SHIFT_AMOUNT)],
        name="hr_image_normal"
    )

    # Analyze.
    hri85_hr_shifted = HarmonicsRadius()\
        .calculate(y_true=hr_image_normal,
                   y_pred=hr_image_shifted,
                   custom_threshold=HRI_THRESHOLD)
    hri85_hr_normal = HarmonicsRadius()\
        .calculate(y_true=hr_image_normal,
                   y_pred=hr_image_normal,
                   custom_threshold=HRI_THRESHOLD)
    ssim_hr_shifted = StructuralSimilarityIndex()\
        .calculate(y_true=hr_image_normal,
                   y_pred=hr_image_shifted)
    ssim_hr_normal = StructuralSimilarityIndex()\
        .calculate(y_true=hr_image_normal,
                   y_pred=hr_image_normal)

    print(f"HRI{int(HRI_THRESHOLD*100)
                } - HR Shifted: {hri85_hr_shifted.value}")
    print(f"HRI{int(HRI_THRESHOLD*100)} - HR Normal: {hri85_hr_normal.value}")
    print(f"SSIM - HR Shifted: {ssim_hr_shifted.value}")
    print(f"SSIM - HR Normal: {ssim_hr_normal.value}")

    # Check the differences in percentage.
    hri85_diff = abs(hri85_hr_shifted.value - hri85_hr_normal.value) / \
        hri85_hr_normal.value * 100
    ssim_diff = abs(ssim_hr_shifted.value - ssim_hr_normal.value) / \
        ssim_hr_normal.value * 100

    print(f"HRI85 Diff: {hri85_diff:.2f}%")
    print(f"SSIM Diff: {ssim_diff:.2f}%")

    SHIFT_AMOUNT_PERC -= REDUCE_SHIFT_AMOUNT_PERC
