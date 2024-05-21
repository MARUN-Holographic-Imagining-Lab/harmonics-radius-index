
import matplotlib.pyplot as plt
import numpy
import cv2
from harmonicsradius.image import Image

SHIFT_AMOUNT_PERC = 4
REDUCE_SHIFT_AMOUNT_PERC = 1
IMAGE = "images/image_SRF_2/img_002_SRF_2_HR.png"
NORMAL_ADD = False

# Check if SHIFT_AMOUNT_PERC / REDUCE_SHIFT_AMOUNT_PERC is an odd number.
if (SHIFT_AMOUNT_PERC / REDUCE_SHIFT_AMOUNT_PERC) % 2 != 1 and NORMAL_ADD:
    raise ValueError(
        "The SHIFT_AMOUNT_PERC / REDUCE_SHIFT_AMOUNT_PERC must be an odd "
        "number. OR Disable NORMAL_ADD.")

# Generated images
images_list: list[numpy.ndarray] = []

# Normal image
if NORMAL_ADD:
    shift_amount = int(288 * SHIFT_AMOUNT_PERC / 100)
    hr_image_normal = Image(
        Image(IMAGE, name="hr_image_normal")
        .get_image()[0:288, 0:(288-shift_amount)],
        name="hr_image_normal"
    )
    hr_image_normal_rgb = cv2.cvtColor(
        hr_image_normal.get_image(), cv2.COLOR_BGR2RGB)
    images_list.append(hr_image_normal_rgb)

while SHIFT_AMOUNT_PERC > 0:
    # Shift amount in pixels.
    shift_amount = int(288 * SHIFT_AMOUNT_PERC / 100)

    # Select a subset of HR image.
    hr_image_shifted = Image(
        Image(IMAGE, name="hr_image_shifted")
        .get_image()[0:288, shift_amount:288],
        name="hr_image_shifted"
    )
    hr_image_shifted_rgb = cv2.cvtColor(
        hr_image_shifted.get_image(), cv2.COLOR_BGR2RGB)
    images_list.append(hr_image_shifted_rgb)

    SHIFT_AMOUNT_PERC -= REDUCE_SHIFT_AMOUNT_PERC

# Show images in 2x2 subplot using matplotlib.
fig, axs = plt.subplots(len(images_list)//2, 2)
if NORMAL_ADD:
    axs[0, 0].imshow(images_list[0])
    axs[0, 0].set_title("Normal")
    axs[0, 0].axis("off")

start_position = 0 if not NORMAL_ADD else 1
for i in range(start_position, len(images_list)):
    axs[i//2, i % 2].imshow(images_list[i])
    axs[i//2, i % 2].set_title(f"Shifted {len(images_list)-i}%")
    axs[i//2, i % 2].axis("off")
plt.tight_layout()
fig.tight_layout()
plt.savefig("figure.png",
            transparent=True,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.0)
