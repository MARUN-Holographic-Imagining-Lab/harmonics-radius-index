"""
This script calculates the average frame rate of the harmonic radius function
on 10000 iterations with 5 different images.
"""
import time
from harmonicsradius.image import Image
from scripts.harmonic_radius import harmonic_radius

if __name__ == "__main__":
    total_time_consumed: list[float] = []
    for IMAGE_NO in range(1, 6):
        print("Testing image:", IMAGE_NO)

        # Get file paths.
        IMAGE_ORIGINAL = "original_image.png"
        IMAGE_LINEAR = "linear_image.png"
        IMAGE_NEIGHBOUR = "neighbour_image.png"
        IMAGE_BICUBIC = "bicubic_image.png"
        IMAGE_HAT = "hat_image.png"
        IMAGE_ESRGAN = "esrgan_image.png"

        # Read images.
        original = Image(IMAGE_ORIGINAL, name="original")
        linear = Image(IMAGE_LINEAR, name="linear")
        neighbour = Image(IMAGE_NEIGHBOUR, name="neighbour")
        bicubic = Image(IMAGE_BICUBIC, name="bicubic")
        hat = Image(IMAGE_HAT, name="hat")
        esrgan = Image(IMAGE_ESRGAN, name="esrgan")
        test_images = [linear, neighbour, bicubic, hat, esrgan]

        # Calculate the average time consumed by the harmonic radius function
        # on 10000 iterations with 5 different images.
        time_consumed_image: list[float] = []
        for _ in range(500):
            print(".", end="")
            for image in test_images:
                start = time.time()
                harmonic_radius(original, image)
                end = time.time()
                total_time_consumed.append(end - start)
                time_consumed_image.append(end - start)
        print(f"\nAverage FPS image {IMAGE_NO}:"
              f"{1/(sum(time_consumed_image) / len(time_consumed_image))}")

    print("Average time consumed for images:"
          f"{1/(sum(total_time_consumed) / len(total_time_consumed))}")
