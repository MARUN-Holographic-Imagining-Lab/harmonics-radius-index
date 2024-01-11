"""
Main application script
"""
import time
from core.image import Image
from scripts.harmonic_radius import harmonic_radius

if __name__ == "__main__":
    total_time_consumed: list[float] = []
    for IMAGE_NO in range(1, 6):
        print("Testing image:", IMAGE_NO)

        # Get file paths.
        IMAGE_LINEAR = f"datasets/linear_results/image_{IMAGE_NO}_x2.png"
        IMAGE_NEIGHBOUR = f"datasets/nearest_neighbour_results/image_{IMAGE_NO}_x2.png"
        IMAGE_BICUBIC = f"datasets/bicubic_results/image_{IMAGE_NO}_x2.png"
        IMAGE_HAT = f"datasets/hat_results/img_00{IMAGE_NO}_out.png"
        IMAGE_ESRGAN = f"datasets/real_esrgan_results/img_00{IMAGE_NO}_out.png"

        # Read images.
        original    = Image(f"datasets/Set5/image_SRF_2/img_00{IMAGE_NO}_SRF_2_HR.png", name="original")
        linear      = Image(IMAGE_LINEAR, name="linear")
        neighbour   = Image(IMAGE_NEIGHBOUR, name="neighbour")
        bicubic     = Image(IMAGE_BICUBIC, name="bicubic")
        hat         = Image(IMAGE_HAT, name="hat")
        esrgan      = Image(IMAGE_ESRGAN, name="esrgan")
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
        print(f"\nAverage FPS image {IMAGE_NO}: {1/(sum(time_consumed_image) / len(time_consumed_image))}")

    print(f"Average time consumed for images: {1/(sum(total_time_consumed) / len(total_time_consumed))}")
