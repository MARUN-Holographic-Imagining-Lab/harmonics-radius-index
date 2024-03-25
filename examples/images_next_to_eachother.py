""
from PIL import Image


def combine_and_save_images(image_path1, image_path2, output_path):
    """Combines two images next to each other and saves the result.

    Args:
        image_path1 (str): First image path.
        image_path2 (str): Second image path.
        output_path (str): Output image path.
    """
    # Open the images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Get the size of the first image
    width1, height1 = image1.size

    # Get the size of the second image
    width2, height2 = image2.size

    # Calculate the total width and height for the new image
    total_width = width1 + width2
    max_height = max(height1, height2)

    # Create a new image with the calculated size
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste the first image onto the new image
    new_image.paste(image1, (0, 0))

    # Paste the second image next to the first image
    new_image.paste(image2, (width1, 0))

    # Save the result to the specified output path
    new_image.save(output_path)


if __name__ == "__main__":
    # Provide the file paths for your two images and the desired output path
    IMAGE_PATH_1 = "outputs/ssim_test_linear.png"
    IMAGE_PATH_2 = "outputs/ssim_test_linear_mirror.png"
    OUTPUT_PATH = "outputs/ssim_test.png"

    combine_and_save_images(IMAGE_PATH_1, IMAGE_PATH_2, OUTPUT_PATH)
