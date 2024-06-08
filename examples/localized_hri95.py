"""
This script find the average HRI95
sharmonicsradius of the given images.
"""

from dataclasses import dataclass

import cv2
from numpy import ndarray
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from harmonicsradius.utils import get_fft_of_image


@dataclass
class SubImagePair:
    """This class represents a pair of sub images."""
    original: ndarray
    generated: ndarray
    location_width_start: int
    location_height_start: int
    location_width_end: int
    location_height_end: int
    result_hri95: int = None
    result_psnr: float = None
    result_ssim: float = None
    result_mse: float = None


def get_sub_image(image: ndarray,
                  location: tuple[int, int],
                  size: tuple[int, int]) -> ndarray:
    """
    Returns the sub image at the given location.
    :param image: Image.
    :param location: Sub image location.
    :param size: Sub image size.
    :return: Sub image.
    """
    return image[location[0]:location[0] + size[0],
                 location[1]:location[1] + size[1]]


if __name__ == "__main__":
    # Paths
    HIGH_RES_PATH = "images/hr.png"
    GENERATED_PATH = "images/hat.png"
    high_res_image = cv2.imread(HIGH_RES_PATH, cv2.IMREAD_GRAYSCALE)
    hat_image = cv2.imread(GENERATED_PATH, cv2.IMREAD_GRAYSCALE)

    #  Settings
    SUB_IMAGE_SIZE = {
        "width": high_res_image.shape[0]//4,
        "height": high_res_image.shape[1]//4}
    SLIDE_SIZE = {
        "width": high_res_image.shape[0]//8,
        "height": high_res_image.shape[1]//8}
    HRI_SUCCESS_THRES = 0.95
    SHOW_SUB_IMAGE_LOCATIONS = True
    SHOW_SUB_IMAGE_METRICS = True
    SAVE_METRICS_TO_CSV = True
    SAVE_VIDEO_SUB_IMAGE_METRICS = True
    CALCULATE_AVERAGE_METRICS = False
    RESIZE_FACTOR = 4

    print(f"Settings: \n\t-{SUB_IMAGE_SIZE=},\n"
          f"\t-{SLIDE_SIZE=},\n"
          f"\t-{HRI_SUCCESS_THRES=},\n"
          f"\t-{SHOW_SUB_IMAGE_LOCATIONS=},\n"
          f"\t-{SHOW_SUB_IMAGE_METRICS=},\n"
          f"\t-{SAVE_METRICS_TO_CSV=},\n"
          f"\t-{SAVE_VIDEO_SUB_IMAGE_METRICS=},\n"
          f"\t-{CALCULATE_AVERAGE_METRICS=},\n",
          f"\t-{RESIZE_FACTOR=}\n")

    # Read the images and calculate their Fourier mags.
    hr_fft = get_fft_of_image(high_res_image, scale_log=True)
    hat_fft = get_fft_of_image(hat_image, scale_log=True)

    # Get the sub image size.
    width, height = high_res_image.shape
    sub_width, sub_height = SUB_IMAGE_SIZE["width"], SUB_IMAGE_SIZE["height"]
    print(f"Image size: {width=}, {height=}")

    # Check if sub image size is valid.
    if sub_width > width or sub_height > height:
        raise ValueError(
            "Sub image size cannot be greater than the image size.")

    #  Check if image sizes can be divided by sub image size.
    if width % sub_width != 0 or height % sub_height != 0:
        raise ValueError("Image sizes must be divided by sub image size.")

    # Get the sub image locations as sliding window with sub_iamge size.
    print("Calculating sub image locations...")
    sub_image_locations = []
    for width_index in range(0,
                             width-SUB_IMAGE_SIZE["width"]+1,
                             SLIDE_SIZE["width"]):
        for height_index in range(0,
                                  height-SUB_IMAGE_SIZE["height"]+1,
                                  SLIDE_SIZE["height"]):
            sub_image_locations.append({
                "start_width": width_index,
                "start_height": height_index,
                "end_width": width_index + sub_width,
                "end_height": height_index + sub_height
            })

            # Check if the sub image locations are valid.
            if width_index + sub_width > width:
                print(f"ERROR: {width_index + sub_width=} > {width=}")
            if height_index + sub_height > height:
                print(f"ERROR: {height_index + sub_height=} > {height=}")

            if SHOW_SUB_IMAGE_LOCATIONS:
                # Show the sub image locations as red squares in the FFT image.
                print("Drawing sub image locations...")
                fft_uint8 = (hr_fft / hr_fft.max() * 255).astype("uint8")
                # Add +15 white pixels to the right and bottom of the image.
                fft_uint8 = cv2.copyMakeBorder(fft_uint8, 15, 15, 15, 15,
                                               cv2.BORDER_CONSTANT,
                                               value=(255, 255, 255))
                fft_uint8 = cv2.cvtColor(fft_uint8, cv2.COLOR_GRAY2BGR)
                for sub_image_loc in sub_image_locations:
                    cv2.rectangle(
                        fft_uint8,
                        (sub_image_loc["start_width"]+15,
                         sub_image_loc["start_height"]+15),
                        (sub_image_loc["end_width"]+15,
                         sub_image_loc["end_height"]+15),
                        (0, 0, 255),
                        1
                    )
                print("Sub image locations are drawn.")
                cv2.imshow("Sub-image Locations", fft_uint8)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                del fft_uint8
    # Add the last sub image location.
    print("Sub image locations are calculated."
          f"{len(sub_image_locations)} sub images found.")

    # Create the sub image pairs.
    print("Creating sub image pairs...")
    sub_image_pairs = []
    for sub_img_loc in sub_image_locations:
        original_sub = get_sub_image(
            hr_fft,
            (sub_img_loc["start_width"], sub_img_loc["start_height"]),
            (SUB_IMAGE_SIZE["width"], SUB_IMAGE_SIZE["height"])
        )
        generated_sub = get_sub_image(
            hat_fft,
            (sub_img_loc["start_width"], sub_img_loc["start_height"]),
            (SUB_IMAGE_SIZE["width"], SUB_IMAGE_SIZE["height"])
        )

        sub_image_pairs.append(SubImagePair(
            original=original_sub,
            generated=generated_sub,
            location_width_start=sub_img_loc["start_width"],
            location_height_start=sub_img_loc["start_height"],
            location_width_end=sub_img_loc["end_width"],
            location_height_end=sub_img_loc["end_height"],
        ))
    print("Sub image pairs are created. "
          f"{len(sub_image_pairs)} sub image pairs found.")

    # Calculate the HRI95 for each sub image pair.
    print("Calculating HRI95, PSNR, SSIM, MSE for each sub image pair...")
    for index, sub_image_pair in enumerate(sub_image_pairs):
        if sub_image_pair.generated.shape != sub_image_pair.original.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

        # Find the center point.
        pred_x_center = sub_image_pair.generated.shape[0] // 2
        pred_y_center = sub_image_pair.generated.shape[1] // 2

        # Assign half of the image size as the initial grid size.
        grid_size = sub_image_pair.generated.shape[0]

        while True:
            # Get the grid.
            grid_pred = sub_image_pair.generated[
                pred_x_center - grid_size//2: pred_x_center + grid_size//2,
                pred_y_center - grid_size//2: pred_y_center + grid_size//2,
            ]
            grid_true = sub_image_pair.original[
                pred_x_center - grid_size//2: pred_x_center + grid_size//2,
                pred_y_center - grid_size//2: pred_y_center + grid_size//2,
            ]

            # Check the results of the grid.
            psnr_result = psnr(
                grid_true,
                grid_pred,
                data_range=sub_image_pair.original.max() - sub_image_pair.original.min(),
            )
            ssim_result = ssim(
                grid_true,
                grid_pred,
                data_range=sub_image_pair.original.max() - sub_image_pair.original.min(),
            )
            mse_result = mse(
                grid_true,
                grid_pred,
            )

            if grid_size == sub_image_pair.generated.shape[0]:
                # Save the initial results.
                sub_image_pairs[index].result_psnr = psnr_result
                sub_image_pairs[index].result_ssim = ssim_result
                sub_image_pairs[index].result_mse = mse_result

            if ssim_result > HRI_SUCCESS_THRES:
                # The radius is the half of the grid size.
                sub_image_pairs[index].result_hri95 = grid_size // 2
                break

            # Decrease the grid size by 2.
            grid_size -= 2

            # A stop condition.
            if grid_size < 7:
                break

        # If the radius is not found, return the minimum radius.
        if sub_image_pairs[index].result_hri95 is None:
            sub_image_pairs[index].result_hri95 = 0
    print("HRI95, PSNR, SSIM, MSE are calculated.")

    if SHOW_SUB_IMAGE_METRICS:
        # Convert the FFT into an image with 0->255 range.
        # Draw the sub-image locations with red rectangles.
        # Place the psnr result on the middle of the red rectangle.
        fft_uint8 = (hr_fft / hr_fft.max() * 255).astype("uint8")
        cv2.imshow("FFT", fft_uint8)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Convert fft_uint8 to MatLike and resize it by x4.
        fft_uint8 = cv2.cvtColor(fft_uint8, cv2.COLOR_GRAY2BGR)
        fft_uint8 = cv2.resize(fft_uint8,
                               (fft_uint8.shape[1] * RESIZE_FACTOR,
                                fft_uint8.shape[0] * RESIZE_FACTOR))

        print("Drawing sub image locations...")
        for sub_image_pair in sub_image_pairs:
            # Draw the red rectangle.
            cv2.rectangle(
                fft_uint8,
                (sub_image_pair.location_width_start*RESIZE_FACTOR,
                 sub_image_pair.location_height_start*RESIZE_FACTOR),
                (sub_image_pair.location_width_start*RESIZE_FACTOR +
                 (SUB_IMAGE_SIZE["width"] * RESIZE_FACTOR),
                 sub_image_pair.location_height_start*RESIZE_FACTOR +
                 (SUB_IMAGE_SIZE["height"] * RESIZE_FACTOR)),
                (0, 0, 255),
                1
            )

            # Place the psnr result on the middle of the red rectangle.
            cv2.putText(
                fft_uint8,
                f"{sub_image_pair.result_hri95}px",
                (
                    sub_image_pair.location_width_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["width"]//2,
                    sub_image_pair.location_height_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["height"]//2
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                fft_uint8,
                f"{sub_image_pair.result_psnr:.1f}dB",
                (
                    sub_image_pair.location_width_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["width"]//2-10,
                    sub_image_pair.location_height_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["height"]//2+15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                fft_uint8,
                f"{sub_image_pair.result_ssim:.2f}",
                (
                    sub_image_pair.location_width_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["width"]//2-10,
                    sub_image_pair.location_height_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["height"]//2+30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                fft_uint8,
                f"{sub_image_pair.result_mse:.2f}px2",
                (
                    sub_image_pair.location_width_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["width"]//2-10,
                    sub_image_pair.location_height_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["height"]//2+50
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    if SAVE_METRICS_TO_CSV:
        # Save the results into metadata.
        print("Saving the results into CSV file...")
        with open("results.csv", "w", encoding="utf-8") as file:
            file.write(
                "X_start, X_end, Y_start, Y_end, HRI95, PSNR, SSIM, MSE\n")
            for sub_image_pair in sub_image_pairs:
                file.write(f"{sub_image_pair.location_width_start},"
                           f"{sub_image_pair.location_width_end},"
                           f"{sub_image_pair.location_height_start},"
                           f"{sub_image_pair.location_height_end},"
                           f"{sub_image_pair.result_hri95},"
                           f"{sub_image_pair.result_psnr},"
                           f"{sub_image_pair.result_ssim},"
                           f"{sub_image_pair.result_mse}\n")
        print("Results are saved into CSV file.")

    if SAVE_VIDEO_SUB_IMAGE_METRICS:
        # Show the results one by one per sub image,
        # and save them into a video.
        print("Showing the results one by one per sub image...")
        video_writer = cv2.VideoWriter(
            "results.mp4",
            cv2.VideoWriter_fourcc(*"MP4V"),
            1,
            (hr_fft.shape[1]*RESIZE_FACTOR, hr_fft.shape[0]*RESIZE_FACTOR)
        )

        for sub_image_pair in sub_image_pairs:
            fft_uint8 = (hr_fft / hr_fft.max() * 255).astype("uint8")
            fft_uint8 = cv2.cvtColor(fft_uint8, cv2.COLOR_GRAY2BGR)
            fft_uint8 = cv2.resize(fft_uint8,
                                   (fft_uint8.shape[1] * RESIZE_FACTOR,
                                    fft_uint8.shape[0] * RESIZE_FACTOR))

            # Draw the red rectangle.
            cv2.rectangle(
                fft_uint8,
                (sub_image_pair.location_width_start*RESIZE_FACTOR,
                 sub_image_pair.location_height_start*RESIZE_FACTOR),
                (sub_image_pair.location_width_start*RESIZE_FACTOR +
                 (SUB_IMAGE_SIZE["width"] * RESIZE_FACTOR),
                 sub_image_pair.location_height_start*RESIZE_FACTOR +
                 (SUB_IMAGE_SIZE["height"] * RESIZE_FACTOR)),
                (0, 0, 255),
                4
            )

            # Place the psnr result on the middle of the red rectangle.
            cv2.putText(
                fft_uint8,
                f"{sub_image_pair.result_hri95}px",
                (
                    sub_image_pair.location_width_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["width"]//2 + 30,
                    sub_image_pair.location_height_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["height"]//2 + 20
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                fft_uint8,
                f"{sub_image_pair.result_psnr:.1f}dB",
                (
                    sub_image_pair.location_width_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["width"]//2 + 15,
                    sub_image_pair.location_height_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["height"]//2 + 70
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                fft_uint8,
                f"{sub_image_pair.result_ssim:.2f}",
                (
                    sub_image_pair.location_width_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["width"]//2 + 15,
                    sub_image_pair.location_height_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["height"]//2 + 120
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                fft_uint8,
                f"{sub_image_pair.result_mse:.2f}px2",
                (
                    sub_image_pair.location_width_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["width"]//2 + 15,
                    sub_image_pair.location_height_start *
                    RESIZE_FACTOR + SUB_IMAGE_SIZE["height"]//2 + 170
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
            video_writer.write(fft_uint8)
            del fft_uint8

        video_writer.release()
        print("Results are shown one by one per sub image.")

    if CALCULATE_AVERAGE_METRICS:
        # Calculate the average metrics.
        print("Calculating the average metrics...")
        hri95_sum = 0
        psnr_sum = 0
        ssim_sum = 0
        mse_sum = 0
        for sub_image_pair in sub_image_pairs:
            hri95_sum += sub_image_pair.result_hri95
            psnr_sum += sub_image_pair.result_psnr
            ssim_sum += sub_image_pair.result_ssim
            mse_sum += sub_image_pair.result_mse
        hri95_avg = hri95_sum / len(sub_image_pairs)
        psnr_avg = psnr_sum / len(sub_image_pairs)
        ssim_avg = ssim_sum / len(sub_image_pairs)
        mse_avg = mse_sum / len(sub_image_pairs)
        print(f"Average HRI95: {hri95_avg:.2f}px")
        print(f"Average PSNR: {psnr_avg:.2f}dB")
        print(f"Average SSIM: {ssim_avg:.2f}")
        print(f"Average MSE: {mse_avg:.2f}px2")
        print("Average metrics are calculated.")
