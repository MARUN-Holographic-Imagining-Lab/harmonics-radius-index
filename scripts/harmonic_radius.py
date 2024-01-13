"""
This is the standalone version of the harmonic radius index.
"""

import numpy
from skimage.metrics import structural_similarity

from core.image import Image
from core.utils import get_fft_of_image


def harmonic_radius(y_pred: Image,
                    y_true: Image,
                    success_thres: float = 0.95) -> float:
    """Calculate the Harmonics Radius' index.
    :param y_pred: The predicted image.
    :param y_true: The ground truth image.
    :return: The Harmonics Radius' index.
    """
    # Check the shapes of the parameters.
    if y_true.get_shape() != y_pred.get_shape():
        raise ValueError("y_true and y_pred must have the same shape.")

    # Get the 2D magnitude spectrum in log scale.
    fft_of_true: numpy.ndarray = get_fft_of_image(y_true.get_image(), scale_log=True)
    fft_of_pred: numpy.ndarray = get_fft_of_image(y_pred.get_image(), scale_log=True)

    # Find the center point.
    pred_x_center = fft_of_pred.shape[0] // 2
    pred_y_center = fft_of_pred.shape[1] // 2

    # Assign half of the image size as the initial grid size.
    grid_size = fft_of_pred.shape[0] // 2
    while True:
        # Get the grid.
        grid_pred = fft_of_pred[
            pred_x_center - grid_size//2 : pred_x_center + grid_size//2,
            pred_y_center - grid_size//2 : pred_y_center + grid_size//2,
        ]
        grid_true = fft_of_true[
            pred_x_center - grid_size//2 : pred_x_center + grid_size//2,
            pred_y_center - grid_size//2 : pred_y_center + grid_size//2,
        ]

        # Check the SSIM of the grid.
        ssim_result = structural_similarity(
            grid_true,
            grid_pred,
            data_range=fft_of_true.max() - fft_of_true.min(),
            multichannel=False,
        )

        if ssim_result > success_thres:
            # The radius is the half of the grid size.
            return grid_size // 2

        # Decrease the grid size by 2.
        grid_size -= 2

        # A stop condition.
        if grid_size < 7:
            break

    # If the radius is not found, return the minimum radius.
    return 0
