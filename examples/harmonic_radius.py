"""
This is the standalone version of the harmonic radius index.
"""

import numpy
from skimage.metrics import structural_similarity

from harmonicsradius.image import Image
from harmonicsradius.utils import get_fft_of_image, draw_square_from_center


def harmonic_radius(y_pred: Image | numpy.ndarray,
                    y_true: Image | numpy.ndarray,
                    success_thres: float = 0.95) -> float:
    """Calculate the Harmonics Radius' index.
    :param y_pred: The predicted image.
    :param y_true: The ground truth image.
    :return: The Harmonics Radius' index.
    """
    # Check the shapes of the parameters.
    y_true_shape = y_true.shape if isinstance(y_true, Image) else y_true.shape
    y_pred_shape = y_pred.shape if isinstance(y_pred, Image) else y_pred.shape
    if y_true_shape != y_pred_shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # Get the 2D magnitude spectrum in log scale.
    fft_of_true: numpy.ndarray = get_fft_of_image(
        y_true.get_image() if isinstance(y_true, Image) else y_true,
        scale_log=True
    )
    fft_of_pred: numpy.ndarray = get_fft_of_image(
        y_pred.get_image() if isinstance(y_pred, Image) else y_pred,
        scale_log=True
    )

    # Find the center point.
    pred_x_center = fft_of_pred.shape[0] // 2
    pred_y_center = fft_of_pred.shape[1] // 2

    # Assign half of the image size as the initial grid size.
    grid_size = fft_of_pred.shape[0] // 2
    while True:
        # Get the grid.
        grid_pred = fft_of_pred[
            pred_x_center - grid_size//2: pred_x_center + grid_size//2,
            pred_y_center - grid_size//2: pred_y_center + grid_size//2,
        ]
        grid_true = fft_of_true[
            pred_x_center - grid_size//2: pred_x_center + grid_size//2,
            pred_y_center - grid_size//2: pred_y_center + grid_size//2,
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
            draw_square_from_center(
                fft_of_pred, (pred_x_center, pred_y_center), grid_size // 2)
            return grid_size // 2

        # Decrease the grid size by 2.
        grid_size -= 2

        # A stop condition.
        if grid_size < 7:
            break

    # If the radius is not found, return the minimum radius.
    return 0
