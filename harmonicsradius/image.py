"""
Holds the Image class to be used in the analisys.
"""
from numpy import ndarray
from typing import Union
from harmonicsradius.utils import read_image, save_image


class Image:
    """Holds the image objects and preproceses if needed."""

    def __init__(self, image_path: Union[str, 'Image', ndarray],
                 name: str, preprocess: callable = None) -> None:
        """Constructor of the Image class.

        :param image_path: The path of the image, an Image object or
        NumPy Ndarray.
        :param name: The name of the image.
        :param preprocess: The preprocess function to be applied to the image.
        """
        self._name = name

        # Read the image if it is Image or a path.
        if isinstance(image_path, Image):
            self._path = None
            self._original_image = image_path.get_image()
        elif isinstance(image_path, str):
            self._path = image_path
            try:
                self._original_image = read_image(image_path)
            except Exception as e:
                raise ValueError(f"Error while reading the image: {e}.")
            if self._original_image is None:
                raise ValueError("Error while reading the image.")
        elif isinstance(image_path, ndarray):
            self._path = None
            self._original_image = image_path
        else:
            raise ValueError("Image path must be a string or an Image object.")

        # Preprocess the image if needed.
        self._preprocess_function = preprocess
        if self._preprocess_function is not None:
            self._image = self._preprocess_function(self._original_image)
        else:
            self._image = self._original_image

    def get_image(self) -> ndarray:
        """
        Return the image.

        :return: The image in NumPy ndarray.
        """
        return self._image

    def get_shape(self) -> tuple[int, int, int]:
        """
        Return the shape of the image.

        :return: The shape of the image as (height, width, channels).
        """
        return self._image.shape

    def get_name(self) -> str:
        """
        Return the name of the image.

        :return: The name of the image.
        """
        return self._name

    def save_image(self, path: str) -> None:
        """
        Save the image to the path.

        :param path: The path to save the image.
        """
        save_image(self._image, path)
