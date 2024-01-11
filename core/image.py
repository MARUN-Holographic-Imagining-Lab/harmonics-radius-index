"""
Holds the Image class to be used in the analisys.
"""
from numpy import ndarray
from core.utils import read_image, save_image

class Image:
    """Holds the image objects and preproceses if needed."""
    def __init__(self, image_path: str or 'Image', name: str, preprocess: callable = None) -> None:
        self._name = name

        # Read the image if it is Image or a path.
        if isinstance(image_path, Image):
            self._path = None
            self._original_image = image_path.get_image()
        else:
            self._path = image_path
            self._original_image = read_image(image_path)

        # Preprocess the image if needed.
        self._preprocess_function = preprocess
        if self._preprocess_function is not None:
            self._image = self._preprocess_function(self._original_image)
        else:
            self._image = self._original_image

    def get_image(self) -> ndarray:
        """
        Return the image.
        """
        return self._image

    def get_shape(self) -> tuple[int, int, int]:
        """
        Return the shape of the image.
        """
        return self._image.shape

    def get_name(self) -> str:
        """
        Return the name of the image.
        """
        return self._name

    def save_image(self, path: str) -> None:
        """
        Save the image to the path.
        """
        save_image(self._image, path)
