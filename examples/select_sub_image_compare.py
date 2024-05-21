
import argparse
from harmonicsradius.image import Image
from harmonicsradius.sr_analyzer import SRAnalyzer, SRAnalyzerSettings
from harmonicsradius.metrics import HarmonicsRadius, StructuralSimilarityIndex


def shift_perc_to_pixels(shift_amount_perc: int,
                         image: Image) -> int:
    """Convert shift amount in percentage to pixels."""
    size = image.get_shape()[1]
    return int(size * shift_amount_perc / 100)


def argument_parser() -> dict[str, str | list[int]]:
    """Parse the arguments."""
    _parser = argparse.ArgumentParser(description="Find shift SSIM/HRI95.")
    _parser.add_argument("--image", type=str, required=True,
                         dest="image",
                         help="Path to the image.")
    _parser.add_argument("--shift_amount", type=int, required=False,
                         dest="shift_amount",
                         help="Shift amount in percentage.")
    _parser.add_argument("--shift_amounts", type=str, required=False,
                         dest="shift_amounts",
                         help="Shift amounts in percentage.")
    _args = _parser.parse_args()

    # Check if shift_amounts or shift_amount is provided.
    if _args.shift_amounts:
        return {
            "image": _args.image,
            "shift_amount": [int(i) for i in _args.shift_amounts.split(",")]
        }
    if _args.shift_amount:
        return {
            "image": _args.image,
            "shift_amount": [int(_args.shift_amount)]
        }
    raise ValueError("Either shift_amount or shift_amounts is required.")


if __name__ == "__main__":
    # Retrieve the arguments.
    arguments: dict = argument_parser()

    # Read the image.
    image_path: str = arguments['image']
    image: Image = Image(image_path, name="hr_image")

    # Traverse the shift amounts.
    for shift in arguments['shift_amount']:
        # Define analyzer.
        analyzer: SRAnalyzer = SRAnalyzer(SRAnalyzerSettings(name=None))
        analyzer.add_metric(HarmonicsRadius())
        analyzer.add_metric(StructuralSimilarityIndex())

        # Select a subset of HR image to use as reference.
        shift_amount: int = shift_perc_to_pixels(
            shift,
            image
        )
        reference: Image = Image(
            image.get_image()[
                0:288,
                0:(288-shift_amount)
            ], name="reference"
        )
        print(f"Reference image selected with shift amount: {shift_amount}")
        print(f"Reference image shape: {reference.get_shape()}")

        # Add the reference image.
        analyzer.add_reference_image(reference)

        # Calculate the x-axis start and end positions by
        # applying a shift to the reference image shape.
        _, ref_x, _ = reference.get_shape()
        shift_amount: int = shift_perc_to_pixels(
            shift, reference)
        new_x_end: int = ref_x + shift_amount
        new_x_start: int = shift_amount

        # Select a subset of the image from original s"image".
        shifted: Image = Image(
            image.get_image()[
                0:288,
                new_x_start:new_x_end
            ], name="shifted"
        )
        print(f"Shifted image selected with shift amount: {shift_amount}")
        print(f"Shifted image shape: {shifted.get_shape()}")

        # Add the shifted image.
        analyzer.add_image(shifted)

        # Calculate the metrics.
        results = analyzer.calculate()
        for result in results:
            print(result)
