"""
Main application script
"""
import argparse
from harmonicsradius.sr_analyzer import SRAnalyzer
from harmonicsradius.metrics import HarmonicsRadius
from harmonicsradius.image import Image


def argument_parser() -> dict[str, str]:
    """Parse the command line arguments and return the parsed arguments.
    :return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate HRI95 metric for given image pairs."
    )
    parser.add_argument(
        "--true",
        "-t",
        type=str,
        help="Path to the true image",
        required=True,
        dest="true_image",
    )
    parser.add_argument(
        "--predicted",
        "-p",
        type=str,
        help="Path to the predicted image",
        required=True,
        dest="predicted_image",
    )

    arguments = parser.parse_args()
    return {"true": arguments.true_image,
            "predicted": arguments.predicted_image}


def main() -> None:
    """Main function of the application."""
    # Return the file paths of the images.
    images = argument_parser()

    # Read the images.
    true_image = Image(images['true'], name="true_image")
    predicted_image = Image(images['predicted'], name="predicted_image")

    # Create the analyzer.
    analyzer = SRAnalyzer()

    # Add metrics.
    analyzer.add_metric(HarmonicsRadius())

    # Add images.
    analyzer.add_reference_image(true_image)
    analyzer.add_image(predicted_image)

    # Calculate the metrics.
    [hri95_result] = analyzer.calculate()
    print(f"{hri95_result.value:.2f}%")


if __name__ == "__main__":
    main()
