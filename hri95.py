"""
Main application script
"""
import argparse

from core.metrics import (
    MeanSquaredError,
    HarmonicsRadius,
    StructuralSimilarityIndex,
    PeakSignalToNoiseRatio
)

from core.settings import SRAnalyzerSettings
from core.image import Image
from core.sr_analyzer import SRAnalyzer


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


if __name__ == "__main__":
    # Return the file paths of the images.
    images = argument_parser()

    # Read the images.
    true_image = Image(images['true'], name="true_image")
    predicted_image = Image(images['predicted'], name="predicted_image")

    # Create the analyzer.
    analyzer = SRAnalyzer(
        SRAnalyzerSettings(name="HRI95 Calculator")
    )

    # Add metrics.
    analyzer.add_metric(HarmonicsRadius())
    analyzer.add_metric(MeanSquaredError())
    analyzer.add_metric(StructuralSimilarityIndex())
    analyzer.add_metric(PeakSignalToNoiseRatio())

    # Add images.
    analyzer.add_reference_image(true_image)
    analyzer.add_image(predicted_image)

    # Calculate the metrics.
    results = analyzer.calculate()
    print("\nImage Quality Metrics\n")
    print("True Image: ", images['true'])
    print("Predicted Image: ", images['predicted'])
    print("")
    for result in results:
        print(f"- {result.name}: {result.value:.3f} {result.unit}")
