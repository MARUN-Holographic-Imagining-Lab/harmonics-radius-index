# Harmonics Radius Index
Harmonics Radius Index is a performance index for evaluating the quality of super-resolution images. It is based on the harmonic mean of the radii of the circles that contain the same amount of energy in the Fourier domain of the true and predicted images.

Please refer to the following paper for more details:

```
The paper is under review. Please check back later.
```


## Installation & Usage
First run for hr95 program may take a while, however, it will be faster in the following runs.

```bash
pip install harmonicsradius
hri95 -t <true_image_path> -p <predicted_image_path>
```

## API
The harmonicsradius package provides an API for calculating the Harmonics Radius Index and other image quality metrics. The API is designed to be simple and easy to use. The following metrics are available:
- Harmonics Radius Index
- Mean Squared Error
- Structural Similarity Index
- Peak Signal to Noise Ratio

The API is designed to be simple and easy to use. The following example demonstrates how to use the API to calculate the Harmonics Radius Index and other image quality metrics.

```python
from harmonicsradius.metrics import (
    MeanSquaredError,
    HarmonicsRadius,
    StructuralSimilarityIndex,
    PeakSignalToNoiseRatio
)

from harmonicsradius.image import Image
from harmonicsradius.sr_analyzer import SRAnalyzer

# Read the images.
true_image = Image(TRUE_IMAGE_PATH, name="true_image")
predicted_image = Image(PRED_IMAGE_PATH, name="predicted_image")

# Create the analyzer.
analyzer = SRAnalyzer()

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
for result in results:
    print(result)
```