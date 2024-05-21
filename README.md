# Harmonics Radius Index

The description will be added soon.

### Example API Usage

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