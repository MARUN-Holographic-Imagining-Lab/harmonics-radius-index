from core.analyze import SuperResolutionAnalyzer
from core.image_preperation_factory import ImagePossibleTypes

analyzer = SuperResolutionAnalyzer("test_images/sehir_hr.png", scale=8)

analyzer.show_image("all", domain="spatial")
analyzer.show_image("all", domain="frequency")

# Different Thresholds for Bicubic Interpolation
for threshold in range(1, 13):
    analyzer.remove_components_below(ImagePossibleTypes.BICUBIC, threshold=threshold)

for threshold in [9, 7]:
    for possible_type in ImagePossibleTypes:
        if possible_type != ImagePossibleTypes.SMALL:
            analyzer.remove_components_below(possible_type, threshold=threshold)
