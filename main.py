from core.analyze import SuperResolutionAnalyzer
from core.image_preperation_factory import ImagePossibleTypes

analyzer = SuperResolutionAnalyzer("inji_hr.png", scale=8)
analyzer.show_image("all", domain="spatial")
analyzer.show_image("all", domain="frequency")

for possible_type in ImagePossibleTypes:
    if possible_type != ImagePossibleTypes.SMALL:
        analyzer.remove_components_below(possible_type, threshold=7)
