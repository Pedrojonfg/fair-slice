"""
Module: Vision (Computer Vision Pipeline)
Owner: Pablo

Responsible for:
  - Receiving a photo of a dish
  - Calling Vertex AI / Gemini for ingredient identification
  - Producing a per-pixel ingredient density map

See README.md Section 2 for the full contract specification.
"""

import numpy as np


def segment_dish(image_path: str) -> tuple[np.ndarray, dict[int, str]]:
    """
    Segments a dish photo into per-pixel ingredient maps.

    Args:
        image_path: Path to the input image (JPG/PNG).

    Returns:
        ingredient_map: np.ndarray of shape (H, W, K), dtype float32.
                        Values in [0, 1]. Each channel k represents
                        the density of ingredient k at that pixel.
                        Channels should sum to ~1.0 at each pixel.
        ingredient_labels: dict mapping channel index to ingredient name.
                           Example: {0: "dough", 1: "crust", 2: "pepperoni"}

    Raises:
        ValueError: If image cannot be loaded or is not a valid photo.
    """
    raise NotImplementedError("TODO: Implement with Vertex AI + segmentation")