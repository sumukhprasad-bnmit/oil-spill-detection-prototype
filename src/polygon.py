import cv2
import numpy as np


def mask_to_polygons(
    mask,
    min_area=20,
    approximation_epsilon=0.001,
):
    """
    Convert a binary segmentation mask into polygons.

    Args:
        mask:
            Binary numpy array [H, W].

        min_area:
            Ignore connected regions smaller than this
            many pixels.

        approximation_epsilon:
            Controls polygon simplification.

    Returns:
        List of polygons, where each polygon is an
        Nx2 numpy array containing (x, y) coordinates.
    """

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    polygons = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            closed=True,
        )

        epsilon = (
            approximation_epsilon
            * perimeter
        )

        approximated = cv2.approxPolyDP(
            contour,
            epsilon,
            closed=True,
        )

        polygon = (
            approximated
            .reshape(-1, 2)
        )

        polygons.append(polygon)

    return polygons