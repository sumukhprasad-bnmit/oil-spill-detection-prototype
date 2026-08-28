import cv2
import numpy as np


def mask_to_polygons(
    mask,
    min_area=100,
    approximation_epsilon=0.001,
):
    """
    Convert a binary segmentation mask into polygon regions.

    Returns a list of dictionaries containing:
        polygon
        area
        perimeter
        bbox
    """

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    regions = []

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

        polygon = cv2.approxPolyDP(
            contour,
            epsilon,
            closed=True,
        )

        polygon = polygon.reshape(-1, 2)

        x, y, w, h = cv2.boundingRect(contour)

        regions.append({
            "polygon": polygon,
            "area": float(area),
            "perimeter": float(perimeter),
            "bbox": (x, y, w, h),
        })

    # largest regions first
    regions.sort(
        key=lambda x: x["area"],
        reverse=True,
    )

    return regions