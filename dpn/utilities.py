import numpy as np
from scipy.spatial.distance import pdist
from skimage.morphology import convex_hull_image
from skimage.measure import find_contours


def get_major_bbox_side_length(bboxes):
    """Extract the largest dimensions of each bounding box of a set of bounding boxes.

    :param bboxes: List or numpy array of bounding boxes.
    :return: List of largest bounding box side lengths.
    """

    bboxes = np.asarray(bboxes)

    y1 = bboxes[:, 0]
    x1 = bboxes[:, 1]
    y2 = bboxes[:, 2]
    x2 = bboxes[:, 3]

    widths = x2 - x1
    heights = y2 - y1

    # Keep largest dimensions of each bbox as diameter.
    number_of_bboxes = widths.size
    major_bbox_side_length = np.zeros(number_of_bboxes)
    major_bbox_side_length[widths >= heights] = widths[widths >= heights]
    major_bbox_side_length[heights >= widths] = heights[heights >= widths]

    return major_bbox_side_length.tolist()


def calculate_equivalent_diameter(areas):
    """Calculate the equivalent diameters of a list or numpy array of areas.

    :param areas: List or numpy array of areas.
    :return: List of equivalent diameters.
    """

    areas = np.asarray(areas)

    diameters = np.sqrt(4 * areas / np.pi)
    return diameters.tolist()


def get_maximum_feret_diameter(masks):
    """Calculates the maximum feret diameter for a list of masks.
    Based on: https://github.com/scikit-image/scikit-image/issues/2320#issuecomment-256057683
    See also:   https://github.com/scikit-image/scikit-image/pull/1780

    :param masks: List of masks.
    :return: List of maximum Feret diameters.
    """

    max_feret_diameters = []

    for mask in masks:
        try:
            mask_convex_hull = convex_hull_image(mask)
            coordinates = np.vstack(find_contours(mask_convex_hull, 0.5, fully_connected="high"))
            distances = pdist(coordinates, "sqeuclidean")
            max_feret_diameter = np.sqrt(np.max(distances))

            max_feret_diameters.append(max_feret_diameter)
        except ValueError:
            print("Ignored mask, due to small size.")

    return max_feret_diameters
