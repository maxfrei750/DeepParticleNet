import numpy as np
from scipy.spatial.distance import pdist
from skimage.morphology import convex_hull_image
from skimage.measure import find_contours


def get_major_bbox_side_length(bboxes):
    """Function to extract the largest dimensions of each bounding box of a set of bounding boxes."""

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
    """Function to calculate the set of equivalent diameters of a set of areas."""

    areas = np.asarray(areas)

    diameters = np.sqrt(4 * areas / np.pi)
    return diameters.tolist()


def get_maximum_feret_diameter(masks):
    """ Calculates the maximum feret diameter for each mask in masks.
    masks: [height, width, N]

    Based on: https://github.com/scikit-image/scikit-image/issues/2320#issuecomment-256057683
    See also:   https://github.com/scikit-image/scikit-image/pull/1780
    """

    masks = np.atleast_3d(masks)
    number_of_masks = masks.shape[2]

    max_feret_diameters = []

    for i_mask in range(number_of_masks):
        mask = masks[:, :, i_mask]
        mask_convex_hull = convex_hull_image(mask)
        coordinates = np.vstack(find_contours(mask_convex_hull, 0.5, fully_connected="high"))
        distances = pdist(coordinates, "sqeuclidean")
        max_feret_diameter = np.sqrt(np.max(distances))

        max_feret_diameters.append(max_feret_diameter)

    return np.array(max_feret_diameters)