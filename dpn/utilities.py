import numpy as np


def get_largest_bbox_dimension(bboxes):
    """Function to extract the largest dimensions of each bounding box of a set of bounding boxes."""
    y1 = bboxes[:, 0]
    x1 = bboxes[:, 1]
    y2 = bboxes[:, 2]
    x2 = bboxes[:, 3]

    widths = x2 - x1
    heights = y2 - y1

    # Keep largest dimensions of each bbox as diameter.
    number_of_bboxes = widths.size
    largest_dimension = np.zeros(number_of_bboxes)
    largest_dimension[widths >= heights] = widths[widths >= heights]
    largest_dimension[heights >= widths] = heights[heights >= widths]

    return largest_dimension


def calculate_equivalent_diameter(areas):
    """Function to calculate the set of equivalent diameters of a set of areas."""
    diameters = np.sqrt(4 * areas / np.pi)
    return diameters