from dpn.sizedistribution import SizeDistribution
from dpn.utilities import calculate_equivalent_diameter, get_major_bbox_side_length
from itertools import compress
import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import clear_border


class Results:
    def __init__(self, results_dict_list=None):
        self.masks = list()
        self.class_ids = list()
        self.bboxes = list()
        self.scores = list()

        if results_dict_list is not None:
            self.append_from_dicts(results_dict_list)

    def __add__(self, other):
        self.masks += other.masks
        self.class_ids += other.class_ids
        self.bboxes += other.bboxes
        self.scores += other.scores

    def append_raw(self, masks, bboxes, scores, class_ids):
        self.masks += masks
        self.class_ids += class_ids
        self.bboxes += bboxes
        self.scores += scores

    def append_from_dicts(self, results_dict_list):
        for results_dict in results_dict_list:
            # Extract values results from dict.
            new_masks = results_dict["masks"]
            new_class_ids = results_dict["class_ids"]
            new_bboxes = results_dict["rois"]
            new_scores = results_dict["scores"]

            # Convert class_ids to list.
            new_class_ids = new_class_ids.tolist()

            # Convert bboxes to list.
            new_bboxes = new_bboxes.tolist()

            # Convert scores to list.
            new_scores = new_scores.tolist()

            # Get number of instances
            number_of_new_instances = len(new_class_ids)

            # Convert masks to list of masks.
            new_masks = np.split(new_masks, number_of_new_instances, axis=2)
            new_masks = [np.squeeze(new_mask) for new_mask in new_masks]

            # Append new results.
            self.masks += new_masks
            self.class_ids += new_class_ids
            self.bboxes += new_bboxes
            self.scores += new_scores

    def filter_by_list(self, do_keep):
        self.masks = list(compress(self.masks, do_keep))
        self.class_ids = list(compress(self.class_ids, do_keep))
        self.bboxes = list(compress(self.bboxes, do_keep))
        self.scores = list(compress(self.scores, do_keep))

    def filter_by_score(self, minimum_score):
        # Remove instances with scores smaller then the given minimum score.
        do_keep = [score >= minimum_score for score in self.scores]
        self.filter_by_list(do_keep)

    def filter_by_class(self, class_id_to_keep):
        # Remove instances with a class other than the given class.
        do_keep = [class_id is class_id_to_keep for class_id in self.class_ids]
        self.filter_by_list(do_keep)

    def clear_border_objects(self):
        # Remove instances that touch the border of the image.
        cleared_masks = [clear_border(mask) for mask in self.masks]
        do_keep = [np.any(cleared_mask) for cleared_mask in cleared_masks]
        self.filter_by_list(do_keep)

    def to_size_distribution(self, measurand):
        # Return a size distribution based on a certain measurand.

        measurand = measurand.lower()

        # Check inputs.
        assert measurand in \
            ["equivalent_diameter", "equivalent_diameter_convex", "major_bbox_side_length", "major_axis_length"], \
            "Expected measurand to be one of the following:" \
            "equivalent_diameter, " \
            "equivalent_diameter_convex, " \
            "major_bbox_side_length, " \
            "major_axis_length"

        # Analyze region properties of the masks.
        instances = [regionprops(mask.astype(int))[0] for mask in self.masks]

        if measurand == "equivalent_diameter":
            areas = [instance.filled_area for instance in instances]
            measurements = calculate_equivalent_diameter(areas)
        elif measurand == "equivalent_diameter_convex":
            areas = [instance.convex_area for instance in instances]
            measurements = calculate_equivalent_diameter(areas)
        elif measurand == "major_bbox_side_length":
            measurements = get_major_bbox_side_length(self.bboxes)
        elif measurand == "major_axis_length":
            measurements = [instance.major_axis_length for instance in instances]

        # Create and return a SizeDistribution-object.
        size_distribution = SizeDistribution("px")
        size_distribution.sizes = np.asarray(measurements)

        return size_distribution
