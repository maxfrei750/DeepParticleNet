from dpn.sizedistribution import SizeDistribution
from dpn.utilities import calculate_equivalent_diameter, get_major_bbox_side_length
import numpy as np
from skimage.measure import regionprops
import os
from dpn.storable import Storable


class Results(Storable):
    def __init__(self, detection=None):
        self.detections = list()

        if detection is not None:
            self.append_detection(detection)

    # Dependant attributes
    @property
    def masks(self):
        all_masks = list()
        for detection in self.detections:
            all_masks += detection.masks
        return all_masks

    @property
    def images(self):
        all_images = list()
        for detection in self.detections:
            all_images += detection.images
        return all_images

    @property
    def bboxes(self):
        all_bboxes = list()
        for detection in self.detections:
            all_bboxes += detection.bboxes
        return all_bboxes

    @property
    def class_ids(self):
        class_ids = list()
        for detection in self.detections:
            class_ids += detection.class_ids
        return class_ids

    @property
    def scores(self):
        all_scores = list()
        for detection in self.detections:
            all_scores += detection.scores
        return all_scores

    @property
    def number_of_detections(self):
        return len(self.detections)

    @property
    def detection_ids(self):
        return range(self.number_of_detections-1)

    # Methods
    def append_detection(self, detection):
        self.detections += [detection]

    def filter_by_minimum_score(self, minimum_score, verbose=False):
        # Remove instances with a score below a certain threshold.
        for detection in self.detections:
            detection.filter_by_minimum_score(minimum_score, verbose=verbose)

    def filter_by_class(self, class_id_to_keep, verbose=False):
        # Remove instances with a class other than the given class.
        for detection in self.detections:
            detection.filter_by_class(class_id_to_keep, verbose=verbose)

    def filter_by_minimum_area(self, minimum_area, verbose=False):
        # Remove instances with areas smaller than the given minimum area.
        for detection in self.detections:
            detection.filter_by_minimum_area(minimum_area, verbose=verbose)

    def filter_by_maximum_area(self, maximum_area, verbose=False):
        # Remove instances with areas larger than the given maximum area.
        for detection in self.detections:
            detection.filter_by_maximum_area(maximum_area, verbose=verbose)

    def filter_by_minimum_circularity(self, minimum_circularity, verbose=False):
        # Remove instances with circularities smaller than the given minimum circularity.
        for detection in self.detections:
            detection.filter_by_minimum_circularity(minimum_circularity, verbose=verbose)

    def clear_border_objects(self, verbose=False):
        # Remove instances that touch the border of the image.
        for detection in self.detections:
            detection.clear_border_objects(verbose=verbose)

    def to_size_distribution(self, measurand):
        # Return a size distribution based on a certain measurand.

        measurand = measurand.lower()

        # Check inputs.
        assert measurand in \
            ["equivalent_diameter", "equivalent_diameter_convex", "major_bbox_side_length", "major_axis_length"], \
            "Expected measurand to be one of the following: " \
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

    def display_detection_image(self, detection_id):
        self.detections[detection_id].display_detection_image()

    def save_detection_image(self, detection_id, output_path, do_display_detections=False):
        self.detections[detection_id].save_detection_image(output_path, do_display_detections=do_display_detections)

    def save_all_detection_images(self, output_folder, filename_prefix="", filetype="png", do_display_detections=False):
        for detection_id in self.detection_ids:
            filename = filename_prefix+"_detection_{:d}.".format(detection_id)+filetype
            output_path = os.path.join(output_folder, filename)
            self.detections[detection_id].save_detection_image(output_path, do_display_detections=do_display_detections)
