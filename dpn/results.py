from dpn.sizedistribution import SizeDistribution
from dpn.utilities import calculate_equivalent_diameter, get_major_bbox_side_length, get_maximum_feret_diameter
import numpy as np
from skimage.measure import regionprops
import os
from dpn.storable import Storable


class Results(Storable):
    """Class to store, filter and convert detection results, i.e. images, classes, scores, boundingboxes, masks."""
    def __init__(self, detection=None):
        """Create and initialize a Results object.

        :param detection: First detection (optional).
        """

        self.detections = list()

        if detection is not None:
            self.append_detection(detection)

    # Dependant attributes
    @property
    def masks(self):
        """List of masks."""
        all_masks = list()
        for detection in self.detections:
            all_masks += detection.masks
        return all_masks

    @property
    def images(self):
        """List of images."""
        all_images = list()
        for detection in self.detections:
            all_images += detection.images
        return all_images

    @property
    def bboxes(self):
        """List of bounding boxes."""
        all_bboxes = list()
        for detection in self.detections:
            all_bboxes += detection.bboxes
        return all_bboxes

    @property
    def class_ids(self):
        """List of class IDs."""
        class_ids = list()
        for detection in self.detections:
            class_ids += detection.class_ids
        return class_ids

    @property
    def scores(self):
        """List of scores."""
        all_scores = list()
        for detection in self.detections:
            all_scores += detection.scores
        return all_scores

    @property
    def number_of_detections(self):
        """Number of detections stored in the results object."""
        return len(self.detections)

    @property
    def detection_ids(self):
        """List of detection IDs."""
        return range(self.number_of_detections-1)

    # Methods
    def append_detection(self, detection):
        """Append a detection to the Results object.

        :param detection: Detection object that is going to be appended.
        :return: nothing
        """
        self.detections += [detection]

    def filter_by_minimum_score(self, minimum_score, verbose=False):
        """Filter results based on their score.

        :param minimum_score: Minimum allowed score.
        :param verbose: If True, then the number of filtered instances is printed (default: False).
        :return: nothing
        """

        for detection in self.detections:
            detection.filter_by_minimum_score(minimum_score, verbose=verbose)

    def filter_by_class(self, class_id_to_keep, verbose=False):
        """Filter results based on the class of the detections.

        :param class_id_to_keep: Allowed class ID.
        :param verbose: If True, then the number of filtered instances is printed (default: False).
        :return: nothing
        """

        for detection in self.detections:
            detection.filter_by_class(class_id_to_keep, verbose=verbose)

    def filter_by_minimum_area(self, minimum_area, verbose=False):
        """Filter results based on a threshold for the minimum area of a detection.

        :param minimum_area: Minimum allowed area.
        :param verbose: If True, then the number of filtered instances is printed (default: False).
        :return: nothing
        """

        for detection in self.detections:
            detection.filter_by_minimum_area(minimum_area, verbose=verbose)

    def filter_by_maximum_area(self, maximum_area, verbose=False):
        """Filter results based on a threshold for the maximum area of an instance.

        :param maximum_area: Maximum allowed area.
        :param verbose: If True, then the number of filtered instances is printed (default: False).
        :return: nothing
        """
        for detection in self.detections:
            detection.filter_by_maximum_area(maximum_area, verbose=verbose)

    def filter_by_minimum_circularity(self, minimum_circularity, verbose=False):
        """Filter results based on circularity of the detected instances.

        :param minimum_circularity: Minimum allowed circularity.
        :param verbose: If True, then the number of filtered instances is printed (default: False).
        :return: nothing
        """

        for detection in self.detections:
            detection.filter_by_minimum_circularity(minimum_circularity, verbose=verbose)

    def clear_border_objects(self, verbose=False):
        """Remove instances that touch the border of an image from the results.

        :param verbose: If True, then the number of filtered instances is printed (default: False).
        :return: nothing
        """

        for detection in self.detections:
            detection.clear_border_objects(verbose=verbose)

    def to_size_distribution(self, measurand):
        """Convert the results to a particle size distribution, based on a certain measurand.

        :param measurand: Measurand to use for the conversion:
                          "equivalent_diameter"
                          "equivalent_diameter_convex"
                          "major_bbox_side_length"
                          "major_axis_length"
                          "maximum_feret_diameter"
        :return: A SizeDistribution object.
        """

        measurand = measurand.lower()

        # Check inputs.
        assert measurand in \
            ["equivalent_diameter", "equivalent_diameter_convex", "major_bbox_side_length", "major_axis_length", "maximum_feret_diameter"], \
            "Expected measurand to be one of the following: " \
            "equivalent_diameter, " \
            "equivalent_diameter_convex, " \
            "major_bbox_side_length, " \
            "major_axis_length, " \
            "maximum_feret_diameter"

        # Analyze region properties of the masks, if necessary.
        if measurand is not "maximum_feret_diameter":
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
        elif measurand == "maximum_feret_diameter":
            measurements = get_maximum_feret_diameter(self.masks)

        # Create and return a SizeDistribution-object.
        size_distribution = SizeDistribution("px")
        size_distribution.sizes = np.asarray(measurements)

        return size_distribution

    def display_detection_image(self, detection_id):
        """Display an image with overlayed detections.

        :param detection_id: ID of the detection to display.
        :return: nothing
        """
        
        self.detections[detection_id].display_detection_image()

    def save_detection_image(self, detection_id, output_path, do_display_detections=False):
        """Save an image with overlayed detections.

        :param detection_id: ID of the detection to save.
        :param output_path:
        :param do_display_detections: Display detection before saving it (default: False).
        :return: nothing
        """

        self.detections[detection_id].save_detection_image(output_path, do_display_detections=do_display_detections)

    def save_all_detection_images(self, output_folder, filename_prefix="", filetype="png", do_display_detections=False):
        """Save images with overlayed detections for all images of the Results object.

        :param output_folder: Folder, where the detection images will be saved.
        :param filename_prefix: Prefix for the filename (default: "")
        :param filetype: Filetype to use (default: "png")
        :param do_display_detections: Display detections before saving them (default: False).
        :return: nothing
        """

        for detection_id in self.detection_ids:
            filename = filename_prefix+"_detection_{:d}.".format(detection_id)+filetype
            output_path = os.path.join(output_folder, filename)
            self.detections[detection_id].save_detection_image(output_path, do_display_detections=do_display_detections)
