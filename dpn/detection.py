from dpn.visualize import display_instance_outlines
import numpy as np
from itertools import compress
from skimage.segmentation import clear_border
from skimage.morphology import binary_erosion
import matplotlib.pyplot as plt
from .storable import Storable


class Detection(Storable):
    def __init__(self, image, masks, class_ids, bboxes, scores, data_set=None, image_file_name=None, comment=None):
        self.image = image
        self.masks = masks
        self.class_ids = class_ids
        self.bboxes = bboxes
        self.scores = scores
        self.data_set = data_set
        self.image_file_name = image_file_name
        self.comment = comment

    # Dependant properties
    @property
    def areas(self):
        return [np.sum(mask) for mask in self.masks]

    @property
    def perimeters(self):
        # Calculate the perimeter as the sum of the pixels in the outline of a mask.
        # The outline is retrieved by xor-ing mask with an erosion of itself.
        return [np.sum(mask ^ binary_erosion(mask)) for mask in self.masks]

    @property
    def number_of_instances(self):
        return len(self.masks)

    # Methods
    def display_detection_image(self,
                                do_return_figure_handle=False,
                                linewidth=1.5,
                                alpha=1,
                                dpi=100):
        figure_handle = display_instance_outlines(self.image,
                                                  self.masks,
                                                  linewidth=linewidth,
                                                  alpha=alpha,
                                                  dpi=dpi)
        if do_return_figure_handle:
            return figure_handle

    def save_detection_image(self, output_path, do_display_detections=False):
        figure_handle = self.display_detection_image(do_return_figure_handle=True)
        figure_handle.savefig(output_path, dpi=100)

        # Close figure if it is not required.
        if not do_display_detections:
            plt.close(figure_handle)

    def clear_border_objects(self, verbose=False):
        # Remove instances that touch the border of the image.
        cleared_masks = [clear_border(mask) for mask in self.masks]
        do_keep = [np.any(cleared_mask) for cleared_mask in cleared_masks]
        self.filter_by_list(do_keep, verbose=verbose)

    def filter_by_list(self, do_keep, verbose=False):
        if verbose:
            number_of_kept_instances = np.sum(do_keep)
            number_of_filtered_instances = self.number_of_instances - number_of_kept_instances
            print(
                "Filtered {} of {} instances (~{:.1f} %).".format(
                    number_of_filtered_instances,
                    self.number_of_instances,
                    number_of_filtered_instances/self.number_of_instances*100))

        self.masks = list(compress(self.masks, do_keep))
        self.class_ids = list(compress(self.class_ids, do_keep))
        self.bboxes = list(compress(self.bboxes, do_keep))
        self.scores = list(compress(self.scores, do_keep))

    def filter_by_minimum_score(self, minimum_score, verbose=False):
        # Remove instances with scores smaller then the given minimum score.
        do_keep = [score >= minimum_score for score in self.scores]
        self.filter_by_list(do_keep, verbose=verbose)

    def filter_by_class(self, class_id_to_keep, verbose=False):
        # Remove instances with a class other than the given class.
        do_keep = [class_id is class_id_to_keep for class_id in self.class_ids]
        self.filter_by_list(do_keep, verbose=verbose)

    def filter_by_minimum_area(self, minimum_area, verbose=False):
        # Remove instances with areas smaller than the given minimum area.
        do_keep = [area >= minimum_area for area in self.areas]
        self.filter_by_list(do_keep, verbose=verbose)

    def filter_by_maximum_area(self, maximum_area, verbose=False):
        # Remove instances with areas larger than the given maximum area.
        do_keep = [area <= maximum_area for area in self.areas]
        self.filter_by_list(do_keep, verbose=verbose)

    def filter_by_minimum_circularity(self, minimum_circularity, verbose=False):
        # Remove instances with circularities smaller than the given minimum circularity.

        areas = np.asarray(self.areas)
        perimeters = np.asarray(self.perimeters)

        circularities = 4 * np.pi * areas / perimeters**2

        circularities = circularities.tolist()

        do_keep = [circularity >= minimum_circularity for circularity in circularities]
        self.filter_by_list(do_keep, verbose=verbose)
