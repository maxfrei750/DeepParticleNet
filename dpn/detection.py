from dpn.visualize import display_instance_outlines
import numpy as np
from itertools import compress
from skimage.segmentation import clear_border
import os
import pathlib
import matplotlib.pyplot as plt


class Detection:
    def __init__(self, image, masks, class_ids, bboxes, scores):
        self.image = image
        self.masks = masks
        self.class_ids = class_ids
        self.bboxes = bboxes
        self.scores = scores

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

    def clear_border_objects(self):
        # Remove instances that touch the border of the image.
        cleared_masks = [clear_border(mask) for mask in self.masks]
        do_keep = [np.any(cleared_mask) for cleared_mask in cleared_masks]
        self.filter_by_list(do_keep)

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