import numpy as np

from mrcnn.model import MaskRCNN
from dpn.sizedistribution import SizeDistribution
from dpn.utilities import *

from skimage.segmentation import clear_border


class Model(MaskRCNN):
    def train(self, dataset_train, dataset_val):
        # Save config in the log dir.
        self.config.save(self.log_dir)

        # Call the training method of the super class.
        super(Model, self).train(dataset_train, dataset_val,
                                 learning_rate=self.config.LEARNING_RATE,
                                 epochs=self.config.EPOCHS,
                                 layers=self.config.LAYERS,
                                 augmentation=self.config.AUGMENTATION,
                                 custom_callbacks=self.config.CUSTOM_CALLBACKS,
                                 no_augmentation_sources=self.config.NO_AUGMENTATION_SOURCES)

    def analyze_dataset(self, dataset, mode="mask", filter_border_objects=False):

        mode = mode.lower()

        # Check inputs.
        assert mode in ["mask", "bbox"], "Expected mode to be \"mask\" or \"bbox\"."

        # Initialize diameters array.
        diameters = np.uint32([])

        for image_id in dataset.image_ids:
            image = dataset.load_image(image_id)

            # Run object detection
            results = self.detect([image])
            results = results[0]

            # Extract masks and bounding boxes.
            masks = results["masks"]
            bboxes = results["rois"]

            if filter_border_objects:
                # Filter objects that touch the image border.
                cleared_masks = clear_border(masks)
                do_keep = np.any(cleared_masks, (0, 1))

                masks = masks[:, :, do_keep]
                bboxes = bboxes[do_keep, :]

            if mode == "mask":
                # Calculate areas as the sum of the pixels of the masks.
                areas = masks.sum(axis=(0, 1))

                # Calculate new diameters
                diameters_new = calculate_equivalent_diameter(areas)
            else:  # elif mode == "bbox":
                # Keep only the largest bbox dimension.
                diameters_new = get_largest_bbox_dimension(bboxes)

            # Concatenate diameters_new to the diameters array.
            diameters = np.concatenate((diameters, diameters_new))

        # Create and return a SizeDistribution-object.
        psd = SizeDistribution("px")
        psd.diameters = diameters

        return psd
