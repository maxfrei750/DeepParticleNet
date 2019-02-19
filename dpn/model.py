from mrcnn.model import MaskRCNN
from dpn.results import Results
from dpn.detection import Detection
import numpy as np


class Model(MaskRCNN):
    def train(self, dataset_train, dataset_val):
        # Save config in the log dir.
        self.config.save(self.log_dir)

        # Call the training method of the super class.
        super().train(dataset_train, dataset_val,
                      learning_rate=self.config.LEARNING_RATE,
                      epochs=self.config.EPOCHS,
                      layers=self.config.LAYERS,
                      augmentation=self.config.AUGMENTATION,
                      custom_callbacks=self.config.CUSTOM_CALLBACKS,
                      no_augmentation_sources=self.config.NO_AUGMENTATION_SOURCES)

    def detect(self, image, verbose=0):
        # Call the training method of the super class.
        results_dict_list = super(Model, self).detect([image], verbose=verbose)
        results_dict = results_dict_list[0]

        # Extract properties from results_dict.
        masks = results_dict["masks"]
        class_ids = results_dict["class_ids"]
        bboxes = results_dict["rois"]
        scores = results_dict["scores"]

        # Convert class_ids to list.
        class_ids = class_ids.tolist()

        # Convert bboxes to list.
        bboxes = bboxes.tolist()

        # Convert scores to list.
        scores = scores.tolist()

        # Get number of instances
        number_of_instances = len(class_ids)

        # Convert masks to list of masks.
        masks = np.split(masks, number_of_instances, axis=2)
        masks = [np.squeeze(mask) for mask in masks]

        return Detection(image, masks, class_ids, bboxes, scores)

    def analyze_dataset(self, dataset):
        # Create a Results-object.
        results = Results()

        for image_id in dataset.image_ids:

            # Load image.
            image = dataset.load_image(image_id)

            # Perform detection.
            new_detection = self.detect(image)

            # Append results.
            results.append_detection(new_detection)

        return results
