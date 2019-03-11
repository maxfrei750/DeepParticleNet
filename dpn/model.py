from mrcnn.model import MaskRCNN
from dpn.results import Results
from dpn.detection import Detection
import numpy as np
from keras.callbacks import CSVLogger
import os
import inspect


class Model(MaskRCNN):
    def __init__(self, mode, config, model_dir):
        super().__init__(mode, config, model_dir)

        # If the user set USE_PRETRAINED_WEIGHTS in the config, then try to load a preset weight set. If that fails,
        # then try to load the weights from a file that the user may have specified.
        if config.USE_PRETRAINED_WEIGHTS is not None:
            print("Using pretrained weights: {}".format(config.USE_PRETRAINED_WEIGHTS))

            try:
                self.load_pretrained_weights(config.USE_PRETRAINED_WEIGHTS)
            except AssertionError:
                self.load_weights(config.USE_PRETRAINED_WEIGHTS, by_name=True)

    def train(self, dataset_train, dataset_val):
        # Save config in the log dir.
        self.config.save(self.log_dir)

        # Append a CSVLogger to the custom callbacks by default.
        csv_path = os.path.join(self.log_dir, self.config.NAME.lower()+"_training.csv")
        csv_logger = CSVLogger(csv_path, append=True)
        self.config.CUSTOM_CALLBACKS.append(csv_logger)

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
        results_dict_list = super().detect([image], verbose=verbose)
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

    def load_pretrained_weights(self, weight_name, verbose=False):
        module_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        model_dir = os.path.join(module_dir, "..", "models")

        weight_name = weight_name.lower()
        expected_weight_names = ["resnet50_coco", "resnet101_coco", "resnet50_imagenet"]

        assert weight_name in expected_weight_names, \
            "Expected weight_name to be one of the following: {}.".format(expected_weight_names)

        if weight_name == "resnet50_coco":
            weight_name = "resnet101_coco"
            if verbose:
                print("Using resnet101_coco weights file but loading only the resnet50 part.")

        weight_path = os.path.abspath(os.path.join(model_dir, weight_name+".h5"))

        if verbose:
            print("Loading weights from: "+weight_path)

        # Exclude the last layers because they require a matching number of classes.
        self.load_weights(
            weight_path,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        )
