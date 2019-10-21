from mrcnn.model import MaskRCNN
from dpn.results import Results
from dpn.detection import Detection
import numpy as np
from keras.callbacks import CSVLogger, TerminateOnNaN
import os
import inspect
import urllib.request
import shutil
from pathlib import Path


class Model(MaskRCNN):
    """Offers MaskRCNN models that can be trained and used for the detection of primary particles."""

    def __init__(self, mode, config, model_dir):
        """Create and initialize a model.

        :param mode: Can be either "training" or "inference".
        :param config: Config object, storing the config of the model.
        :param model_dir: Directory, where the config, training logs and trained weights of the model are saved.
        """

        # Create MaskRCNN model.
        super().__init__(mode, config, model_dir)

        # If the user set USE_PRETRAINED_WEIGHTS in the config, then try to load a preset weight set. If that fails,
        # then try to load the weights from a file that the user may have specified.
        if config.USE_PRETRAINED_WEIGHTS is not None:
            print("Using pretrained weights: {}".format(config.USE_PRETRAINED_WEIGHTS))

            try:
                self.load_pretrained_weights(config.USE_PRETRAINED_WEIGHTS)
            except AssertionError:
                self.load_weights(config.USE_PRETRAINED_WEIGHTS, by_name=True)

    def train(self, dataset_train, dataset_val, save_best_only=False):
        """ Method to train the model.
        
        :param dataset_train: Dataset object storing the training data.
        :param dataset_val: Dataset object storing the validation data.
        :param save_best_only: When true, only the best models are saved in the log directory.
        :return: Training history.
        """

        # Save config in the log dir.
        self.config.save(self.log_dir)

        # Append a CSVLogger to the custom callbacks by default.
        csv_path = os.path.join(self.log_dir, self.config.NAME.lower()+"_training.csv")
        csv_logger = CSVLogger(csv_path, append=True)
        self.config.CUSTOM_CALLBACKS.append(csv_logger)

        # Append a TerminateOnNaN callback to the custom callbacks by default.
        nan_terminator = TerminateOnNaN()
        self.config.CUSTOM_CALLBACKS.append(nan_terminator)

        # Call the training method of the super class.
        history = super().train(dataset_train, dataset_val,
                                learning_rate=self.config.LEARNING_RATE,
                                epochs=self.config.EPOCHS,
                                layers=self.config.LAYERS,
                                augmentation=self.config.AUGMENTATION,
                                custom_callbacks=self.config.CUSTOM_CALLBACKS,
                                no_augmentation_sources=self.config.NO_AUGMENTATION_SOURCES,
                                save_best_only=save_best_only,
                                monitored_quantity='val_loss')

        return history

    def detect(self, image, verbose=0):
        """ Find primary particles on an image.

        :param image: Input image.
        :param verbose: Verbose mode.
        :return: Detection object that stores the bounding boxes, masks and classes of the detected primary particles.
        """

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

        if number_of_instances > 0:
            # Convert masks to list of masks.
            masks = np.split(masks, number_of_instances, axis=2)
            masks = [np.squeeze(mask) for mask in masks]

        return Detection(image, masks, class_ids, bboxes, scores)

    def analyze_dataset(self, dataset):
        """ Analyze a complete set of images.

        :param dataset: Dataset object that stores the images to be analyzed.
        :return: List of Detection objects.
        """

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

    def get_pretrained_model_dir(self):
        """Get path of the directory, where the pretrained models are stored.

        :return: Path of the directory storing the pretrained models.
        """

        module_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        model_dir = os.path.join(module_dir, "..", "pretrained_models")

        return model_dir     

    def download_pretrained_weights(self, weight_name, verbose=True):
        """Download a set of pretrained weights.

        :param weight_name: Name of the pretrained weight set to download.
        :param verbose: Verbosity mode.
        :return: nothing
        """
        model_base_url = "https://github.com/maxfrei750/DeepParticleNet/releases/download/v1.0/"

        # Get directory of to store the pretrained model.
        model_dir = self.get_pretrained_model_dir()

        # Input checking.
        weight_name = weight_name.lower()
        expected_weight_names = ["resnet101_coco", "resnet50_imagenet", "resnet50_mpac"]

        assert weight_name in expected_weight_names, \
            "Expected weight_name to be one of the following: {}.".format(expected_weight_names)

        weight_path = os.path.abspath(os.path.join(model_dir, weight_name+".h5"))
        weight_url = model_base_url + weight_name+".h5"

        if verbose:
            print("Starting donwload of weights from: "+weight_url)
            print("Saving weights at: "+weight_path)

        # Download the desired weights.
        with urllib.request.urlopen(weight_url) as response, open(weight_path, 'wb') as output:
            shutil.copyfileobj(response, output)

        if verbose:
            print("Download finished.")

    def load_pretrained_weights(self, weight_name, verbose=False):
        """ Load a set of pretrained weights.

        :param weight_name: Name of the set of pretrained weights.
        :param verbose: Verbosity mode.
        :return: nothing
        """

        # Get directory of to store the pretrained model.
        model_dir = self.get_pretrained_model_dir()

        # Input checking.
        weight_name = weight_name.lower()
        expected_weight_names = ["resnet50_coco", "resnet101_coco", "resnet50_imagenet", "resnet50_mpac"]

        assert weight_name in expected_weight_names, \
            "Expected weight_name to be one of the following: {}.".format(expected_weight_names)

        if weight_name == "resnet50_coco":
            weight_name = "resnet101_coco"
            if verbose:
                print("Using resnet101_coco weights file but loading only the resnet50 part.")

        weight_path = os.path.abspath(os.path.join(model_dir, weight_name+".h5"))

        # Download the weightfile if it is not available yet.
        weight_file = Path(weight_path)

        if not weight_file.is_file():
            self.download_pretrained_weights(weight_name)

        if verbose:
            print("Loading weights from: "+weight_path)

        # Exclude the last layers because they require a matching number of classes.
        self.load_weights(
            weight_path,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
        )
