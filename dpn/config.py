from mrcnn.config import Config as MrcnnConfig
from dpn.storable import Storable
import os
import pathlib


class Config(MrcnnConfig, Storable):
    # Dataset
    AUGMENTATION = None
    NO_AUGMENTATION_SOURCES = None
    DATASET_PATH = None
    NUMBER_OF_SAMPLES_TRAIN = 100
    NUMBER_OF_SAMPLES_VAL = 10

    # Architecture
    DETECTION_MAX_INSTANCES = 100
    NUM_CLASSES = 1 + 2  # Background + sphere + cube
    USE_PRETRAINED_WEIGHTS = None

    # Training
    CUSTOM_CALLBACKS = []
    MAX_GT_INSTANCES = 200
    LAYERS = "all"
    LEARNING_RATE = 0.01
    EPOCHS = 10000

    # Method to save the config.
    def save(self, directory):
        """Save Configuration values."""

        # Gather attributes.
        attributes = []

        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                attributes.append("{:30} {}".format(a, getattr(self, a)))

        # Create directory if it does not exist yet.
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        # Write attributes to text file.
        with open(os.path.join(directory, "config.txt"), "w") as file:
            for attribute in attributes:
                file.write("{}\n".format(attribute))

        # Pickle config.
        super().save(os.path.join(directory, "config.pkl"))

    # Method to load a config.
    @staticmethod
    def load(directory):
        """Load Configuration values."""

        # Load pickled config.
        return super(Config, Config).load(os.path.join(directory, "config.pkl"))
