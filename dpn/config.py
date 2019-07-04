from mrcnn.config import Config as MrcnnConfig
from dpn.storable import Storable
import os
import pathlib


class Config(MrcnnConfig, Storable):
    """Class to store DPN configurations."""

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

    def __init__(self):
        """Create and initialize a configuration."""

        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Calculate number of training steps per epoch, so that all trainings samples are used once per epoch.
        self.STEPS_PER_EPOCH = round(self.NUMBER_OF_SAMPLES_TRAIN/self.BATCH_SIZE)
        # Calculate number of validation steps per epoch, so that all validation samples are used once per epoch.
        self.VALIDATION_STEPS = round(self.NUMBER_OF_SAMPLES_VAL/self.BATCH_SIZE)
        
        if self.VALIDATION_STEPS < 1:
            self.VALIDATION_STEPS = 1

        super().__init__()

    def save(self, directory):
        """Save the configuration.

        :param directory: Path of the directory to store the configuration.
        :return: nothing
        """

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

    @staticmethod
    def load(directory):
        """Load configuration.

        :param directory: Directory to load the configuration from.
        :return: Loaded configuration object.
        """

        # Load pickled config.
        return super(Config, Config).load(os.path.join(directory, "config.pkl"))
