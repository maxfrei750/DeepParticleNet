from mrcnn.config import Config as MrcnnConfig
from dpn.storable import Storable
import os
import pathlib


class Config(MrcnnConfig, Storable):
    CUSTOM_CALLBACKS = []
    NO_AUGMENTATION_SOURCES = None
    AUGMENTATION = None
    DATASET_PATH = ""
    NUM_CLASSES = 1 + 2  # Background + sphere + cube

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
