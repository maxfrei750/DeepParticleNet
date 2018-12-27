from mrcnn.config import Config as MrcnnConfig
import dill
import os


class Config(MrcnnConfig):
    # Image augmenter
    AUGMENTATION = None

    # Path to the dataset
    DATASET_PATH = ""

    # Method to save the config.
    def save(self, directory):
        """Save Configuration values."""

        # Gather attributes.
        attributes = []

        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                attributes.append("{:30} {}".format(a, getattr(self, a)))

        # Create directory if it does not exist yet.
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Write attributes to text file.
        with open(os.path.join(directory, 'config.txt'), 'w') as hConfigFile_text:
            for listitem in attributes:
                hConfigFile_text.write('%s\n' % listitem)

        # Pickle config.
        with open(os.path.join(directory, 'config.pkl'), 'wb') as hConfigFile_pickle:
            dill.dump(self, hConfigFile_pickle)

    # Method to load a config.
    @staticmethod
    def load(directory):
        """Load Configuration values."""

        # Pickle config.
        with open(os.path.join(directory, 'config.pkl'), 'rb') as hConfigFile_pickle:
            pickeled_config = dill.load(hConfigFile_pickle)

        return pickeled_config
