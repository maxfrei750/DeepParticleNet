import os
import numpy as np
import skimage

from mrcnn.utils import Dataset as MrcnnDataset
from dpn.utilities import *
from dpn.sizedistribution import SizeDistribution


class Dataset(MrcnnDataset):
    # Allow the user to define a class for the dataset, if there is only one.
    MONOCLASS = False  # e.g. MONOCLASS = "sphere"

    def load_dataset(self, dataset_dir, subset, limit=None):
        """Load a subset of a particle dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load, specified by the name of the sub-directory.

        """
        # Add classes.
        # Naming the dataset dataset, and the class particle
        self.add_class("dataset", 1, "sphere")
        self.add_class("dataset", 2, "cube")

        # Which subset?
        # use the data from the specified sub-directory
        subset_dir = subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)

        # Get image ids from directory names
        image_ids = next(os.walk(dataset_dir))[1]

        # Add images
        image_counter = 0

        for image_id in image_ids:
            self.add_image(
                "dataset",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images", "{}.png".format(image_id)))

            # Enforce the limit of the number of images.
            if limit is not None:
                image_counter += 1
                if image_counter >= limit:
                    break

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        masks = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                masks.append(m)
        masks = np.stack(masks, axis=-1)

        # Check if the dataset has only one class.
        if self.MONOCLASS:
            number_of_masks = masks.shape[2]
            annotations = [self.MONOCLASS]*number_of_masks
        else:
            # Get annotations.
            annotations = self.get_annotations(image_id)

        # Convert annotations to array of class IDs.
        class_id_array = self.map_classname_id(annotations)

        # Return mask, and array of class IDs of each instance.
        return masks, class_id_array

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "dataset":
            return info["id"]
        else:
            super(Dataset, self).image_reference(image_id)

    def map_classname_id(self, classname_list):
        """Converts a list of class names to a list of class ID."""

        dictionary = dict(zip(self.class_names, self.class_ids))
        id_array = np.array(list(map(dictionary.get, classname_list)), dtype=np.int32)

        return id_array

    def get_annotations(self, image_id):
        """Retrieve annotations for an image."""

        info = self.image_info[image_id]

        # Get annotations file path.
        annotation_path = os.path.join(
            os.path.dirname(os.path.dirname(info['path'])),
            "annotations.txt")

        # Read annotations.
        with open(annotation_path) as f:
            annotations = f.read().splitlines()

        return annotations

    def get_particlesizedistribution_gt(self, mode="mask"):
        """Retrieve the PSD of the dataset, assuming spherical particles."""

        mode = mode.lower()

        # Check inputs.
        assert mode in ["mask", "bbox"], "Expected mode to be \"mask\" or \"bbox\"."

        # Initialize areas array.
        diameters = np.uint32([])

        # Iterate all images
        for image_id in self.image_ids:

            # Load the masks of the current image.
            (masks, _) = self.load_mask(image_id)

            if mode == "mask":

                # Calculate areas as the sum of the pixels of the masks.
                areas = masks.sum(axis=(0, 1))

                # Calculate new diameters
                diameters_new = calculate_equivalent_diameter(areas)

            else:  # elif mode == "bbox":
                bboxes = mrcnn.utils.extract_bboxes(masks)

                # Keep only the largest bbox dimension.
                diameters_new = get_largest_bbox_dimension(bboxes)

            # Concatenate diameters_new to the diameters array.
            diameters = np.concatenate((diameters, diameters_new))

        # Create and return a SizeDistribution-object.
        psd = SizeDistribution("px")
        psd.diameters = diameters

        return psd
