from mrcnn.model import MaskRCNN
from dpn.results import Results


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

    def analyze_dataset(self, dataset):
        # Create a Results-object.
        results = Results()

        for image_id in dataset.image_ids:

            # Load image.
            image = dataset.load_image(image_id)

            # Perform detection.
            new_results_dict = self.detect([image])

            # Append results.
            results.append_from_dicts(new_results_dict)

        return results
