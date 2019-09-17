import dill


class Storable:
    """Abstract class for storable objects"""

    def save(self, output_path):
        """Save an object to a file.

        :param output_path: Path of the output file.
        :return: nothing
        """
        with open(output_path, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(input_path):
        """Load an object from a file.

        :param input_path: Path of the input file.
        :return: nothing
        """

        dill._dill._reverse_typemap['ClassType'] = type
        with open(input_path, "rb") as file:
            return dill.load(file)
