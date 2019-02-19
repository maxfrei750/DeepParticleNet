import dill


class Storable:
    def save(self, output_path):
        with open(output_path, "wb") as file:
            dill.dump(self, file)

    @staticmethod
    def load(input_path):
        with open(input_path, "rb") as file:
            return dill.load(file)
