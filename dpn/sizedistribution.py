import numpy as np
import scipy


class SizeDistribution(object):
    def __init__(self, unit):
        self.sizes = np.uint32([])
        unit = unit.lower()

        # Check input.
        assert unit in ["px", "m"], "Expected unit to be \"px\" or \"m\"."
        self.unit = unit

    # Dependant attributes
    @property
    def geometric_mean(self):
        return scipy.stats.mstats.gmean(self.sizes)

    @property
    def geometric_standard_deviation(self):
        return np.exp(np.std(np.log(self.sizes)))

    @property
    def number_of_particles(self):
        return len(self.sizes)

    # Public methods
    def concatenate(size_distributions):
        # Assert that all the size distributions have the same unit.
        units = [size_distribution.unit for size_distribution in size_distributions]
        assert all(x == units[0] for x in units), "You cannot concatenate sizedistributions with different units."

        # Extract the diameter arrays.
        size_arrays = [psd.diameters for psd in size_distributions]

        size_distribution_new = SizeDistribution(units[0])

        size_distribution_new.sizes = np.concatenate(size_arrays)

        return size_distribution_new

    def to_meter(self, scalingfactor_meterperpixel):
        self.sizes = self.sizes * scalingfactor_meterperpixel
        self.unit = "m"

    def to_pixel(self, scalingfactor_meterperpixel):
        self.sizes = self.sizes / scalingfactor_meterperpixel
        self.unit = "px"
