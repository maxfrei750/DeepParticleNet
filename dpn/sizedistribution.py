import numpy as np
import scipy


class SizeDistribution(object):
    sizes = np.uint32([])

    def __init__(self, unit):
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
        return self.sizes.size

    # Public methods
    def concatenate(sizedistributions):
        # Assert that all the PSDs have the same unit.
        units = [psd.unit for psd in sizedistributions]
        assert all(x == units[0] for x in units), "You cannot concatenate PSDs with different units."

        # Extract the diameter arrays.
        size_arrays = [psd.diameters for psd in size_distributions]

        psd_new = SizeDistribution(units[0])

        psd_new.diameters = np.concatenate(diameter_arrays)

        return psd_new

    def to_meter(self, scalingfactor_meterperpixel):
        self.sizes = self.sizes * scalingfactor_meterperpixel
        self.unit = "m"

        return self

    def to_pixel(self, scalingfactor_meterperpixel):
        self.sizes = self.sizes / scalingfactor_meterperpixel
        self.unit = "px"

        return self