import numpy as np
import scipy


class SizeDistribution(object):
    diameters = np.uint32([])

    def __init__(self, unit):
        unit = unit.lower()

        # Check input.
        assert unit in ["px", "m"], "Expected unit to be \"px\" or \"m\"."
        self.unit = unit

    # Dependant attributes
    @property
    def geometric_mean(self):
        return scipy.stats.mstats.gmean(self.diameters)

    @property
    def geometric_standard_deviation(self):
        return np.exp(np.std(np.log(self.diameters)))

    @property
    def number_of_particles(self):
        return self.diameters.size

    # Public methods
    def concatenate(sizedistributions):
        # Assert that all the PSDs have the same unit.
        units = [psd.unit for psd in sizedistributions]
        assert all(x == units[0] for x in units), "You cannot concatenate PSDs with different units."

        # Extract the diameter arrays.
        diameter_arrays = [psd.diameters for psd in sizedistributions]

        psd_new = SizeDistribution(units[0])

        psd_new.diameters = np.concatenate(diameter_arrays)

        return psd_new

    def to_meter(self, scalingfactor_meterperpixel):
        self.diameters = self.diameters * scalingfactor_meterperpixel
        self.unit = "m"

        return self

    def to_pixel(self, scalingfactor_meterperpixel):
        self.diameters = self.diameters / scalingfactor_meterperpixel
        self.unit = "px"

        return self