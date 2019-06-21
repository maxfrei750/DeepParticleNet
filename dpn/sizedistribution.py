import numpy as np
from scipy.stats.mstats import gmean
from dpn.storable import Storable


class SizeDistribution(Storable):
    def __init__(self, unit):
        self.sizes = np.uint32([])
        unit = unit.lower()

        # Check input.
        assert unit in ["px", "m"], "Expected unit to be \"px\" or \"m\"."
        self.unit = unit

    # Dependant attributes
    @property
    def geometric_mean(self):
        return gmean(self.sizes)

    @property
    def geometric_standard_deviation(self):
        return np.exp(np.std(np.log(self.sizes)))

    @property
    def number_of_particles(self):
        return len(self.sizes)

    # Methods
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

    def compare(self, ground_truth, do_return_errors=False, do_print_output=True):
        # Assert that the size distributions have the same unit.
        assert self.unit == ground_truth.unit, "You cannot concatenate sizedistributions with different units."

        d_g = self.geometric_mean
        s_g = self.geometric_standard_deviation
        N = self.number_of_particles

        d_g_gt = ground_truth.geometric_mean
        s_g_gt = ground_truth.geometric_standard_deviation
        N_gt = ground_truth.number_of_particles

        error_d_g = d_g / d_g_gt - 1
        error_s_g = s_g / s_g_gt - 1
        error_N = N / N_gt - 1

        if do_print_output:
            print("d_g = {:.3f}".format(d_g))
            print("d_g_gt = {:.3f}".format(d_g_gt))
            print("error_d_g = {:.3f}".format(error_d_g))
            print("\n")
            print("s_g = {:.3f}".format(s_g))
            print("s_g_gt = {:.3f}".format(s_g_gt))
            print("error_s_g = {:.3f}".format(error_s_g))
            print("\n")
            print("N = {:.0f}".format(N))
            print("N_gt = {:.0f}".format(N_gt))
            print("error_N = {:.3f}".format(error_N))

        if do_return_errors:
            return error_d_g, error_s_g, error_N
