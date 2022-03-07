import numpy as np
import scipy.constants


class Experiment:

    def __init__(self):

        self.Z0 = 120.0 * np.pi  # Ohms
        self.ZL = 50.0  # Ohms
        self.c = scipy.constants.c  # m/s

        self.time_offset = (35.55e-6 - 34.59e-6)  # s
        self.ice_att = 0.0  # dBm
        self.air_att = 46.0  # dBm

        # t0 = 35.55
        # t1 = 36.05
        # R E (0.1, 1.0) uniform in log
        # focusing_factor = 1.61 +/- 0.24
        # ice_prop = 6008.0 +/- 100.0
        # air_prop = 244.0 +/- 1.0
        # T_Ratio = 1.05 +/- 0.05

        self.gb_start = 35.55e-6  # Seconds
        self.gb_end = 36.05e-6  # Seconds
        self.noise_start = 22.0e-6  # Seconds
        self.noise_end = 34.0e-6  # Seconds

        self.gb_duration = 10e-6  # Seconds

        self.m_depth = 6008.0  # m
        self.m_depth_uncert = 100.0  # m

        self.m_ff = 1.61
        self.m_ff_uncert = 0.24

        self.m_air_prop = 244.0  # m
        self.m_air_prop_uncert = 1.0  # m

        self.m_T_ratio = 1.0
        self.m_T_ratio_uncert = 0.05

        self.m_R_low = 0.1
        self.m_R_high = 1.0
