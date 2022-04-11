import copy
import numpy as np
import scipy.constants


class ExperimentConstants:

    def __init__(self):

        self.Z0 = 120.0 * np.pi  # Ohms
        self.ZL = 50.0  # Ohms
        self.c = scipy.constants.c  # m/s

        self.fs = 2.5e9  # GSa/s
        
        self.time_offset = (35.55e-6 - 34.59e-6)  # s
        self.ice_att = 0.0  # dBm
        self.air_att = 46.0  # dBm

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
        
        self.m_R_low = np.power(10.0, -20.0/10.0)
        self.m_R_high = np.power(10.0, 0.0/10.0)


class Experiment:


    def __init__(self, exper_constants, ice_file_name, air_file_name):

        self.exper_constants = exper_constants
        self.gb_time = np.arange(self.exper_constants.gb_duration * self.exper_constants.fs) / self.exper_constants.fs
        self.gb_freq = np.fft.rfftfreq(len(self.gb_time), 1.0 / self.exper_constants.fs)
        
        self.load_file(ice_file_name)
        self.load_file(air_file_name, air=True)

        self.Pxx_air = np.abs(np.square(np.fft.rfft(self.air_trace)))

        noise = self.ice_trace[np.logical_and(self.ice_time > self.exper_constants.noise_start,
                                              self.ice_time < self.exper_constants.noise_end)]
        
        Pxx_noise = np.abs(np.square(np.fft.rfft(noise)))
        f_noise = scipy.interpolate.interp1d(np.fft.rfftfreq(len(noise),
                                                             1.0 / self.exper_constants.fs),
                                             Pxx_noise,
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value=0.0)
        self.Pxx_noise = f_noise(self.gb_freq)
        self.Pxx_noise /= (self.exper_constants.noise_end - self.exper_constants.noise_start)
        
        self.prepare_ice_signal()
        

    def mc_set_ice_properties(self):
        self.R = np.random.uniform(np.log10(self.exper_constants.m_R_low),
                                   np.log10(self.exper_constants.m_R_high))
        self.R = np.power(10.0, self.R)        
        self.focusing_factor = np.random.normal(self.exper_constants.m_ff,
                                                 self.exper_constants.m_ff_uncert)
        self.ice_prop = np.random.normal(self.exper_constants.m_depth,
                                         self.exper_constants.m_depth_uncert)
        self.air_prop = np.random.normal(self.exper_constants.m_air_prop,
                                         self.exper_constants.m_air_prop_uncert)
        self.T_ratio = np.random.normal(self.exper_constants.m_T_ratio,
                                        self.exper_constants.m_T_ratio_uncert)


    def load_file(self, file_name, air=False):
        """
        Load the pickle file, and adjust the time 
        and amplitude to adjust detector effects and 
        correct for attenuators. 
        """
    
        try:
            ice = np.load(file_name)
        except FileNotFoundError:
            print("File not found: %s" % file_name)
            raise

        t = ice["template_time"]
        trace = ice["template_trace"]

        t += self.exper_constants.time_offset
        if(air):
            trace *= np.power(10.0, self.exper_constants.air_att / 20.0)

            f = scipy.interpolate.interp1d(t,
                                           trace,
                                           kind='linear',
                                           bounds_error=False,
                                           fill_value=0.0)

            trace = np.array(f(self.gb_time - t[0]))
            t = copy.deepcopy(self.gb_time - t[0])
        
            self.air_time = t
            self.air_trace = trace
        else:
            trace *= np.power(10.0, self.exper_constants.ice_att / 20.0)
            self.ice_time = t
            self.ice_trace = trace


    def prepare_ice_signal(self):

        gb_select = np.logical_and(self.ice_time > self.exper_constants.gb_start,
                                   self.ice_time < self.exper_constants.gb_end)

        window_mine = scipy.signal.windows.tukey(np.sum(gb_select),
                                                 alpha=0.25)

        ground_bounce = window_mine * self.ice_trace[gb_select]
        ground_bounce = np.append(ground_bounce,
                                  np.zeros(len(self.gb_time) - len(ground_bounce)))
        self.Pxx_ice = np.abs(np.square(np.fft.rfft(ground_bounce)))

        # Subtract the noise power from the ice signal.
        self.Pxx_ice -= (self.Pxx_noise) * (self.exper_constants.gb_end - self.exper_constants.gb_start)

        # If the power in the ice signal goes below zero
        # due to noise fluctuating high, set the ice signal
        # to a very small value.
        self.Pxx_ice[self.Pxx_ice < 0.0] = 1e-10


    def calculate_att(self):

        # Calculates the attenuation.
        power_ratio = (np.sqrt(self.Pxx_air) * self.air_prop) / (np.sqrt(self.Pxx_ice) * self.ice_prop)
        corrections = self.T_ratio * np.sqrt(self.R * self.focusing_factor)
        att = self.ice_prop / np.log(corrections * power_ratio)
        
        att[self.Pxx_ice == 1e-10] = 0.0
        att[self.Pxx_air <= 0.0] = 1e10

        return att

