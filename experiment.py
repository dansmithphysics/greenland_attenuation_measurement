import copy
import numpy as np
import scipy.constants
import matplotlib.pyplot as plt
import analysis_funcs


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


class AttenuationModel:

    def __init__(self, depths, h_plus, cl, nh4, sigma_pure, sigma_pure_uc, mus, mus_uc, e_acts, e_acts_uc, temp, T_r, k=8.616e-5):
        self.depths = depths
        self.h_plus = h_plus
        self.cl = cl
        self.nh4 = nh4
        self.sigma_pure = sigma_pure
        self.sigma_pure_uc = sigma_pure_uc
        self.mu_h, self.mu_cl, self.mu_nh4 = mus
        self.mu_h_uc, self.mu_cl_uc, self.mu_nh4_uc = mus_uc
        self.e_act_pure, self.e_act_h, self.e_act_cl, self.e_act_nh4 = e_acts
        self.e_act_pure_uc, self.e_act_h_uc, self.e_act_cl_uc, self.e_act_nh4_uc = e_acts_uc
        self.temp = temp
        self.T_r = T_r
        self.k = k
        self.C_to_K = 273.15

        self.scale = 1.0
        
        self.const_sigma_pure = self.sigma_pure
        self.const_mu_h = self.mu_h
        self.const_mu_cl = self.mu_cl
        self.const_mu_nh4 = self.mu_nh4
        self.const_e_act_pure = self.e_act_pure
        self.const_e_act_h = self.e_act_h
        self.const_e_act_cl = self.e_act_cl
        self.const_e_act_nh4 = self.e_act_nh4

    def reset(self):
        self.sigma_pure = self.const_sigma_pure
        self.mu_h = self.const_mu_h 
        self.mu_cl = self.const_mu_cl 
        self.mu_nh4 = self.const_mu_nh4 
        self.e_act_pure = self.const_e_act_pure
        self.e_act_h = self.const_e_act_h
        self.e_act_cl = self.const_e_act_cl
        self.e_act_nh4 = self.const_e_act_nh4
        
    def toy_mc_values(self):
        self.sigma_pure = np.random.normal(self.const_sigma_pure, self.sigma_pure_uc)
        self.mu_h = np.random.normal(self.const_mu_h, self.mu_h_uc)
        self.mu_cl = np.random.normal(self.const_mu_cl, self.mu_cl_uc)
        self.mu_nh4 = np.random.normal(self.const_mu_nh4, self.mu_nh4_uc)
        
        self.e_act_pure = np.random.normal(self.const_e_act_pure, self.e_act_pure_uc)
        self.e_act_h = np.random.normal(self.const_e_act_h, self.e_act_h_uc)
        self.e_act_cl = np.random.normal(self.const_e_act_cl, self.e_act_cl_uc)
        self.e_act_nh4 = np.random.normal(self.const_e_act_nh4, self.e_act_nh4_uc)
        
    def conductivity(self, z=None):
        z = self.clean_z(z)
        temperature_scale = (1 / (self.T_r + self.C_to_K) - 1 / (self.temp + self.C_to_K)) / self.k
        sigma_infty = self.sigma_pure * np.exp(self.e_act_pure * temperature_scale)
        sigma_infty += self.mu_h * self.h_plus * np.exp(self.e_act_h * temperature_scale)
        sigma_infty += self.mu_cl * self.cl * np.exp(self.e_act_cl * temperature_scale)
        sigma_infty += self.mu_nh4 * self.nh4 * np.exp(self.e_act_nh4 * temperature_scale)
        f_sigma_infty = scipy.interpolate.interp1d(self.depths,
                                                   sigma_infty,
                                                   kind='linear',
                                                   bounds_error=False,
                                                   fill_value=0.0)

        return f_sigma_infty(z)

    def rho(self, z=None):
        """
        Calculated by Deaconu et al. 
        https://arxiv.org/pdf/1805.12576.pdf
        """        
        z = self.clean_z(z)
        rho_ = np.zeros(len(z))
        rho_[z <= 14.9] = 0.917 - 0.594 * np.exp(-z[z <= 14.9] / 30.8)
        rho_[z > 14.9] = 0.917 - 0.367 * np.exp(-(z[z > 14.9] - 14.9)/ 40.5)
        return rho_

    def epsilon_prime_real(self, z=None):
        """
        Derived from A. Kovacs et al. 
        """
        z = self.clean_z(z)
        return np.square(1.0 + 0.854 * self.rho(z))

    def L_alpha(self, z=None):
        z = self.clean_z(z)
        sigma_infty = self.conductivity(z)
        epsilon_prime_ = self.epsilon_prime_real(z)
        L_alpha = self.scale * np.sqrt(epsilon_prime_) / sigma_infty
        return L_alpha

    def N_alpha(self, z=None):
        z = self.clean_z(z)
        N_alpha = 1.0 / self.L_alpha(z)
        N_alpha *= 1000.0 * (10.0 * np.log10(np.e))
        return N_alpha

    def N_alpha_star(self, z=None):
        z = self.clean_z(z)
        N_alpha_star = np.cumsum(self.N_alpha(z)) / np.arange(1, len(z)+1)
        return N_alpha_star

    def clean_z(self, z=None):
        if(z is None):            
            z = self.depths
        else:
            z = np.abs(z)
        return z

    def convert_N_to_L(self, N):
        L_alpha = 1000.0 * (10.0 * np.log10(np.e)) / N
        return L_alpha

    def smooth(self, depths, trace, window_length=4):
        smooth_trace = np.convolve(trace,
                                   np.ones(window_length) / float(window_length),
                                   'valid')
        depths = (depths[(window_length - 1):]
                  + depths[:-(window_length - 1)]) / 2.0
        smooth_trace = smooth_trace[::window_length]
        depths = depths[::window_length]
        return depths, smooth_trace

    def smooth_chemistry(self, window_length=4):
        depths_, self.cl = self.smooth(self.depths,
                                       self.cl,
                                       window_length=window_length)
        depths_, self.h_plus = self.smooth(self.depths,
                                           self.h_plus,
                                           window_length=window_length)
        depths_, self.nh4 = self.smooth(self.depths,
                                        self.nh4,
                                        window_length=window_length)
        depths_, self.temp = self.smooth(self.depths,
                                         self.temp,
                                         window_length=window_length)
        self.depths = depths_
        
    def plot_conductivity(self, label="", color=None, new_figure=True):
        if(new_figure):
            plt.figure()
        plt.plot(self.depths,
                 self.conductivity(),
                 color=color,
                 label=label)
        if(new_figure):
            plt.xlabel("Depth [m]")
            plt.ylabel("Conductivity [uS/m]")
            plt.xlim(0.0, 3004.0)
            plt.ylim(0.0, 45.0)
            plt.grid()

    def plot_chemistry(self, new_figure=True):
        if(new_figure):
            plt.figure()
        plt.plot(self.depths,
                 self.h_plus,
                 label="H+")
        plt.plot(self.depths,
                 self.cl,
                 label="Cl-")
        plt.plot(self.depths,
                 self.nh4,
                 label="NH4+")
        if(new_figure):
            plt.xlabel("Depth [m]")
            plt.ylabel("Molarity [uM]")
            plt.xlim(0.0, 3004.0)
            plt.ylim(0.0, 8.0)
            plt.grid()
            plt.legend()

    def plot_L_alpha(self, new_figure=True):
        if(new_figure):
            plt.figure()
        plt.plot(self.depths,
                 self.L_alpha())
        if(new_figure):
            plt.xlabel("Depth [m]")
            plt.ylabel("Att. Length [m]")
            plt.xlim(0.0, 3004.0)
            plt.ylim(0.0, 2000)
            plt.grid()

    def plot_N_alpha(self, new_figure=True):
        if(new_figure):
            plt.figure()
        plt.plot(self.depths,
                 self.N_alpha_star())
        if(new_figure):
            plt.xlabel("Depth [m]")
            plt.ylabel("Attenuation Rate [dB / km]")
            plt.xlim(0.0, 3004.0)
            plt.grid()

    def plot_avg_L(self, new_figure=True):
        if(new_figure):
            plt.figure()
        plt.plot(self.depths,
                 self.scale * self.convert_N_to_L(self.N_alpha_star()))
        if(new_figure):
            plt.xlabel("Depth [m]")
            plt.ylabel("Average Attenuation Length to Depth [m]")
            plt.xlim(0.0, 3004.0)
            plt.grid()

    def plot_L_alpha_with_errorbars_and_scale(self, measured_att_cv, measured_att_uc, n_throws=100,
                                              new_figure=True, edgecolor="black", alpha=1.0, hatch="//", facecolor="none", label=""):
    

        measured_att = np.random.normal(measured_att_cv,
                                        measured_att_uc,
                                        n_throws)
        
        entries = np.zeros((n_throws, len(self.depths)))

        avg_L_alphas = np.zeros(n_throws)
        for i_throw in range(n_throws):
            self.toy_mc_values()
            target_att = measured_att[i_throw]        
            avg_L_alpha = self.convert_N_to_L(self.N_alpha_star())[-1]
            avg_L_alphas[i_throw] = avg_L_alpha
            self.scale = target_att / avg_L_alpha            
            entries[i_throw] = self.L_alpha()                
            self.scale = 1.0
            
        self.reset()

        middle_val = np.zeros(len(self.depths))
        low_bound = np.zeros(len(self.depths))
        high_bound = np.zeros(len(self.depths))

        for i_depth in range(len(self.depths)):
            entries_ = entries[:, i_depth].flatten()
            entries_, cumsum = analysis_funcs.calculate_uncertainty(entries_)
            
            entries_min, entries_mid, entries_max = analysis_funcs.return_confidence_intervals(entries_, cumsum)

            middle_val[i_depth] = entries_mid
            low_bound[i_depth] = entries_min
            high_bound[i_depth] = entries_max

        x = self.depths
        yerr_min = low_bound
        yerr_max = high_bound
        y = middle_val

        if(new_figure):
            plt.figure(figsize=(5, 4))
        plt.fill_between(x,
                         yerr_min,
                         yerr_max,
                         alpha=alpha,
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         hatch=hatch,
                         label=label)
        if(new_figure):
            plt.xlabel("Depth [m]")
            plt.ylabel("Att. Length [m]")
            plt.xlim(0.0, 3004.0)
            plt.ylim(0.0, 2000.0)
            plt.grid()
            
    def calculate_scale(self, measured_att_cv, measured_att_uc, n_throws=1000):    

        measured_att = np.random.normal(measured_att_cv,
                                        measured_att_uc,
                                        n_throws)
        
        entries = np.zeros((n_throws, len(self.depths)))
        avg_L_alphas = np.zeros(n_throws)
        
        for i_throw in range(n_throws):
            self.toy_mc_values()
            target_att = measured_att[i_throw]        
            avg_L_alpha = self.convert_N_to_L(self.N_alpha_star())[-1]
            self.scale = target_att / avg_L_alpha
            avg_L_alpha = self.convert_N_to_L(self.N_alpha_star(self.depths[self.depths < 1500.0]))[-1]
            avg_L_alphas[i_throw] = avg_L_alpha
            entries[i_throw] = self.L_alpha()                
            self.scale = 1.0
            
        self.reset()

        return np.mean(avg_L_alphas / measured_att), np.std(avg_L_alphas / measured_att)
                    
    def plot_power_trace_lookalike(self, measured_att_cv, measured_att_uc, n_throws=1000,
                                   new_figure=True, edgecolor="black", hatch=None, label=""):

        measured_att = np.random.normal(measured_att_cv,
                                        measured_att_uc,
                                        n_throws)
        
        n = np.sqrt(self.epsilon_prime_real(self.depths))
        average_n_at_depth = np.cumsum(n) / np.arange(1, len(n) + 1)
        prop_time = np.sqrt(np.square(self.depths) + np.square(244)) * (average_n_at_depth / scipy.constants.c) * 2.0

        f_tof = scipy.interpolate.interp1d(prop_time,
                                           self.depths * 2.0,
                                           kind='linear',
                                           bounds_error=False,
                                           fill_value=0.0)

        ts = np.linspace(0.0e-6, 40e-6, 1000)
        d_ti_ = f_tof(ts)        

        if(new_figure):
            plt.figure()

        entries = np.zeros((n_throws, len(d_ti_)))
        for i_throw in range(n_throws):
            self.toy_mc_values()
            target_att = measured_att[i_throw]        
            avg_L_alpha = self.convert_N_to_L(self.N_alpha_star())[-1]

            self.scale = target_att / avg_L_alpha        
            avg_L_alpha = self.convert_N_to_L(self.N_alpha_star())

            f_avg_L_alpha = scipy.interpolate.interp1d(self.depths * 2.0,
                                                       avg_L_alpha,
                                                       kind='linear',
                                                       bounds_error=False,
                                                       fill_value=0.0)

            d_ti_[d_ti_ == 0] = 1e-10  # to remove divide by zero errors
            P_ti = 1.0 / np.square(d_ti_) * np.exp(-2.0 * d_ti_ / f_avg_L_alpha(d_ti_))
            entries[i_throw] = P_ti
            self.scale = 1.0
        self.reset()
        
        middle_val = np.zeros(len(d_ti_))
        low_bound = np.zeros(len(d_ti_))
        high_bound = np.zeros(len(d_ti_))

        for i_bin in range(len(d_ti_)):
            entries_ = entries[:, i_bin]
            entries_, cumsum = analysis_funcs.calculate_uncertainty(entries_)            
            entries_min, entries_mid, entries_max = analysis_funcs.return_confidence_intervals(entries_, cumsum)
            middle_val[i_bin] = entries_mid
            low_bound[i_bin] = entries_min
            high_bound[i_bin] = entries_max

        plt.fill_between(ts,
                         low_bound,
                         high_bound,
                         alpha=0.5,
                         facecolor=edgecolor,
                         edgecolor=edgecolor,
                         hatch=hatch,
                         label=label)

        if(new_figure):
            plt.yscale('log')
            plt.xlabel("Time of Flight [s]")
            plt.ylabel("Scaled Power []")
            plt.grid()
            plt.xlim(0.0, 4e-5)
            plt.ylim(1e-14, 1e-3)
