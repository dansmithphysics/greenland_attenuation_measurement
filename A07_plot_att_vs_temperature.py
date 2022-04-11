import copy
import analysis_funcs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.constants
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit

from NuRadioMC.SignalProp import propagation
from NuRadioMC.SignalProp.analyticraytracing import solution_types, ray_tracing_2D
from NuRadioMC.utilities import medium
from NuRadioReco.utilities import units


def load_and_interpolate(depths, file_name, skiprows, usecols=None, delimiter=None):

    # Load up grip data
    depth_, vals_ = np.loadtxt(file_name,
                               skiprows=skiprows,
                               usecols=usecols,
                               unpack=True,
                               delimiter=delimiter)

    f = scipy.interpolate.interp1d(depth_,
                                   vals_, 
                                   kind='linear',
                                   bounds_error=False,
                                   fill_value=np.mean(vals_[:100]))

    vals_ = f(depths)
    return vals_


def load(file_name,
         att_correction,
         time_offset=0):

    try:
        data_ice = np.load(file_name)
    except FileNotFoundError:
        print("File not found: %s" % file_name)
        return

    data_t = data_ice["template_time"]
    data_trace = data_ice["template_trace"]

    data_t += time_offset
    data_trace *= np.power(10.0, att_correction / 20.0)

    return data_t, data_trace


def load_synth_data_trace():
    
    time_offset = (35.55e-6 - 34.59e-6)
    
    data_t, data_trace = load("data_processed/averaged_in_ice_trace.npz",
                              att_correction=0.0,
                              time_offset=time_offset)
    data_t_20db, data_trace_20db = load("data_processed/averaged_in_ice_trace_20db.npz",
                                        att_correction=20.0,
                                        time_offset=time_offset)
    data_t_40db, data_trace_40db = load("data_processed/averaged_in_ice_trace_40db.npz",
                                        att_correction=40.0,
                                        time_offset=time_offset)

    f_data_trace_20db = scipy.interpolate.interp1d(data_t_20db,
                                                   data_trace_20db,
                                                   kind='linear',
                                                   bounds_error=False,
                                                   fill_value=0.0)
    data_trace_20db = f_data_trace_20db(data_t)
    f_data_trace_40db = scipy.interpolate.interp1d(data_t_40db,
                                                   data_trace_40db,
                                                   kind='linear',
                                                   bounds_error=False,
                                                   fill_value=0.0)
    data_trace_40db = f_data_trace_40db(data_t)
    
    normalize_time_range = (5.0e-6, 10.0e6)
    selection_range = np.logical_and(data_t > normalize_time_range[0], data_t < normalize_time_range[1])
    scale = np.sum(np.square(data_trace[selection_range])) / np.sum(np.square(data_trace_20db[selection_range]))
    data_trace_20db *= np.sqrt(scale)
    data_trace_40db *= np.sqrt(scale) # for some reason, this works well ..... not the best sign.
    
    data_trace_synth = np.zeros(len(data_trace))
    data_trace_synth[data_t <= 2e-6] = data_trace_40db[data_t < 2e-6]
    data_trace_synth[np.logical_and(data_t > 2e-6, data_t <= 10e-6)] = data_trace_20db[np.logical_and(data_t > 2e-6, data_t <= 10e-6)]
    data_trace_synth[data_t > 10e-6] = data_trace[data_t > 10e-6]

    return data_t, data_trace_synth


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class DepthConverter:

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
            #plt.ylim(0.0, 2000)
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
            #plt.ylim(0.0, 2000)
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
            plt.xlim(0.0, max_depth)
            plt.ylim(300.0, 2000.0)
            plt.grid()
            
    def calculate_scale(self, measured_att_cv, measured_att_uc, n_throws=1000):    

        measured_att = np.random.normal(measured_att_cv,
                                        measured_att_uc,
                                        n_throws)

        print(measured_att_cv)
        print(measured_att_uc)
        
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

        print("Scale:", np.mean(avg_L_alphas / measured_att))
        print("Std Scale:", np.std(avg_L_alphas / measured_att))

        plt.hist(avg_L_alphas / measured_att, range=(1.0, 1.5), bins=50)
        plt.show()
        exit()
        
        plt.scatter(measured_att, avg_L_alphas / measured_att, alpha=0.2)
        plt.xlabel("Bulk attenuation")
        plt.ylabel("Attenuation of top 1500 m")
        plt.xlim(500.0, 1200.0)
        #plt.ylim(600.0, 1300.0)
        plt.grid()
        plt.show()
        exit()
        
        middle_val = np.zeros(len(self.depths))
        low_bound = np.zeros(len(self.depths))
        high_bound = np.zeros(len(self.depths))

        for i_bin in range(len(self.depths)):
            entries_ = entries[:, i_bin].flatten()
            entries_, cumsum = analysis_funcs.calculate_uncertainty(entries_)
            
            entries_min, entries_mid, entries_max = analysis_funcs.return_confidence_intervals(entries_, cumsum)

            middle_val[i_bin] = entries_mid
            low_bound[i_bin] = entries_min
            high_bound[i_bin] = entries_max

        x = self.depths
        yerr_min = low_bound
        yerr_max = high_bound
        y = middle_val

        avg_L_top_1500 = np.sum(self.depths < 1500.0) / np.sum(1.0 / middle_val[self.depths < 1500.0])
        avg_L_top_1500_max = np.sum(self.depths < 1500.0) / np.sum(1.0 / yerr_max[self.depths < 1500.0])
        avg_L_top_1500_min = np.sum(self.depths < 1500.0) / np.sum(1.0 / yerr_min[self.depths < 1500.0])

        new_figure = True
        if(new_figure):
            plt.figure(figsize=(5, 4))        
        plt.fill_between(x,
                         yerr_min,
                         yerr_max,
                         alpha=0.95,
                         facecolor="none")
        plt.axhline(avg_L_top_1500)
        plt.axhline(avg_L_top_1500_max)
        plt.axhline(avg_L_top_1500_min)
        if(new_figure):
            plt.xlabel("Depth [m]")
            plt.ylabel("Att. Length [m]")
            plt.xlim(0.0, max_depth)
            plt.ylim(300.0, 2000.0)
            plt.grid()
            
    def plot_power_trace_lookalike(self, measured_att_cv, measured_att_uc, n_throws=1000,
                                   new_figure=True, edgecolor="black", hatch=None, label=""):

        measured_att = np.random.normal(measured_att_cv,
                                        measured_att_uc,
                                        n_throws)
        
        #n = np.sqrt(self.epsilon_prime_real(self.depths))
        #average_n_at_depth = np.cumsum(n) / np.arange(1, len(n) + 1)
        #prop_time = np.sqrt(np.square(self.depths) + np.square(244)) * (average_n_at_depth / scipy.constants.c) * 2.0

        prop_time = np.zeros(len(self.depths))

        prop = propagation.get_propagation_module('analytic')
        ref_index_model = 'greenland_simple'
        ice = medium.get_ice_model(ref_index_model)
        # Let us work on the y = 0 plane
        initial_point = np.array([0, 0, 0]) * units.m
        attenuation_model = 'GL1'

        rays = prop(ice, attenuation_model,
                    n_frequencies_integration=25,
                    n_reflections=0)
        for i_depth, depth_ in enumerate(self.depths):
            final_point = np.array([244.0/2.0, 0, -1 * depth_]) * units.m

            rays.set_start_and_end_point(initial_point,final_point)
            rays.find_solutions()

            for i_solution in range(rays.get_number_of_solutions()):
                solution_int = rays.get_solution_type(i_solution)
                if(solution_types[solution_int] != 'direct'):
                    continue
                tof = rays.get_travel_time(i_solution)
                prop_time[i_depth] = 2.0 * tof * 1e-9

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


    def fit_power_trace_lookalike(self, data_t, data_trace, n_throws=100,
                                  new_figure=True, edgecolor="black", hatch=None, label=""):
        
        #n = np.sqrt(self.epsilon_prime_real(self.depths))
        #average_n_at_depth = np.cumsum(n) / np.arange(1, len(n) + 1)
        #prop_time = np.sqrt(np.square(self.depths) + np.square(244)) * (average_n_at_depth / scipy.constants.c) * 2.0

        prop_time = np.zeros(len(self.depths))

        prop = propagation.get_propagation_module('analytic')
        ref_index_model = 'greenland_simple'
        ice = medium.get_ice_model(ref_index_model)
        # Let us work on the y = 0 plane
        initial_point = np.array([0, 0, 0]) * units.m
        attenuation_model = 'GL1'

        rays = prop(ice, attenuation_model,
                    n_frequencies_integration=25,
                    n_reflections=0)
        for i_depth, depth_ in enumerate(self.depths):
            final_point = np.array([244.0/2.0, 0, -1 * depth_]) * units.m

            rays.set_start_and_end_point(initial_point,final_point)
            rays.find_solutions()

            for i_solution in range(rays.get_number_of_solutions()):
                solution_int = rays.get_solution_type(i_solution)
                if(solution_types[solution_int] != 'direct'):
                    continue
                tof = rays.get_travel_time(i_solution)
                prop_time[i_depth] = 2.0 * tof * 1e-9

        f_tof = scipy.interpolate.interp1d(prop_time,
                                           self.depths * 2.0,
                                           kind='linear',
                                           bounds_error=False,
                                           fill_value=0.0)

        #ts = np.linspace(0.0e-6, 40e-6, 1000)
        d_ti_ = f_tof(data_t)

        #plt.plot(d_ti_)
        #plt.show()
        #exit()
        
        #if(new_figure):
        #    plt.figure()

        entries = np.zeros((n_throws, len(d_ti_)))
        atts = np.zeros(n_throws)
        for i_throw in range(n_throws):
            self.toy_mc_values()
            avg_L_alphas = self.convert_N_to_L(self.N_alpha_star())

            # fit that returns a target_att
            def func(x, P_0, att):
                scale = att / avg_L_alphas[len(avg_L_alphas) // 2]

                #scale = att / avg_L_alphas[-1]
                
                #self.scale = target_att / avg_L_alpha   
                #avg_L_alpha = self.convert_N_to_L(self.N_alpha_star())
                
                f_avg_L_alpha = scipy.interpolate.interp1d(self.depths * 2.0,
                                                           scale * avg_L_alphas,
                                                           kind='linear',
                                                           bounds_error=False,
                                                           fill_value=0.0)
                
                P_ti = np.log(P_0) - np.log(np.square(d_ti_)) + (-2 * d_ti_ / f_avg_L_alpha(d_ti_))
                return P_ti
            
            popt, pcov = curve_fit(func, data_t, np.log(data_trace))

            #plt.plot(data_t, func(data_t, *popt))
            #plt.plot(data_t, np.log(data_trace))
            #plt.show()
            #exit()

            #print(popt)
            #print(pcov)
            #print(np.sqrt(np.diag(pcov)))
            #exit()
            
            atts[i_throw] = popt[1]
            
            #entries[i_throw] = P_ti
            #self.scale = 1.0
        self.reset()
        
        print("att = %.2f +/- %.2f" % (np.mean(atts), np.std(atts)))
        return np.mean(atts), np.std(atts)

        
        plt.hist(atts)
        plt.show()
        exit()
        
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
    

            
if __name__ == "__main__":

        
    # Load bulk attenuation measurement
    result_data = np.load("./data_processed/A05_mc_results.npz")    
    att_freq = result_data['freqs']
    att_yerr_min = result_data['low_bound']
    att_yerr_max = result_data['high_bound']
    att_middle = result_data['middle_val']

    # Load GRIP borehole data.
    max_depth = 3004.0
    depths = np.linspace(0.0, max_depth, 3004)

    temps = load_and_interpolate(depths, "./data_raw/griptemp.txt",
                                 skiprows=40, delimiter="\t")
    cls = load_and_interpolate(depths, "./data_raw/gripion.txt",
                               skiprows=75, usecols=[0, 6])
    nh4s = load_and_interpolate(depths, "./data_raw/gripion.txt",
                                skiprows=75, usecols=[0, 11])
    deps = load_and_interpolate(depths, "./data_raw/gripdep.txt",
                                skiprows=80, usecols=[0, 1])
    hs = load_and_interpolate(depths, "./data_raw/gripdep.txt",
                              skiprows=80, usecols=[0, 2])    

    # H+ (hs) is in units of micromolarity.
    # reported as "micromols per kg"

    density = np.zeros(len(depths))
    density[depths <= 14.9] = 0.917 - 0.594 * np.exp(-depths[depths <= 14.9] / 30.8)
    density[depths > 14.9] = 0.917 - 0.367 * np.exp(-(depths[depths > 14.9] - 14.9)/ 40.5)
    
    # Cl (cls) is in units of ppm.
    cl_molar_mass = 35.453  # g/mol
    #cls = cls / cl_molar_mass # uM
    cls = cls * density / cl_molar_mass # uM
    
    # NH4 (nh4s) is in units of ppm.
    nh4_molar_mass = 18.04  # g/mol
    #nh4s = nh4s / nh4_molar_mass # uM
    nh4s = nh4s * density / nh4_molar_mass # uM

    T_r = -21.0  # C

    sigma_pure = 9.2 # uS/m
    sigma_pure_uc = 0.2 
    
    mu_h = 3.2  # S/m/M
    mu_h_uc = 0.5
    mu_cl = 0.43  # S/m/M
    mu_cl_uc = 0.07
    mu_nh4 = 0.8  # S/m/M
    mu_nh4_uc = 0.05  # estimated

    E_pure = 0.51 #0.55  # eV
    E_pure_uc = 0.05
    E_h = 0.20  # eV
    E_h_uc = 0.04
    E_cl = 0.19  # eV
    E_cl_uc = 0.02
    E_nh4 = 0.23  # eV
    E_nh4_uc = 0.03  # estimated
    
    dc = DepthConverter(depths, hs, cls, nh4s,
                        sigma_pure, sigma_pure_uc,                        
                        [mu_h, mu_cl, mu_nh4],
                        [mu_h_uc, mu_cl_uc, mu_nh4_uc],
                        [E_pure, E_h, E_cl, E_nh4],
                        [E_pure_uc, E_h_uc, E_cl_uc, E_nh4_uc],
                        temps, T_r)

    print(dc.depths)
    #exit()
    
    dc.smooth_chemistry(window_length=10) # 10 m averaging
    
    dc_pure = DepthConverter(depths, np.zeros(len(depths)), np.zeros(len(depths)), np.zeros(len(depths)),
                             sigma_pure, sigma_pure_uc,                             
                             [0, 0, 0], 
                             [0, 0, 0],
                             [E_pure, 0, 0, 0],
                             [E_pure_uc, 0, 0, 0],
                             temps, T_r)

    # Have to cheat a bit to make this work with the current framework.
    # Reconfigures the sigma_pure to be equal to the old functional form.
    temperature_scale = (1 / (T_r + 273.15) - 1 / (temps + 273.15)) / 8.616e-5
    original_func = np.sqrt(dc.epsilon_prime_real(depths)) / np.power(10.0, -0.017 * temps) / np.exp(E_pure * temperature_scale)
    dc_original = DepthConverter(depths, np.zeros(len(depths)), np.zeros(len(depths)), np.zeros(len(depths)),
                                 original_func, 1e-10,                                 
                                 [0, 0, 0], 
                                 [0, 0, 0],
                                 [E_pure, 0, 0, 0],
                                 [1e-10, 0, 0, 0],
                                 temps, T_r)

    # Also have to cheat a bit to make this work with the current framework.
    deps -= sigma_pure
    deps[deps < 0] = 0
    dc_paden = DepthConverter(depths, deps, np.zeros(len(depths)), np.zeros(len(depths)),
                              sigma_pure, sigma_pure_uc,                              
                              [1.0, 0, 0],
                              [1e-5, 0, 0],
                              [E_pure, 0.22, 0, 0],
                              [E_pure_uc, 0.02, 0, 0],
                              temps, -15.0)                              


    freqs_oi = [9] #[0, 11] # 158 and 342 MHz

    #dc.calculate_scale(measured_att_cv=att_middle[freqs_oi[0]],
    #                   measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]])


    #plt.plot(dc.temp, dc.conductivity())
    #plt.plot(dc_paden.temp, dc_paden.conductivity())
    #plt.plot(dc_pure.temp, dc_pure.conductivity())
    #plt.plot(dc_original.temp, dc_original.conductivity())
    #plt.xlim(-35.0, 0.0)
    #plt.ylim(0.0, 80.0)
    #plt.grid()
    #plt.show()
    #exit()
    
    # Plot the chemistry of the situation
    dc.plot_chemistry()
    plt.title("Chemical Molarity measured at GRIP Borehole")
    plt.savefig("./new_L_alpha_plots/chemistry.png", dpi=300)

    plt.xlim(0.0, 3000.0)
    #plt.ylim(0.0, 200.0)
    
    
    # Plot the conductivities from different models
    dc_pure.plot_conductivity(label="Pure Ice", color="green", new_figure=True)
    dc_paden.plot_conductivity(label="Paden", color="red", new_figure=False)
    dc.plot_conductivity(label="MacGregor", color="purple", new_figure=False)
    plt.legend()
    plt.savefig("./new_L_alpha_plots/conductivity.png", dpi=300)

    #plt.show()
    #exit()

    
    # Plot the Attenuation vs. depth
    att_freq = result_data['freqs']
    att_yerr_min = result_data['low_bound']
    att_yerr_max = result_data['high_bound']
    att_middle = result_data['middle_val']

    for i in range(len(att_freq)):
        print(i, att_freq[i]/1e6)

    dc_original.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_oi[0]],
                                                      measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                                      new_figure=True, edgecolor="black", hatch="//", label="Draft")
    dc_pure.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_oi[0]],
                                                  measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                                  new_figure=False, edgecolor="green", hatch="\\\\", label="Pure Ice")
    plt.title("Attenuation vs. Depths at 300 MHz")
    plt.ylim(300, 2000)
    plt.legend()
    plt.savefig("./new_L_alpha_plots/L_alpha_with_errorbars_pure_ice.png", dpi=300)
    
    dc_original.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_oi[0]],
                                                      measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                                      new_figure=True, edgecolor="black", hatch="//", label="Draft")
    dc_paden.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_oi[0]],
                                                   measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                                   new_figure=False, edgecolor="red", hatch="--", label="Paden")
    plt.title("Attenuation vs. Depths at 300 MHz")
    plt.ylim(300, 2000)
    plt.legend()
    plt.savefig("./new_L_alpha_plots/L_alpha_with_errorbars_paden.png", dpi=300)
    
    dc_original.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_oi[0]],
                                                      measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                                      new_figure=True, edgecolor="red", hatch="//", label="Bogorodsky Model")
    dc.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_oi[0]],
                                             measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                             #new_figure=False, edgecolor="black", hatch="\\\\", facecolor=None, label="MacGregor Model")
                                             new_figure=False, edgecolor="black", hatch=None, facecolor="black", alpha = 0.50, label="MacGregor Model")
    plt.ylim(0, 2000)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("./new_L_alpha_plots/L_alpha_with_errorbars_macgregor.png", dpi=300)
    
    # Plot the real epsilon
    plt.figure()
    plt.loglog(dc.depths, dc.epsilon_prime_real())
    plt.grid()
    plt.xlabel("Depth [m]")
    plt.ylabel("Real Relative Permittivity [unitless]")
    plt.savefig("./new_L_alpha_plots/real_permittivity.png", dpi=300)


    # Plot the attenuation on the data trace
    time_offset = (35.55e-6 - 34.59e-6)

    data_t, data_trace_synth = load_synth_data_trace()
    data_trace = copy.deepcopy(data_trace_synth)

    data_trace = butter_bandpass_filter(data_trace,
                                        lowcut=(att_freq[freqs_oi[0]]/1e6 - 10) * 1e6,
                                        highcut=(att_freq[freqs_oi[0]]/1e6 + 10) * 1e6,
                                        fs=1.0 / (data_t[1] - data_t[0]),
                                        order=5)


    window_length = 250
    data_power_mW = np.power(data_trace, 2.0) / 50.0 * 1e3
    delta_t = (data_t[1] - data_t[0]) * 1e9
    time_length = window_length * delta_t
        
    data_trace_power = np.convolve(data_power_mW,
                                   np.ones(window_length) / float(window_length),
                                   'valid')
    data_t = (data_t[(window_length - 1):]
               + data_t[:-(window_length - 1)]) / 2.0

    '''

    for i in range(len(att_freq)):
        print(i, att_freq[i]/1e6)

    #freqs_oi = [20]
    #freqs_oi = [20]
    #freqs_oi = [20]
    freqs_oi = [0]

    att_freqs = []
    means = []
    stds = []
    for i in range(0, len(att_freq), 1):
    
        data_trace_ = butter_bandpass_filter(data_trace,
                                             lowcut=(att_freq[i]/1e6 - 10) * 1e6,
                                             highcut=(att_freq[i]/1e6 + 10) * 1e6,
                                             fs=1.0 / (data_t[1] - data_t[0]),
                                             order=5)

        window_length = 250
        data_power_mW = np.power(data_trace_, 2.0) / 50.0 * 1e3
        delta_t = (data_t[1] - data_t[0]) * 1e9
        time_length = window_length * delta_t
        
        data_trace_power = np.convolve(data_power_mW,
                                       np.ones(window_length) / float(window_length),
                                       'valid')

        # rolling is the integrated sliding window result and has units of mW * ns
        data_t_ = (data_t[(window_length - 1):]
                  + data_t[:-(window_length - 1)]) / 2.0
        
        #selection_region = np.logical_and(data_t > 5e-6, data_t < 20e-6)
        #selection_region = np.logical_and(data_t_ > 10e-6, data_t_ < 20e-6)
        selection_region = np.logical_and(data_t_ > 7.5e-6, data_t_ < 20e-6)
    
        mean, std = dc.fit_power_trace_lookalike(data_t_[selection_region],
                                                 data_trace_power[selection_region])
        att_freqs += [att_freq[i]]
        means += [mean]
        stds += [std]

        print("freqs_oi:", i, att_freq[i] / 1e6, mean, std)

    plt.errorbar(np.array(att_freqs) / 1e6,
                 means,
                 yerr=stds,
                 color='black')
    #ls='none')

    plt.axvline(145.0, color='red', linestyle='--', label="Bandpass Filters")
    plt.axvline(575.0, color='red', linestyle='--')
    
    plt.xlim(50.0, 600.0)
    plt.ylim(500.0, 1300.0)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(r"Avg. Field Attenuation Length of Top 1500 m [m]")
    plt.grid()
    plt.show()
    
    exit()
    '''
    
    dc.plot_power_trace_lookalike(measured_att_cv=att_middle[freqs_oi[0]],
                                  measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                  label="Attenuation",
                                  edgecolor="purple")
    plt.title("MacGregor Model")
    plt.plot(data_t, data_trace_power / 400.0,
             color='black', alpha=0.5, label="Data")
    plt.legend()
    plt.savefig("./new_L_alpha_plots/power_trace_macgregor.png", dpi=300)


    dc_original.plot_power_trace_lookalike(measured_att_cv=att_middle[freqs_oi[0]],
                                           measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                           label="Power derived from La(z) model.",
                                           edgecolor='black')

    plt.title("Draft Model")
    plt.plot(data_t, data_trace_power / 300.0,
             color='black', alpha=0.5, label="Data, scaled to match model")
    plt.legend()
    plt.savefig("./new_L_alpha_plots/power_trace_original.png", dpi=300)

    dc_paden.plot_power_trace_lookalike(measured_att_cv=att_middle[freqs_oi[0]],
                                        measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                        label="Power derived from La(z) model.",
                                        edgecolor="red")
    plt.title("Paden Model")
    plt.plot(data_t, data_trace_power / 500.0,
             color='black', alpha=0.5, label="Data, scaled to match model")
    plt.legend()
    plt.savefig("./new_L_alpha_plots/power_trace_paden.png", dpi=300)
    
    dc_pure.plot_power_trace_lookalike(measured_att_cv=att_middle[freqs_oi[0]],
                                       measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_oi[0]],
                                       label="Power derived from La(z) model.",
                                       edgecolor="green")
    plt.title("Pure Model")
    plt.plot(data_t, data_trace_power / 100.0,
             color='black', alpha=0.5, label="Data, scaled to match model")
    plt.legend()
    plt.savefig("./new_L_alpha_plots/power_trace_pure.png", dpi=300)

    
    plt.show()





