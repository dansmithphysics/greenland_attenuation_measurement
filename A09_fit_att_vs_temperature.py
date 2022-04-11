import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.constants
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
import A06_plot_att
import A07_plot_att_vs_temperature
import experiment
import analysis_funcs


def fit_power_trace_lookalike(model, data_t, data_trace, n_throws=10,
                              new_figure=True, edgecolor="black", hatch=None, label=""):
        
    n = np.sqrt(model.epsilon_prime_real(model.depths))
    average_n_at_depth = np.cumsum(n) / np.arange(1, len(n) + 1)
    prop_time = np.sqrt(np.square(model.depths) + np.square(244)) * (average_n_at_depth / scipy.constants.c) * 2.0

    f_tof = scipy.interpolate.interp1d(prop_time,
                                       model.depths * 2.0,
                                       kind='linear',
                                       bounds_error=False,
                                       fill_value=0.0)
    d_ti_ = f_tof(data_t)

    entries = np.zeros((n_throws, len(d_ti_)))
    atts = np.zeros(n_throws)
    for i_throw in range(n_throws):
        model.toy_mc_values()
        avg_L_alphas = model.convert_N_to_L(model.N_alpha_star())

        # fit that returns a target_att
        def func(x, P_0, att):
            scale = att / avg_L_alphas[len(avg_L_alphas) // 2]                
            f_avg_L_alpha = scipy.interpolate.interp1d(model.depths * 2.0,
                                                       scale * avg_L_alphas,
                                                       kind='linear',
                                                       bounds_error=False,
                                                       fill_value=0.0)
                
            P_ti = np.log(P_0) - np.log(np.square(d_ti_)) + (-2 * d_ti_ / f_avg_L_alpha(d_ti_))
            return P_ti
            
        popt, pcov = curve_fit(func, data_t, np.log(data_trace))            
        atts[i_throw] = popt[1]
            
    model.reset()

    return np.mean(atts), np.std(atts)


def main(bulk_att_file, freqs_to_plot):
    
    # Load bulk attenuation measurement
    result_data = np.load(bulk_att_file)
    att_freq = result_data['freqs']
    att_yerr_min = result_data['low_bound']
    att_yerr_max = result_data['high_bound']
    att_middle = result_data['middle_val']

    print("index \t Freq [MHz]")
    for i in range(len(att_freq)):
        print("%i \t %.2f" % (i, att_freq[i]/1e6))
    
    # Load GRIP borehole data.
    max_depth = 3004.0
    depths = np.linspace(0.0, max_depth, 3004)

    temps = A07_plot_att_vs_temperature.load_and_interpolate(depths, "./data_raw/griptemp.txt",
                                                             skiprows=40, delimiter="\t")
    cls = A07_plot_att_vs_temperature.load_and_interpolate(depths, "./data_raw/gripion.txt",
                                                           skiprows=75, usecols=[0, 6])
    nh4s = A07_plot_att_vs_temperature.load_and_interpolate(depths, "./data_raw/gripion.txt",
                                                            skiprows=75, usecols=[0, 11])
    deps = A07_plot_att_vs_temperature.load_and_interpolate(depths, "./data_raw/gripdep.txt",
                                                            skiprows=80, usecols=[0, 1])
    hs = A07_plot_att_vs_temperature.load_and_interpolate(depths, "./data_raw/gripdep.txt",
                                                          skiprows=80, usecols=[0, 2])    

    # H+ (hs) is in units of micromolarity.
    # reported as "micromols per kg"
    
    # Cl (cls) is in units of ppm.
    cl_molar_mass = 35.453  # g/mol
    cls = cls / cl_molar_mass # uM
    
    # NH4 (nh4s) is in units of ppm.
    nh4_molar_mass = 18.04  # g/mol
    nh4s = nh4s / nh4_molar_mass # uM

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

    # MacGregor model of att. vs depth
    attmodel_macgregor = experiment.AttenuationModel(depths, hs, cls, nh4s,
                                                     sigma_pure, sigma_pure_uc,
                                                     [mu_h, mu_cl, mu_nh4],
                                                     [mu_h_uc, mu_cl_uc, mu_nh4_uc],
                                                     [E_pure, E_h, E_cl, E_nh4],
                                                     [E_pure_uc, E_h_uc, E_cl_uc, E_nh4_uc],
                                                     temps, T_r)
    attmodel_macgregor.smooth_chemistry(window_length=10) # 10 m averaging
        
    ice_file_name = "./data_processed/averaged_in_ice_trace.npz"
    air_file_name = "./data_processed/averaged_in_air_trace.npz"
    
    exper_constants = experiment.ExperimentConstants()
    exper = experiment.Experiment(exper_constants, ice_file_name, air_file_name)    

    data_time = exper.ice_time
    data_trace = exper.ice_trace        

    att_freqs = np.zeros(len(freqs_to_plot))
    means = np.zeros(len(freqs_to_plot))
    stds = np.zeros(len(freqs_to_plot))

    print("Index \t Freq [MHz] \t Mean \t Std")
    for i, i_freq in enumerate(freqs_to_plot):
    
        data_trace_ = analysis_funcs.butter_bandpass_filter(data_trace,
                                                            lowcut=(att_freq[i_freq]/1e6 - 10) * 1e6,
                                                            highcut=(att_freq[i_freq]/1e6 + 10) * 1e6,
                                                            fs=1.0 / (data_time[1] - data_time[0]),
                                                            order=5)

        window_length = 250
        data_power_mW = np.power(data_trace_, 2.0) / 50.0 * 1e3
        delta_t = (data_time[1] - data_time[0]) * 1e9
        time_length = window_length * delta_t
        
        data_trace_power = np.convolve(data_power_mW,
                                       np.ones(window_length) / float(window_length),
                                       'valid')

        # rolling is the integrated sliding window result and has units of mW * ns
        data_time_ = (data_time[(window_length - 1):]
                      + data_time[:-(window_length - 1)]) / 2.0
        
        selection_region = np.logical_and(data_time_ > 10e-6, data_time_ < 20e-6)
    
        mean, std = fit_power_trace_lookalike(attmodel_macgregor,
                                              data_time_[selection_region],
                                              data_trace_power[selection_region])
        att_freqs[i] = att_freq[i_freq]
        means[i] = mean
        stds[i] = std

        print("%i \t %.2f \t %.2f \t %.2f" % (i, att_freq[i_freq] / 1e6, mean, std))

    ####################################
    # now, plot the attenuation result #
    ####################################
    scale_to_top_1p5k = 1.224577441691559
    scale_to_top_1p5k_sig = 0.06817277754283599
    A06_plot_att.main(exper_constants, scale_to_top_1p5k, scale_to_top_1p5k_sig, save_fig=False)
        
    plt.errorbar(np.array(att_freqs) / 1e6,
                 means,
                 yerr=stds,
                 color='green',
                 label="Att. from Fit")
    plt.legend()


if __name__ == "__main__":

    bulk_att_file = "./data_processed/A05_mc_results.npz"
    freqs_to_use = range(24)
    
    main(bulk_att_file, freqs_to_use)

    plt.savefig("./plots/A09_fit_att_vs_temperature.png", dpi=300)
    plt.show()
