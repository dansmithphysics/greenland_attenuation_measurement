import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.constants
from scipy.optimize import curve_fit
import A06_plot_att
import A07_plot_att_vs_depth
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

    attmodel_macgregor, attmodel_pure, attmodel_bog, attmodel_paden = A07_plot_att_vs_depth.setup_models()
    
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

    plt.savefig("./plots/A09_fit_att_vs_depth.png", dpi=300)
    plt.show()
