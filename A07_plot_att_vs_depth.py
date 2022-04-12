import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.constants
import experiment
import analysis_funcs


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


def setup_models():

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

    # Pure ice model of att. vs depth
    attmodel_pure = experiment.AttenuationModel(depths, np.zeros(len(depths)), np.zeros(len(depths)), np.zeros(len(depths)),
                                                sigma_pure, sigma_pure_uc,
                                                [0, 0, 0],
                                                [0, 0, 0],
                                                [E_pure, 0, 0, 0],
                                                [E_pure_uc, 0, 0, 0],
                                                temps, T_r)

    # Bogorodsky model of att. vs depth
    # Have to do a bit to make this work with the current class.
    # Reconfigures the sigma_pure to be equal to the old functional form.
    temperature_scale = (1 / (T_r + 273.15) - 1 / (temps + 273.15)) / 8.616e-5
    original_func = np.sqrt(attmodel_macgregor.epsilon_prime_real(depths)) / np.power(10.0, -0.017 * temps) / np.exp(E_pure * temperature_scale)
    attmodel_bog = experiment.AttenuationModel(depths, np.zeros(len(depths)), np.zeros(len(depths)), np.zeros(len(depths)),
                                               original_func, 1e-10,
                                               [0, 0, 0], 
                                               [0, 0, 0],
                                               [E_pure, 0, 0, 0],
                                               [1e-10, 0, 0, 0],
                                               temps, T_r)
    
    # Paden model of att. vs depth
    # Have to do a bit to make this work with the current class.
    # Reconfigures the sigma_pure to be equal to the old functional form.
    deps -= sigma_pure
    deps[deps < 0] = 0
    attmodel_paden = experiment.AttenuationModel(depths, deps, np.zeros(len(depths)), np.zeros(len(depths)),
                                                 sigma_pure, sigma_pure_uc,                              
                                                 [1.0, 0, 0],
                                                 [1e-5, 0, 0],
                                                 [E_pure, 0.22, 0, 0],
                                                 [E_pure_uc, 0.02, 0, 0],
                                                 temps, -15.0)

    return attmodel_macgregor, attmodel_pure, attmodel_bog, attmodel_paden    


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

    attmodel_macgregor, attmodel_pure, attmodel_bog, attmodel_paden = setup_models()
        
    ####################################################################################
    # Calculate scale factor to convert bulk attenuation to attenuation for top 1500 m #
    ####################################################################################
    scale_mean, scale_std = attmodel_macgregor.calculate_scale(measured_att_cv=att_middle[freqs_to_plot[0]],
                                                               measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]])

    print("Scale: \t \t %.4f" % scale_mean)
    print("Std Scale:\t %.4f" % scale_std)
    
    #######################################
    # Plot the chemistry of the situation #
    #######################################
    attmodel_macgregor.plot_chemistry()
    plt.title("Chemical Molarity measured at GRIP Borehole")
    plt.savefig("./plots/A07_chemistry.png", dpi=300)

    #################################################
    # Plot the conductivities from different models #
    #################################################
    attmodel_pure.plot_conductivity(label="Pure Ice", color="green", new_figure=True)
    attmodel_paden.plot_conductivity(label="Paden", color="red", new_figure=False)
    attmodel_macgregor.plot_conductivity(label="MacGregor", color="purple", new_figure=False)
    plt.legend()
    plt.savefig("./plots/A07_conductivity.png", dpi=300)

    #######################################################
    # Plot the model of att vs depth for different models #
    #######################################################
    attmodel_bog.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_to_plot[0]],
                                                       measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],
                                                       new_figure=True, edgecolor="black", hatch="//",
                                                       label="Draft")
    attmodel_pure.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_to_plot[0]],
                                                        measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],
                                                        new_figure=False, edgecolor="green", hatch="\\\\",
                                                        label="Pure Ice")
    plt.title("Attenuation vs. Depths at 300 MHz")
    plt.legend()
    plt.savefig("./plots/A07_L_alpha_with_errorbars_pure_ice.png", dpi=300)
    
    attmodel_bog.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_to_plot[0]],
                                                       measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],
                                                       new_figure=True, edgecolor="black", hatch="//",
                                                       label="Draft")
    attmodel_paden.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_to_plot[0]],
                                                         measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],
                                                         new_figure=False, edgecolor="red", hatch="--",
                                                         label="Paden")
    plt.title("Attenuation vs. Depths at 300 MHz")
    plt.legend()
    plt.savefig("./plots/A07_L_alpha_with_errorbars_paden.png", dpi=300)
    
    attmodel_bog.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_to_plot[0]],
                                                       measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],
                                                       new_figure=True, edgecolor="red", hatch="//",
                                                       label="Bogorodsky Model")
    attmodel_macgregor.plot_L_alpha_with_errorbars_and_scale(measured_att_cv=att_middle[freqs_to_plot[0]],
                                                             measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],                                             
                                                             new_figure=False, edgecolor="black", hatch=None, facecolor="black", alpha = 0.50,
                                                             label="MacGregor Model")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("./plots/A07_L_alpha_with_errorbars_macgregor.png", dpi=300)

    #########################
    # Plot the real epsilon #
    #########################
    plt.figure()
    plt.loglog(attmodel_macgregor.depths, attmodel_macgregor.epsilon_prime_real())
    plt.grid()
    plt.xlabel("Depth [m]")
    plt.ylabel("Real Relative Permittivity [unitless]")
    plt.savefig("./plots/A07_real_permittivity.png", dpi=300)

    ###########################################################
    # Plot the power from model of attenuation on top of data #
    ###########################################################    
    ice_file_name = "./data_processed/averaged_in_ice_trace.npz"
    air_file_name = "./data_processed/averaged_in_air_trace.npz"
    
    exper_constants = experiment.ExperimentConstants()
    exper = experiment.Experiment(exper_constants, ice_file_name, air_file_name)    

    data_time = exper.ice_time
    data_trace = exper.ice_trace        

    data_trace = analysis_funcs.butter_bandpass_filter(data_trace,
                                                       lowcut=(att_freq[freqs_to_plot[0]]/1e6 - 10) * 1e6,
                                                       highcut=(att_freq[freqs_to_plot[0]]/1e6 + 10) * 1e6,
                                                       fs=1.0 / (data_time[1] - data_time[0]),
                                                       order=5)

    window_length = 250
    data_power_mW = np.power(data_trace, 2.0) / 50.0 * 1e3
    delta_t = (data_time[1] - data_time[0]) * 1e9
    time_length = window_length * delta_t
        
    data_trace_power = np.convolve(data_power_mW,
                                   np.ones(window_length) / float(window_length),
                                   'valid')
    data_time = (data_time[(window_length - 1):]
                 + data_time[:-(window_length - 1)]) / 2.0

    attmodel_macgregor.plot_power_trace_lookalike(measured_att_cv=att_middle[freqs_to_plot[0]],
                                                  measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],
                                                  label="Attenuation",
                                                  edgecolor="purple")
    plt.title("MacGregor Model")
    plt.plot(data_time, data_trace_power / 80.0,
             color='black', alpha=0.5, label="Data")
    plt.legend()
    plt.savefig("./plots/A07_power_trace_macgregor.png", dpi=300)


    attmodel_bog.plot_power_trace_lookalike(measured_att_cv=att_middle[freqs_to_plot[0]],
                                            measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],
                                            label="Power derived from La(z) model.",
                                            edgecolor='black')

    plt.title("Draft Model")
    plt.plot(data_time, data_trace_power / 80.0,
             color='black', alpha=0.5, label="Data, scaled to match model")
    plt.legend()
    plt.savefig("./plots/A07_power_trace_original.png", dpi=300)

    attmodel_paden.plot_power_trace_lookalike(measured_att_cv=att_middle[freqs_to_plot[0]],
                                              measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],
                                              label="Power derived from La(z) model.",
                                              edgecolor="red")
    plt.title("Paden Model")
    plt.plot(data_time, data_trace_power / 250.0,
             color='black', alpha=0.5, label="Data, scaled to match model")
    plt.legend()
    plt.savefig("./plots/A07_power_trace_paden.png", dpi=300)
    
    attmodel_pure.plot_power_trace_lookalike(measured_att_cv=att_middle[freqs_to_plot[0]],
                                             measured_att_uc=((att_yerr_max - att_yerr_min)/2)[freqs_to_plot[0]],
                                             label="Power derived from La(z) model.",
                                             edgecolor="green")
    plt.title("Pure Model")
    plt.plot(data_time, data_trace_power / 70.0,
             color='black', alpha=0.5, label="Data, scaled to match model")
    plt.legend()
    plt.savefig("./plots/A07_power_trace_pure.png", dpi=300)
    

if __name__ == "__main__":

    bulk_att_file = "./data_processed/A05_mc_results.npz"
    freqs_to_plot = [9]  # 308.333308 MHz

    main(bulk_att_file, freqs_to_plot)

    plt.show()
