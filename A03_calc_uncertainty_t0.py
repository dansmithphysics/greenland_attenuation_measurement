import glob
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import analysis_funcs
import experiment


if __name__ == "__main__":

    exper_constants = experiment.Experiment()

    window_length = 250

    data_t, data_trace = analysis_funcs.load_file(file_name="./data_processed/averaged_in_ice_trace.npz",
                                                  att_correction=exper_constants.ice_att,
                                                  time_offset=exper_constants.time_offset)

    #####################
    #      Raw Data     #
    #####################

    # Calculate the standard deviation of noise.
    noise_std = np.std(data_trace[np.logical_and(data_t > exper_constants.noise_start,
                                                 data_t < exper_constants.noise_end)])

    plt.figure()
    plt.title("Ice Echo Data from Bed Rock")

    plt.plot(data_t * 1e6,
             data_trace * 1e3,
             alpha=1.0, color='black', linewidth=1.0, label="Data")

    plt.axhline(3.0 * noise_std * 1e3, color='red', linestyle='--', label="3x Noise RMS")
    plt.axvline(exper_constants.gb_start * 1e6,
                color='purple',
                linestyle='-',
                label="Start of G.B.")
    plt.axvline(exper_constants.gb_end * 1e6,
                color='purple',
                linestyle="-.",
                label="End of G.B.")
    plt.xlabel("Time Since Trigger $t_0$ [$\mu$s]")
    plt.ylabel("Trace [mV]")
    plt.xlim(34.0, 39.0)
    plt.ylim(-0.7, 0.7)
    plt.legend(loc='upper right')
    plt.grid()

    plt.savefig("./plots/A03_calc_uncertainty_t0.png",
                dpi=300)

    #####################
    # Integrated Window #
    #####################

    rolling_t, rolling = analysis_funcs.power_integration(data_t, data_trace, window_length)

    # Calculate a 95% CI of noise.
    noise = rolling[np.logical_and(rolling_t > exper_constants.noise_start,
                                   rolling_t < exper_constants.noise_end)]

    noise, cumsum = analysis_funcs.calculate_uncertainty(noise)

    noise_threshold = noise[np.argmin(np.abs(cumsum - 0.95))]

    plt.figure()
    plt.title("Ice Echo Data from Bed Rock, Integrated in Sliding %i ns Window" %
              int(250.0 * 1e9 * (data_t[1] - data_t[0])))
    plt.semilogy((data_t[:len(rolling)] + data_t[-len(rolling):]) / 2.0 * 1e6,
                 rolling, alpha=1.0, color='black', linewidth=1.0, label="Integrated Data")

    plt.axhline(noise_threshold, color='red', linestyle='--', label="95% CI of Noise")
    plt.axvline(exper_constants.gb_start * 1e6,
                color='purple',
                linestyle='-',
                label="Start of G.B.")
    plt.axvline(exper_constants.gb_end * 1e6,
                color='purple',
                linestyle="-.",
                label="End of G.B.")
    plt.xlabel("Time Since Trigger $t_0$ [$\mu$s]")
    plt.ylabel("Integrated Power [mW ns]")
    plt.xlim(34.0, 39.0)
    plt.ylim(4e-7, 4e-4)
    plt.legend(loc='upper right')
    plt.grid()

    plt.savefig("./plots/A03_calc_uncertainty_t0_integrated.png", dpi=300)

    plt.show()
