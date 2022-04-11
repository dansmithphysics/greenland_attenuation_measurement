import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import analysis_funcs
import experiment


def rho(z):
    """ From Cosmin's paper """
    if(z <= 14.9):
        return 0.917 - 0.594 * np.exp(-1.0 * z / 30.8)
    else:
        return 0.917 - 0.367 * np.exp(-1.0 * (z - 14.9) / 40.5)


def n(z):
    """ From Cosmin's paper """
    return np.abs(1.0 + 0.845 * rho(z))


def calculate_n(n_avg_top_1k, z):
    """ Estimation to increase simulation rate """
    return (n_avg_top_1k * (1.78 / 1.774865) * 200.0 + 1.78 * (z - 200.0)) / z


def get_average_n_of_top_1k():
    n_avg_top_1k, steps_top_1k = 0, 0
    z, t = 0, 0
    space_step = 0.1  # units of meters
    for i in range(1000000):
        z += space_step
        t += space_step / (0.3 / n(z))

        if(z < 200.0):
            n_avg_top_1k += n(z)
            steps_top_1k += 1
        else:
            break

    n_avg_top_1k = float(n_avg_top_1k / float(steps_top_1k))

    return n_avg_top_1k


def main(exper_constants, exper):

    # Time corrections.
    # I don't use the time offset as calculated in exper_constants
    # because this is a different, more accurate experimental setup.
    t0_from_data = 0.035e-6
    t_tx = 74.2e-9
    t_rx = 74.5e-9
    prop_time_delay = 8.0 / 3e8  # measured by eye
    time_offset = prop_time_delay - t0_from_data - t_rx

    window_length = 250

    file_names = glob.glob("./data_processed/averaged_in_ice_trace_biref.npz")

    data_time, data_trace = analysis_funcs.load_file(file_name=file_names[0],
                                                     att_correction=0,
                                                     time_offset=time_offset)    
    
    data_time, rolling = analysis_funcs.power_integration(data_time, data_trace, window_length)

    noise = rolling[np.logical_and(data_time > 25.0e-6, data_time < 32.0e-6)]

    entries, cumsum = analysis_funcs.calculate_uncertainty(np.abs(noise))
    threshold = entries[np.argmin(np.abs(cumsum - 0.95))]

    plt.figure()
    plt.semilogy(data_time * 1e6, rolling,
                 color='black', linewidth=1.0)
    plt.axvline(exper_constants.gb_start * 1e6, color='purple', alpha=0.85, label="Arrival of Bedrock Echo")
    plt.axhline(threshold, color='red', linestyle="--", label="95% CI of Noise")

    plt.grid()
    plt.title("Ice Echo Data From Bed Rock, Integrated in Sliding 100 ns Window \n From Birefringence Data Run")
    plt.xlabel(r"Absolute Time Since Pulser [$\mu$s]")
    plt.ylabel("Integrated Power [mW ns]")
    plt.xlim(33.0, 38.0)
    plt.ylim(1e-7, 1e-4)

    x_ticks = np.arange(33, 39, 1)
    x_ticks = np.append(x_ticks,
                        exper_constants.gb_start * 1e6)
    x_ticks = np.sort(x_ticks)
    x_tick_labels = [str(int(tick)) if int(tick) == tick else str(tick) for tick in x_ticks]
    plt.xticks(x_ticks, x_tick_labels)
    plt.legend(loc='upper left')

    plt.savefig("./plots/A03_calc_uncertainty_depth_time_integrate.png",
                dpi=300)

    #######################################################################
    #                                                                     #
    # First half of the script is to find the ground bounce time          #
    #                                                                     #
    # Second half is to calculate the uncertainty from ice model of depth #
    #                                                                     #
    #######################################################################

    nthrows = 10000

    # get average index of refraction
    n_avg_top_1k = get_average_n_of_top_1k()

    # For each iteration of the toy mc,
    # the index of refraction of deep ice is drawn from
    # the distribution 1.78+/-0.03.
    # The ray tracing is then performed,
    # and the value of the depth at gb_t0 is returned.

    # This is turned into a distribution to calculate the
    # uncertainty on the bedrock depth.

    entries = np.zeros(nthrows)
    depths = np.linspace(2600.0, 3400.0, 1000)
    for ithrow in np.arange(nthrows):
        n_with_uncertainty = np.random.normal(calculate_n(n_avg_top_1k, depths), 0.03)
        times = (2.0 * depths) * (n_with_uncertainty / exper_constants.c)
        f = scipy.interpolate.interp1d(times, depths, kind='linear')
        entries[ithrow] = f(exper_constants.gb_start)

    entries, cumsum = analysis_funcs.calculate_uncertainty(entries)

    entries_min, entries_mid, entries_max = analysis_funcs.return_confidence_intervals(entries, cumsum)

    print("%f (+%f)(-%f)" % (entries_mid,
                             entries_mid - entries_min,
                             entries_max - entries_mid))

    plt.figure()
    plt.plot(entries, cumsum, color='black', alpha=0.75)
    plt.axvline(entries_min, color='red', linestyle='--')
    plt.axvline(entries_mid, color='red', linestyle='--',
                linewidth=2.0, label=r"Depth: $3004^{+50}_{-52}$ m")
    plt.axvline(entries_max, color='red', linestyle='--')
    plt.title("Depth from Time of Flight \n through Ice of $n_{asymp.} = 1.78 \pm 0.03$")
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlim(3050 - 300, 3050 + 300)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Depth [m]")
    plt.ylabel("CDF")

    plt.savefig("./plots/A03_calc_uncertainty_plots_time_to_depth.png",
                dpi=300)

    plt.show()


if __name__ == "__main__":

    ice_file_name = "./data_processed/averaged_in_ice_trace.npz"
    air_file_name = "./data_processed/averaged_in_air_trace.npz"
    
    exper_constants = experiment.ExperimentConstants()
    exper = experiment.Experiment(exper_constants, ice_file_name, air_file_name)    

    main(exper_constants, exper)
