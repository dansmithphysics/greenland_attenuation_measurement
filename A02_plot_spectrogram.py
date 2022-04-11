import glob
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import analysis_funcs
import experiment


def main(exper_constants, exper):

    data_time = exper.ice_time
    data_trace = exper.ice_trace
    
    fs = 1.0 / (data_time[1] - data_time[0]) / 1e6
    window_length = 1000

    noverlap = window_length - 1
    time_length = window_length * (data_time[1] - data_time[0]) * 1e9

    t0 = data_time[0] * 1e6

    xlim = (35.0, 40.0)  # Ground bounce

    fig1 = plt.figure(figsize=(16, 8))

    frame2 = fig1.add_axes((.1, .1, .64, .2))
    plt.plot(data_time * 1e6,
             data_trace * 1e3,
             color='black',
             linewidth=1.0)
    plt.xlim(xlim)
    plt.ylim(-1.0, 1.0)
    plt.xlabel(r"Absolute Time since Pulser [$\mu s$]")
    plt.ylabel("Time Trace [mV]")
    plt.grid()

    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    xticks_to_use = plt.xticks()

    frame1 = fig1.add_axes((.1, .33, .8, .6))

    plt.specgram(data_trace * 1e3,
                 Fs=fs,
                 NFFT=window_length,
                 noverlap=noverlap,
                 vmax=-10.0,
                 vmin=-60.0)

    cbar = plt.colorbar()
    cbar.set_label("[dBm]")

    plt.xlim(xlim[0] - t0,
             xlim[1] - t0)
    plt.ylim(50.0, 600.0)
    plt.ylabel("Frequency [MHz]")
    plt.grid()

    plt.xticks(xticks_to_use[0] - t0,
               ["" for i in range(len(xticks_to_use[0]))])

    plt.savefig("./plots/A02_plot_spectrogram.png",
                dpi=300,
                bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    ice_file_name = "./data_processed/averaged_in_ice_trace.npz"
    air_file_name = "./data_processed/averaged_in_air_trace.npz"
    
    exper_constants = experiment.ExperimentConstants()
    exper = experiment.Experiment(exper_constants, ice_file_name, air_file_name)    

    main(exper_constants, exper)
