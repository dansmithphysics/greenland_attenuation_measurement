import glob
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


def load(file_name, att_correction, time_offset):
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


if __name__ == "__main__":

    # correction to bring t0 to zero
    # corrects for cable delays, ect.
    time_offset = (35.55e-6 - 34.59e-6)

    data_time, data_trace = load("data_processed/averaged_in_ice_trace.npz",
                                 att_correction=0.0,
                                 time_offset=time_offset)

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
