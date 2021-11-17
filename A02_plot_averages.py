import glob
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

def load_and_plot(file_name, att_correction, title, xlim, ylim, save_name):
    try:
        data_ice = np.load(file_name)
    except FileNotFoundError:
        print("File not found: %s" % file_name)
        return 
    
    data_t = data_ice["template_time"]
    data_trace = data_ice["template_trace"]

    data_trace *= np.power(10.0, att_correction / 20.0)
    
    plt.figure()
    plt.title(title)
    plt.plot(data_t * 1e6, data_trace * 1e3, alpha = 1.0, color = 'black', linewidth = 1.0)
    plt.xlabel("Time [$\mu$s]")
    plt.ylabel("Voltage [mV]")
    if(xlim != None):
        plt.xlim(xlim) 
    if(ylim != None):
        plt.ylim(ylim)
    plt.grid()
    if(save_name != None):
        plt.savefig(save_name, dpi = 300)

def load_and_plot_sliding_power(file_name, att_correction, title, xlim, ylim, save_name, window_length = 250):
    try:
        data_ice = np.load(file_name)
    except FileNotFoundError:
        print("File not found: %s" % file_name)
        return 
    
    data_t = data_ice["template_time"]
    data_trace = data_ice["template_trace"]

    data_trace *= np.power(10.0, att_correction / 20.0)

    data_power_mW = np.power(data_trace * 1e3, 2.0) / 50.0
    time_length = window_length * (data_t[1] - data_t[0]) * 1e9

    window = scipy.signal.windows.tukey(window_length, alpha = 0.25)
    
    rolling = np.convolve(data_power_mW / time_length,
                          window,
                          'valid')

    data_t = (data_t[(window_length - 1):] + data_t[:-(window_length - 1)]) / 2.0
    
    plt.figure()
    plt.title(title)
    plt.semilogy(data_t * 1e6, rolling, alpha = 1.0, color = 'black', linewidth = 1.0)
    plt.xlabel("Time [$\mu$s]")
    plt.ylabel("Integrated Power [mW / ns]")
    if(xlim != None):
        plt.xlim(xlim) 
    if(ylim != None):
        plt.ylim(ylim) 
    plt.grid()
    if(save_name != None):
        plt.savefig(save_name, dpi = 300)

if __name__ == "__main__":

    load_and_plot("data_processed/averaged_in_ice_trace.npz",
                  att_correction = 0.0,
                  title = "Ice Echo Data",
                  xlim = (-1.0, 40.0),
                  ylim = (-10.0, 10.0),
                  save_name = "./plots/A02_plot_averages_inice_zoomed_out.png")

    load_and_plot("data_processed/averaged_in_ice_trace.npz",
                  att_correction = 0.0,
                  title = "Ice Echo Data",
                  xlim = (-1.0, 40.0),
                  ylim = (-1.0, 1.0),
                  save_name = "./plots/A02_plot_averages_inice_zoomed_in.png")

    load_and_plot("data_processed/averaged_in_ice_trace.npz",
                  att_correction = 0.0,
                  title = "Ice Echo Data from Bed Rock",
                  xlim = (30.0, 40.0),
                  ylim = (-0.75, 0.75),
                  save_name = "./plots/A02_plot_averages_inice_ground_bounce.png")
    
    load_and_plot_sliding_power("data_processed/averaged_in_ice_trace.npz",
                                att_correction = 0.0,
                                title = "Ice Echo Data from Bed Rock, Integrated in Sliding Window",
                                xlim = (33.0, 38.0),
                                ylim = (1e-5, 1e-2),
                                save_name = "./plots/A02_plot_averages_inice_ground_bounce_integrated.png")
    
    load_and_plot("data_processed/averaged_in_air_trace.npz",
                  att_correction = 0.0,
                  title = "Air-to-Air Result",
                  xlim = (-0.07, 0.4),
                  ylim = (-500.0, 500.0),
                  save_name = "./plots/A02_plot_averages_inair_uncorrected.png")
    
    plt.show()    
