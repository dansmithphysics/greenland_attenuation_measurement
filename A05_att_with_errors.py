import copy
import glob
import scipy.signal
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    #b, a = butter(order, [low, high], btype='band')
    b, a = butter(order, low, btype='high')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def open_data(file_name, att_correction):
    data = np.load(file_name)
    time = data["template_time"]
    trace = data["template_trace"]
    fs = 1.0 / (time[1] - time[0])
    trace *= np.power(10.0, att_correction / 20.0)
    fft = np.fft.rfft(trace)
    freq = np.fft.rfftfreq(len(trace), 1.0 / fs)

    return time, trace, freq, fft, fs



def calculate_att(t0, t1, T_ratio, R, focusing_factor, ice_prop, noise_start, ice_time, ice_trace, master_time, master_freq, ice_freq, Pxx_air, air_prop):
    
    window_mine = scipy.signal.windows.tukey(np.sum(np.logical_and(ice_time * 1e6 > t0, ice_time * 1e6 < t1)), alpha = 0.25)
    
    ground_bounce = window_mine * ice_trace[np.logical_and(ice_time * 1e6 > t0, ice_time * 1e6 < t1)]
    ground_bounce = np.append(ground_bounce, np.zeros(len(master_time) - len(ground_bounce)))
    
    ground_bounce_fft = np.fft.rfft(ground_bounce)
    ground_bounce_freq = copy.deepcopy(master_freq)
    
    Pxx_ice = np.abs(np.square(np.fft.rfft(ground_bounce)))
    freqs = copy.deepcopy(master_freq)

    span = (t1 - t0)

    window_mine = scipy.signal.windows.tukey(np.sum(np.logical_and(ice_time * 1e6 > noise_start, ice_time * 1e6 < noise_start + span)))
    
    noise = window_mine * ice_trace[np.logical_and(ice_time * 1e6 > noise_start, ice_time * 1e6 < noise_start + span)]
    noise = np.append(noise, np.zeros(len(master_time) - len(noise)))

    noise_fft = np.fft.rfft(noise)
    noise_freq = copy.deepcopy(master_freq)

    Pxx_noise = np.abs(np.square(np.fft.rfft(noise)))
    freqs = copy.deepcopy(master_freq)
    
    Pxx_ice -= Pxx_noise # get rid of noise power
    Pxx_ice /= focusing_factor

    att_corr = ice_prop / np.log(T_ratio * np.sqrt(R) * (np.sqrt(Pxx_air) * air_prop) / (np.sqrt(Pxx_ice) * ice_prop))

    att_corr[T_ratio * np.sqrt(R) * (np.sqrt(Pxx_air) * air_prop) < (np.sqrt(Pxx_ice) * ice_prop)] = 0.0 #np.random.uniform(0.0, 500.0)
    
    att_corr[Pxx_ice <= 0.0] = 0.0 #np.random.uniform(0.0, 500.0)
    att_corr[Pxx_air <= 0.0] = 1e10
    
    selection_range = np.logical_and(0.145e9 < master_freq, 0.575e9 > master_freq)
    plt.plot(master_freq[selection_range] * 1e-6, att_corr[selection_range], color = 'black', alpha = 0.01)
    
    return att_corr

def main():
    
    ice_time, ice_trace, ice_freq, ice_fft, ice_fs = open_data("./data_processed/averaged_in_ice_trace.npz", 0.0)
    air_time, air_trace, air_freq, air_fft, air_fs = open_data("./data_processed/averaged_in_air_trace.npz", 46.0) 

    air_prop = 244.0
    
    master_time = np.arange(0.6e-6 * ice_fs) / ice_fs
    master_freq = np.fft.rfftfreq(len(master_time), 1.0 / ice_fs)

    air_trace = butter_bandpass_filter(air_trace, 100.0e6, 1000.0e6, air_fs, order = 12)

    air_f = scipy.interpolate.interp1d(air_time, air_trace,
                                       kind = 'linear', bounds_error = False, fill_value = 0.0)

    air_trace = np.array(air_f(master_time + air_time[0]))
    air_time = copy.deepcopy(master_time + air_time[0]) #ice_time)

    air_fft = np.fft.rfft(air_trace)
    air_fs = ice_fs

    Pxx_air = np.abs(np.square(np.fft.rfft(air_trace)))
    freqs = copy.deepcopy(master_freq)

    t0_noise = 22.0
    t1_noise = 34.0
    
    #noise = np.zeros(len(ice_trace))
    noise = ice_trace[np.logical_and(ice_time * 1e6 > t0_noise, ice_time * 1e6 < t1_noise)]
    #noise = np.append(noise, np.zeros(len(master_time) - len(noise)))
    
    noise_fft = np.fft.rfft(noise)
    noise_freq = copy.deepcopy(ice_freq)
    
    Pxx_noise = np.abs(np.square(np.fft.rfft(noise)))
    #freqs = copy.deepcopy(ice_freq)

    Pxx_noise /= (t1_noise - t0_noise)

    # t0 E (34.59 - 38.0) uniform
    # t1 E (34.59 - 38.0) uniform
    # R E (0.1, 1.0) uniform
    # T_Ratio = 1.0 +/- 0.1 gaussian
    # focusing_factor E (1.0, 1.5) uniform
    # ice_prop E (6000.0, 6100.0)
    
    nthrows = 1000
    t0_ = np.ones(nthrows) * 35.6
    t1_ = np.ones(nthrows) * 36.1
    
    R_ = np.power(10.0, np.random.uniform(np.log10(0.1), np.log10(1.0), nthrows))
    focusing_factor_ = np.random.normal(1.24, 0.02, nthrows)
    ice_prop = np.random.normal(6090.0, 80.0, nthrows)
    T_ratio = np.ones(nthrows) 
    noise_start = np.random.uniform(t0_noise, t1_noise, nthrows)
    
    #backward = np.random.uniform(10.0, 9.0, nthrows)
    #t0_ -= backward
    #t1_ -= backward
    
    ##################

    hist_entry = []
    hist_freq = []

    plt.figure()
    plt.plot([], [], color = 'black', alpha = 0.1, label = "Corr. Att.")
    for i_throw in range(nthrows):
        print(i_throw)
        
        att_corr = calculate_att(t0 = t0_[i_throw],
                                 t1 = t1_[i_throw],
                                 T_ratio = T_ratio[i_throw],
                                 R = R_[i_throw],
                                 focusing_factor = focusing_factor_[i_throw],
                                 ice_prop = ice_prop[i_throw],
                                 noise_start = noise_start[i_throw],
                                 ice_time = ice_time,
                                 ice_trace = ice_trace,
                                 master_time = master_time,
                                 master_freq = master_freq,
                                 ice_freq = ice_freq,
                                 Pxx_air = Pxx_air,
                                 air_prop = air_prop)
    
        for i in range(len(att_corr)):

            if(np.isnan(att_corr[i]) or
               np.isinf(att_corr[i])):
                print(att_corr[i])
                continue
        
            hist_entry += [att_corr[i]]
            hist_freq += [freqs[i]]
        
    plt.errorbar([75.0], [947.0], yerr = [90.0], color = 'purple')
    plt.scatter([75.0], [947.0], color = 'purple', label = "Avva et al. Result")

    # correct this bad boy:
    d_ice_avva = 3015.0
    thing_in_natural_log = np.exp(d_ice_avva / 947.0)
    thing_in_natural_log *= 1.13
    result = d_ice_avva / np.log(thing_in_natural_log)
    
    thing_in_natural_log = np.exp(d_ice_avva / (947.0 + 90.0))
    thing_in_natural_log *= 1.13
    result_err = d_ice_avva / np.log(thing_in_natural_log)
    result_err -= result
    
    plt.errorbar([75.0], [result], yerr = [result_err], color = 'green', alpha = 0.5)
    plt.scatter([75.0], [result], color = 'green', alpha = 0.5, label = "Avva et al. Result, Corrected")

    new_freqs = np.linspace(0e6, 605e6, 45)
    #new_freqs = np.linspace(145e6, 575e6, 32)
    for new_freq in new_freqs:
        plt.axvline(new_freq * 1e-6, color = 'black', alpha = 0.25, linestyle = '-', linewidth = 1.0)

    plt.axvline(145.0, color = 'red', linestyle = '--', label = "HP / LP Filter")
    plt.axvline(575.0, color = 'red', linestyle = '--')

    plt.xlim(0.0, 600.0)
    plt.ylim(500.0, 1100.0)
    plt.xlabel("Freq. [MHz]")
    plt.ylabel("Att. [m]")
    plt.grid()
    plt.legend(loc = 'upper right')
    
    plt.show()
    
    plt.close()
    plt.figure()
    
    hist_freq = np.array(hist_freq)
    hist_entry = np.array(hist_entry)
    
    plt.hist2d(hist_freq * 1e-6, hist_entry, range = ((0.0, 700.0), (400.0, 1100.0)), bins = (70, 50))
    
    plt.close()

    x = []
    y = []
    yerr_min = []
    yerr_max = []

    unique_freqs = np.unique(hist_freq)

    #new_freqs = np.linspace(np.min(unique_freqs), np.max(unique_freqs), 128) #32)
    #new_freqs = np.linspace(145e6, 575e6, 10) #32)
    #new_freqs = np.linspace(0e6, 750e6, 40) #32)
    #new_freqs = np.linspace(145e6, 575e6, 32)
    new_freqs = np.linspace(0e6, 605e6, 45)
    #new_freqs = np.linspace(145e6, 575e6, 32)
    #new_freqs = np.linspace(145e6, 575e6, 25) #32)
    
    #print(new_freqs)
    #print(np.min(unique_freqs), np.max(unique_freqs), len(unique_freqs))
    #print(unique_freqs)

    plt.figure()

    #for i_unique_freq, unique_freq in enumerate(unique_freqs):
    for i_unique_freq, unique_freq in enumerate(new_freqs[:-1]):
        selection_region = np.logical_and(hist_freq > new_freqs[i_unique_freq], hist_freq < new_freqs[i_unique_freq + 1])
        hist_entry_ = hist_entry[selection_region]
        hist_entry_ = np.sort(hist_entry_)
        
        cumsum = np.cumsum(np.arange(len(hist_entry_)))
        try:
            cumsum = np.array(cumsum) / float(cumsum[-1])
        except:
            continue
    
        # upper bound
        cumsum_min = np.argmin(np.abs(cumsum - (0.5 - 0.341)))

        # lower bound    
        cumsum_max = np.argmin(np.abs(cumsum - (0.5 + 0.341)))
        
        # middle
        cumsum_middle = np.argmin(np.abs(cumsum - 0.5))

        if(hist_entry_[np.argmin(np.abs(cumsum - 0.05))] == 0):
            cumsum_min = 0
            cumsum_middle = np.argmin(np.abs(cumsum - 0.95))
            cumsum_max = np.argmin(np.abs(cumsum - 0.95))
    
        x += [(new_freqs[i_unique_freq] + new_freqs[i_unique_freq + 1]) / 2.0] #unique_freq]
        y += [hist_entry_[cumsum_middle]]
        yerr_min += [hist_entry_[cumsum_min]]
        yerr_max += [hist_entry_[cumsum_max]]
        
        if(i_unique_freq == 10):
            plt.plot(hist_entry_, cumsum, label = str(round(unique_freq * 1e-6)) + " MHz")
            plt.axvline(hist_entry_[cumsum_min], color = 'red', linewidth = 1.0, label = r"$\pm \sigma$ range")
            plt.axvline(hist_entry_[cumsum_max], color = 'red', linewidth = 1.0)
            plt.axvline(hist_entry_[cumsum_middle], color = 'red', linewidth = 2.0)

    plt.xlim(500.0, 1100.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Att. Length [m]")
    plt.ylabel("CDF")
    plt.grid()
    plt.legend()

    x = np.array(x)
    y = np.array(y)
    yerr_min = np.array(yerr_min)
    yerr_max = np.array(yerr_max)
    yerr = np.array([yerr_min, yerr_max])
    
    #np.savez("A06_mc_results", freqs = x, low_bound = yerr_min, high_bound = yerr_max, middle_val = y)
    
    plt.figure()
    
    plt.errorbar(x[yerr_min != 0] * 1e-6,
                 y[yerr_min != 0],
                 yerr = (y[yerr_min != 0] - yerr_min[yerr_min != 0], yerr_max[yerr_min != 0] - y[yerr_min != 0]),
                 color = 'black', ls = 'none',
                 label = "Data Result with 1 $\sigma$ Errors")

    plt.errorbar(x[yerr_min == 0] * 1e-6,
                 y[yerr_min == 0],
                 yerr = yerr_max[yerr_min == 0] - 550.0,
                 uplims = True, color = 'black', ls = 'none',
                 label = "95% CL Upper Limit")

    plt.scatter(x[yerr_min != 0] * 1e-6,
                y[yerr_min != 0],
                color = 'black')

    plt.scatter(x[yerr_min == 0] * 1e-6,
                y[yerr_min == 0],
                marker = "_",
                color = 'black')
    
    plt.errorbar([75.0], [947.0], yerr = [90.0], color = 'purple')
    plt.scatter([75.0], [947.0], color = 'purple', label = "Avva et al. Result")
        
    plt.errorbar([75.0], [result], yerr = [result_err], color = 'green', alpha = 0.5)
    plt.scatter([75.0], [result], color = 'green', alpha = 0.5, label = "Avva et al. Result, Corrected")
    
    plt.axvline(145.0, color = 'red', linestyle = '--', label = "HP / LP Filter")
    plt.axvline(575.0, color = 'red', linestyle = '--')
    
    plt.xlim(0.0, 600.0)
    plt.ylim(500.0, 1100.0)
    plt.xlabel("Freq. [MHz]")
    plt.ylabel("Att. Length [m]")
    plt.legend(loc = "upper right")
    
    plt.grid()
    plt.show()


    
    
    
if __name__ == "__main__":
    main()
