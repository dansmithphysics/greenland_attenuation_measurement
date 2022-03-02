import copy
import glob
import scipy.signal
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit


def gaus_2d(X, sig_x, sig_y, rho, x0, y0):
    x, y = X
    norm = 2.0 * np.pi * sig_x * sig_y * np.sqrt(1.0 - rho * rho)
    return 1.0 / norm * np.exp(- 0.5 / (1.0 - rho * rho) * (np.square((x - x0) / sig_x) - 2.0 * rho * (x - x0) * (y - y0) / (sig_x * sig_y) + np.square((y - y0) / sig_y)))


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


def open_data(file_name, att_correction):
    data = np.load(file_name)
    time = data["template_time"]
    trace = data["template_trace"]
    fs = 1.0 / (time[1] - time[0])
    trace *= np.power(10.0, att_correction / 20.0)
    fft = np.fft.rfft(trace)
    freq = np.fft.rfftfreq(len(trace), 1.0 / fs)

    return time, trace, freq, fft, fs


def calculate_att(t0, t1, T_ratio, R, focusing_factor, ice_prop, noise_start, ice_time, ice_trace, master_time, master_freq, Pxx_air, Pxx_noise, air_prop):

    c = scipy.constants.c
    d_ice = 3004.0
    start_time = t0
    ice_prop = d_ice * 2.0

    n = start_time * 1e-6 * (c / d_ice / 2.0)  # Can be more precise here

    r_square = np.square(t1 * 1e-6 * (c / n) / 2.0) - np.square(d_ice)

    sigma = np.pi * r_square
    cheat_scale = 1.0 / sigma
    cheat_scale *= np.power(d_ice, 4.0) / np.square(ice_prop)

    cheat_scale = 1.0

    window_mine = scipy.signal.windows.tukey(np.sum(np.logical_and(ice_time * 1e6 > t0, ice_time * 1e6 < t1)), alpha=0.25)

    ground_bounce = window_mine * ice_trace[np.logical_and(ice_time * 1e6 > t0, ice_time * 1e6 < t1)]
    ground_bounce = np.append(ground_bounce, np.zeros(len(master_time) - len(ground_bounce)))

    #ground_bounce = butter_bandpass_filter(ground_bounce, 100.0e6, 600.0e6, 1.0 / (master_time[1] - master_time[0]), order=5)

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

    #Pxx_noise = np.abs(np.square(np.fft.rfft(noise)))
    freqs = copy.deepcopy(master_freq)
    
    '''
    plt.figure()
    plt.plot(master_freq * 1e-6, 10.0 * np.log10(Pxx_ice), label = "Bedrock Echo")
    plt.plot(master_freq * 1e-6, 10.0 * np.log10(Pxx_noise), label = "Noise")
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Spectral Power [dB]")
    plt.xlim(0.0, 1000.0)
    plt.grid()
    plt.legend()
    plt.show()
    exit()
    '''
    
    Pxx_noise *= (t1 - t0)        
    Pxx_ice -= Pxx_noise # get rid of noise power
    Pxx_noise /= (t1 - t0)
    
    Pxx_ice *= cheat_scale
    
    Pxx_air[Pxx_air < 0.0] = 0.0
    Pxx_ice[Pxx_ice < 0.0] = 0.0
    Pxx_ice[Pxx_ice == 0.0] = 1e-10
    att_corr = ice_prop / np.log(T_ratio * np.sqrt(R * focusing_factor) * (np.sqrt(Pxx_air) * air_prop) / (np.sqrt(Pxx_ice) * ice_prop))
    
    att_corr[T_ratio * np.sqrt(R * focusing_factor) * (np.sqrt(Pxx_air) * air_prop) < (np.sqrt(Pxx_ice) * ice_prop)] = 0.0 #np.random.uniform(0.0, 500.0)
    
    att_corr[Pxx_ice == 1e-10] = 0.0 #np.random.uniform(0.0, 500.0)
    att_corr[Pxx_air <= 0.0] = 1e10
    
    selection_range = np.logical_and(0.145e9 < master_freq, 0.575e9 > master_freq)
    
    return att_corr

#def main(i_run):
def main():
    
    # load up the data
    #ice_time, ice_trace, ice_freq, ice_fft, ice_fs = open_data("./data_processed/averaged_in_ice_trace_%i.npz" % i_run, 0.0)
    ice_time, ice_trace, ice_freq, ice_fft, ice_fs = open_data("./data_processed/averaged_in_ice_trace.npz", 0.0)
    air_time, air_trace, air_freq, air_fft, air_fs = open_data("./data_processed/averaged_in_air_trace.npz", 46.0) 
    
    # correct to absolute time
    ice_time -= 34.59e-6
    ice_time += 35.55e-6

    #plt.plot(ice_time, ice_trace)
    #plt.show()
    #exit()

    #ice_trace = butter_bandpass_filter(ice_trace, 100.0e6, 600.0e6, 1.0 / (ice_time[1] - ice_time[0]), order=5)
    
    #master_time = np.arange(0.6e-6 * ice_fs) / ice_fs
    master_time = np.arange(3.0e-6 * ice_fs) / ice_fs
    #master_time = np.arange(6.0e-6 * ice_fs) / ice_fs
    master_freq = np.fft.rfftfreq(len(master_time), 1.0 / ice_fs)

    #air_trace = butter_bandpass_filter(air_trace, 100.0e6, 1000.0e6, air_fs, order = 12)

    air_f = scipy.interpolate.interp1d(air_time, air_trace,
                                       kind='linear', bounds_error=False, fill_value=0.0)

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

    f_noise = scipy.interpolate.interp1d(np.fft.rfftfreq(len(noise), 1.0 / ice_fs), Pxx_noise,
                                         kind='linear', bounds_error=False, fill_value=0.0)
    Pxx_noise = f_noise(master_freq)
    
    Pxx_noise /= (t1_noise - t0_noise)
    
    # t0 = 35.6
    # t1 = 36.1
    # R E (0.1, 1.0) uniform in log
    # T_Ratio = 1.05 +/- 0.05 
    # focusing_factor = 1.24 +/- 0.02 
    # ice_prop = 6090.0 +/- 80.0
    # air_prop = 244.0 +/- 1.0
    
    nthrows = 1000
    #nthrows = 10
    t0_ = np.ones(nthrows) * 35.55
    t1_ = np.ones(nthrows) * 36.05
    #t1_ = np.ones(nthrows) * (35.55 + 2.5)
    
    #t1_ = np.ones(nthrows) * 39.0
    #t1_ = np.ones(nthrows) * 38.0

    #t0_ = np.ones(nthrows) * 35.6
    #t1_ = np.ones(nthrows) * 36.1
    
    #t1_ = np.ones(nthrows) * 35.7
    #t0_ = np.ones(nthrows) * 10.33
    #t1_ = np.ones(nthrows) * 10.40   

    R_ = np.power(10.0, np.random.uniform(np.log10(0.1), np.log10(1.0), nthrows))
    #focusing_factor_ = np.square(np.random.normal(1.27, 0.09, nthrows))
    focusing_factor_ = np.random.normal(1.61, 0.24, nthrows)
    ice_prop = np.random.normal(6008.0, 100.0, nthrows)
    #ice_prop = np.random.normal(1762.0, 30.0, nthrows)
    air_prop = np.random.normal(244.0, 1.0, nthrows)
    T_ratio = np.random.normal(1.00, 0.05, nthrows)
    noise_start = np.random.uniform(t0_noise, t1_noise, nthrows)

    '''
    span = (t1_[0] - t0_[0])

    noise_rms = []

    window_mine = scipy.signal.windows.tukey(np.sum(np.logical_and(ice_time * 1e6 > t0_[0], ice_time * 1e6 < t1_[0])), alpha = 0.25)    
    ground_bounce = window_mine * ice_trace[np.logical_and(ice_time * 1e6 > t0_[0], ice_time * 1e6 < t1_[0])]
    ground_bounce = np.append(ground_bounce, np.zeros(len(master_time) - len(ground_bounce)))
    ground_bounce *= 1e3
    Pxx_ice = np.abs(np.square(np.fft.rfft(ground_bounce))) / 50.0
    
    plt.figure()
    for i in range(nthrows):
        window_mine = scipy.signal.windows.tukey(np.sum(np.logical_and(ice_time * 1e6 > noise_start[i], ice_time * 1e6 < noise_start[i] + span)))
    
        noise = window_mine * ice_trace[np.logical_and(ice_time * 1e6 > noise_start[i], ice_time * 1e6 < noise_start[i] + span)]
        noise = np.append(noise, np.zeros(len(master_time) - len(noise)))
        
        noise_fft = np.fft.rfft(noise * 1e3)
        noise_freq = copy.deepcopy(master_freq)
        
        Pxx_noise = np.abs(np.square(np.fft.rfft(noise * 1e3))) / 50.0
        #plt.plot(noise_freq, (Pxx_noise), color = 'black', alpha = 0.01)
        #plt.plot(noise_freq, Pxx_noise / Pxx_ice, color = 'black', alpha = 0.01)
        #noise_rms += [Pxx_noise[2000]]
        noise_rms += [(np.mean(Pxx_noise[1500]))]
        #plt.plot(10.0 * np.log10(Pxx_noise), color = 'black', alpha = 0.01)
        #plt.hist(np.log10(noise_rms), range = (-10.0, 0.0), bins = 100)

    plt.hist(noise_rms)#, range = (0.5e-6, 2.5e-6), bins = 20)
    plt.axvline(np.mean(noise_rms), color = 'red')
    plt.axvline(np.mean(noise_rms) + np.std(noise_rms), color = 'green')
    plt.axvline(np.mean(noise_rms) - np.std(noise_rms), color = 'green')
    #plt.plot(master_freq, (Pxx_ice), color = 'red')
    plt.axvline((np.mean(Pxx_ice[1500])))
    plt.show()
    exit()
    '''
    
    #backward = np.random.uniform(10.0, 9.0, nthrows)
    #t0_ -= backward
    #t1_ -= backward
    
    #######################
    # Starting the toy MC #
    #######################

    scale_to_top_1p5k = 1.1997683538791095
    #scale_to_top_1p5k = 1.0
    hist_freq, hist_entry = [], []

    # linear fit values
    ms, bs = [], []

    plt.figure()
    
    for i_throw in range(nthrows):        
        att_corr = calculate_att(t0=t0_[i_throw],
                                 t1=t1_[i_throw],
                                 T_ratio=T_ratio[i_throw],
                                 R=R_[i_throw],
                                 focusing_factor=focusing_factor_[i_throw],
                                 ice_prop=ice_prop[i_throw],
                                 noise_start=noise_start[i_throw],
                                 ice_time=ice_time,
                                 ice_trace=ice_trace,
                                 master_time=master_time,
                                 master_freq=master_freq,
                                 Pxx_air=Pxx_air,
                                 Pxx_noise=Pxx_noise,
                                 air_prop=air_prop[i_throw])

        selection_region = np.logical_and(freqs > 140e6, freqs < 400e6)
        #selection_region = np.logical_and(freqs > 140e6, freqs < 300e6)
        #selection_region = np.logical_and(freqs > 140e6, freqs < 575e6)
        #selection_region = np.logical_and(freqs > 140e6, freqs < 575e6)
        selection_region = np.logical_and(selection_region, np.logical_not(np.isnan(att_corr)))
        selection_region = np.logical_and(selection_region, np.logical_not(np.isinf(att_corr)))
        selection_region = np.logical_and(selection_region, att_corr > 100.0)
        selection_region = np.logical_and(selection_region, att_corr < 2000.0)

        #plt.plot(freqs * 1e-6, att_corr, color = 'black', alpha = 0.05)
        
        freqs_to_fit = freqs[selection_region]
        att_corr_to_fit = att_corr[selection_region]

        def func(x, a, b):
            return a / x + b
        
        #popt, pcov = curve_fit(func, freqs_to_fit / 1e8, (att_corr_to_fit * scale_to_top_1p5k))

        #plt.close()
        #plt.figure()
        #plt.scatter(freqs_to_fit, (att_corr_to_fit * scale_to_top_1p5k))
        #plt.plot(freqs_to_fit, func(freqs_to_fit / 1e8, *popt))
        #plt.plot(freqs_to_fit, func(freqs_to_fit, 100.0e8, 0))
        #plt.show()
        #print(popt)
        #exit()

        #m = popt[0]
        #b = popt[1]        

        m, b = np.polyfit(freqs_to_fit, (att_corr_to_fit * scale_to_top_1p5k), 1)
        
        ms += [m]
        bs += [b]
        
        print(i_throw, ")", m * 1e6, "m / MHz", b, "m")
        
        for i in range(len(att_corr)):
            if(np.isnan(att_corr[i]) or
               np.isinf(att_corr[i])):
                print(freqs[i] * 1e-6, att_corr[i])
                continue
        
            hist_entry += [att_corr[i]]
            hist_freq += [freqs[i]]
            
    #p0: [54.43315087176151, 42.991425234399394, -0.95, -652.3704196033967, 885.2049447006741]

    plt.xlim(50.0, 600.0)
    plt.ylim(500.0, 1100.0)
    #plt.ylim(600.0, 1250.0)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel(r"Bulk Field Attenuation Length, $\langle L_\alpha\rangle$ [m]")
    #plt.ylabel(r"Avg. Field Attenuation Length of Top 1500 m [m]")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.grid()

    #plt.show()
    #exit()
            
    ###################################################
    # loading up and correcting the avva et al result #
    ###################################################
    
    d_ice_avva = 3015.0 * 2.0
    thing_in_natural_log = np.exp(d_ice_avva / 947.0)
    thing_in_natural_log *= np.sqrt(np.square(1.27))
    result = d_ice_avva / np.log(thing_in_natural_log) 
    
    thing_in_natural_log = np.exp(d_ice_avva / (947.0 + 92.0))
    thing_in_natural_log *= np.sqrt(np.square(1.27))
    result_err = d_ice_avva / np.log(thing_in_natural_log)
    result_err -= result
    print("Up:", result_err)
    thing_in_natural_log = np.exp(d_ice_avva / (947.0 - 85.0))
    thing_in_natural_log *= np.sqrt(np.square(1.27))
    result_err = d_ice_avva / np.log(thing_in_natural_log)
    result_err -= result
    print("Down:", result_err)
    
    print(947, "vs", result)
    #exit()
    #exit()

    #print(100, "
    
    new_freqs = np.linspace(150e6, 566.6666e6, 26)
    #new_freqs = np.linspace(100e6, 566.6666e6, 29)

    #########################################################################################
    # Performing a fit on the linear fit values to get the 2d gaussian errors / correlation #
    #########################################################################################

    ms = np.array(ms) * 1e9
    bs = np.array(bs)

    H, xedges, yedges = np.histogram2d(ms, bs,
                                       range=((-1000.0, -200.0), (600.0, 1200.0)),
                                       bins=(1000, 1000),
                                       normed=True)

    xcenters = (xedges[:-1] + xedges[1:]) / 2.0
    ycenters = (yedges[:-1] + yedges[1:]) / 2.0

    X, Y = np.meshgrid(xcenters, ycenters)
    x = np.ravel(X)
    y = np.ravel(Y)
    z = np.ravel(H)

    x0 = np.mean(ms)
    y0 = np.mean(bs)
    
    p0 = [np.std(ms), np.std(bs), -0.95, x0, y0]

    #popt, pcov = curve_fit(gaus_2d,
    #                       (x, y), z,                           
    #                       p0,
    #                       ftol = 1e-10,
    #                       maxfev = 100000000)

    #print("Fit:", popt)
    print("p0:", p0)
    
    plt.figure()
    plt.hist2d(ms, bs,
               range=((-10e2, -2e2), (600.0, 1200.0)), bins=(100, 100))
    plt.colorbar()

    '''
    sig_x, sig_y, rho, x0, y0 = popt 
    X, Y = np.meshgrid(np.linspace(-10e2, -2e2, 5001), np.linspace(600.0, 1200.0, 5001))
    Z = gaus_2d((X, Y), sig_x, sig_y, rho, x0, y0)
    levels = np.array([#gaus_2d((x0 + 3.0 * sig_x, y0 + 3.0 * rho * sig_y), sig_x, sig_y, rho, x0, y0),
                       #gaus_2d((x0 + 2.0 * sig_x, y0 + 2.0 * rho * sig_y), sig_x, sig_y, rho, x0, y0),
                       gaus_2d((x0 + 1.0 * sig_x, y0 + 1.0 * rho * sig_y), sig_x, sig_y, rho, x0, y0)])
    plt.plot([], [], color = 'red', label = "1, 2, 3 $\sigma$ Contours, Fit")
    plt.contour(X, Y, Z, levels, colors = 'red')
    '''
    
    sig_x, sig_y, rho, x0, y0 = p0     
    X, Y = np.meshgrid(np.linspace(-10e2, -2e2, 5001), np.linspace(600.0, 1200.0, 5001))
    Z = gaus_2d((X, Y), sig_x, sig_y, rho, x0, y0)
    levels = np.array([#gaus_2d((x0 + 3.0 * sig_x, y0 + 3.0 * rho * sig_y), sig_x, sig_y, rho, x0, y0),
                       #gaus_2d((x0 + 2.0 * sig_x, y0 + 2.0 * rho * sig_y), sig_x, sig_y, rho, x0, y0),
                       gaus_2d((x0 + 1.0 * sig_x, y0 + 1.0 * rho * sig_y), sig_x, sig_y, rho, x0, y0)])
    plt.plot([], [], color='orange', label="1, 2, 3 $\sigma$ Contours, Initial Guess")
    plt.contour(X, Y, Z, levels, colors='orange')

    plt.ylabel("Intercept [m]")
    plt.xlabel("Slope [m / GHz]")
    plt.legend()

    ###################################################
    # Calculating the error bars from the Toy MC runs #
    ###################################################
    
    x, y = [], []
    yerr_min, yerr_max = [], []
    
    hist_freq = np.array(hist_freq)
    hist_entry = np.array(hist_entry)

    plt.figure()    
    for i_unique_freq, unique_freq in enumerate(new_freqs[:-1]):
        selection_region = np.logical_and(hist_freq > new_freqs[i_unique_freq], hist_freq < new_freqs[i_unique_freq + 1])
        hist_entry_ = hist_entry[selection_region]
        hist_entry_ = np.sort(hist_entry_)
        
        cumsum = np.cumsum(np.ones(len(hist_entry_)))
        try:
            cumsum = np.array(cumsum) / float(cumsum[-1])
        except:
            continue
    
        # upper bound
        cumsum_min = np.argmin(np.abs(cumsum - (0.5 - 0.341)))
        cumsum_max = np.argmin(np.abs(cumsum - (0.5 + 0.341)))
        cumsum_middle = np.argmin(np.abs(cumsum - 0.5))

        #if(hist_entry_[np.argmin(np.abs(cumsum - 0.05))] == 0):
        if(hist_entry_[np.argmin(np.abs(cumsum - 0.15))] < 100):
            #if(hist_entry_[np.argmin(np.abs(cumsum - 0.2))] < 100):
            #if(hist_entry_[np.argmin(np.abs(cumsum - 0.30))] < 100):
            cumsum_min = 0
            cumsum_middle = np.argmin(np.abs(cumsum - 0.95))
            cumsum_max = np.argmin(np.abs(cumsum - 0.95))

        x += [(new_freqs[i_unique_freq] + new_freqs[i_unique_freq + 1]) / 2.0] #unique_freq]
        if(cumsum_min != 0):
            y += [hist_entry_[cumsum_middle]]
            yerr_min += [hist_entry_[cumsum_min]]
            yerr_max += [hist_entry_[cumsum_max]]
        else:
            y += [hist_entry_[cumsum_max]]
            yerr_min += [0.0]
            yerr_max += [hist_entry_[cumsum_max]]            
            
        if(i_unique_freq == 10):
            plt.plot(hist_entry_, cumsum, label=str(round(unique_freq * 1e-6)) + " MHz")
            plt.axvline(hist_entry_[cumsum_min], color='red', linewidth=1.0, label=r"$\pm \sigma$ range")
            plt.axvline(hist_entry_[cumsum_max], color='red', linewidth=1.0)
            plt.axvline(hist_entry_[cumsum_middle], color='red', linewidth=2.0)

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
    
    #np.savez("./data_processed/A05_mc_results_scaled", freqs = x, low_bound = yerr_min, high_bound = yerr_max, middle_val = y)
    np.savez("./data_processed/A05_mc_results", freqs=x, low_bound=yerr_min, high_bound=yerr_max, middle_val=y)

    for i in range(len(x)):
        print(i, x[i], y[i], yerr_min[i], yerr_max[i])


    d_ice = 6008.0
        
    for i in range(len(x)):
        print(i,
              x[i],
              y[i],
              20.0 * np.log10(np.exp(-1.0 * d_ice / (y[i] * scale_to_top_1p5k))) / (d_ice / 1e3),
              20.0 * np.log10(np.exp(-1.0 * d_ice / (yerr_min[i] * scale_to_top_1p5k))) / (d_ice / 1e3) - 20.0 * np.log10(np.exp(-1.0 * d_ice / (y[i] * scale_to_top_1p5k))) / (d_ice / 1e3),
              20.0 * np.log10(np.exp(-1.0 * d_ice / (yerr_max[i] * scale_to_top_1p5k))) / (d_ice / 1e3) - 20.0 * np.log10(np.exp(-1.0 * d_ice / (y[i] * scale_to_top_1p5k))) / (d_ice / 1e3))
        

        
    #exit()
    
    plt.figure(figsize = (5, 4))
    plt.errorbar(x[yerr_min != 0] * 1e-6,
                 y[yerr_min != 0] * scale_to_top_1p5k,
                 yerr=((y[yerr_min != 0] - yerr_min[yerr_min != 0]) * scale_to_top_1p5k,
                       (yerr_max[yerr_min != 0] - y[yerr_min != 0]) * scale_to_top_1p5k),
                 color='black', ls='none')
                 #label = "Data Result with 1 $\sigma$ Errors")

    plt.errorbar([], [], yerr = [],
                 color='black', marker='o', ls='none',
                 label="Data Result with 1 $\sigma$ Errors")    
    
    plt.errorbar(x[yerr_min == 0] * 1e-6,
                 y[yerr_min == 0] * scale_to_top_1p5k,
                 yerr=(yerr_max[yerr_min == 0] - 550.0) * scale_to_top_1p5k,
                 uplims=True, color='black', ls='none',
                 label="95% CL Upper Limit")

    plt.scatter(x[yerr_min != 0] * 1e-6,
                y[yerr_min != 0] * scale_to_top_1p5k,
                color='black')

    plt.scatter(x[yerr_min == 0] * 1e-6,
                y[yerr_min == 0] * scale_to_top_1p5k,
                marker="_",
                color='black')

    fit_x, fit_y, fit_yerr_min, fit_yerr_max = [], [], [], []
    for i_unique_freq, unique_freq in enumerate(np.linspace(0.0, 1e9, 100)):
        hist_entry = ms * unique_freq * 1e-9 + bs
        #hist_entry = ms / unique_freq * 1e-9 * 1e8 + bs
        hist_entry_ = np.sort(hist_entry)
    
        cumsum = np.cumsum(np.ones(len(hist_entry_)))
        try:
            cumsum = np.array(cumsum) / float(cumsum[-1])
        except:
            continue

        cumsum_min = np.argmin(np.abs(cumsum - (0.5 - 0.341)))
        cumsum_max = np.argmin(np.abs(cumsum - (0.5 + 0.341)))
        cumsum_middle = np.argmin(np.abs(cumsum - 0.5))

        #cumsum_min = np.argmin(np.abs(cumsum - 0.025))
        #cumsum_max = np.argmin(np.abs(cumsum - 0.975))
        #cumsum_middle = np.argmin(np.abs(cumsum - 0.5))

        fit_x += [unique_freq]
        fit_y += [hist_entry_[cumsum_middle]]
        fit_yerr_min += [hist_entry_[cumsum_min]]
        fit_yerr_max += [hist_entry_[cumsum_max]]
    fit_x = np.array(fit_x)
    fit_y = np.array(fit_y)
    fit_yerr_min = np.array(fit_yerr_min)
    fit_yerr_max = np.array(fit_yerr_max)

    plt.fill_between(fit_x * 1e-6,
                     fit_yerr_min,
                     fit_yerr_max,
                     alpha=0.3,
                     color='red',
                     label=r"Linear Fit with 1 $\sigma$ Errors")
    plt.plot(fit_x * 1e-6,
             fit_y,
             color='red')

    plt.errorbar([75.0], [result * scale_to_top_1p5k],
                 yerr=[result_err * scale_to_top_1p5k], color='purple', alpha=0.75)
    plt.scatter([75.0], [result * scale_to_top_1p5k],
                color='purple', alpha=0.75, label="Avva et al. Result, Corrected")

    #plt.errorbar([75.0], [result * 1.13], yerr = [result_err * 1.13], color = 'purple', alpha = 0.75)
    #plt.scatter([75.0], [result * 1.13], color = 'purple', alpha = 0.75, label = "Avva et al. Result, Corrected")

    plt.axvline(145.0, color='red', linestyle='--', label="Bandpass Filters")
    plt.axvline(575.0, color='red', linestyle='--')

    if(scale_to_top_1p5k == 1.0):
        annotation_string = r"Fit: $\langle L_\alpha(\nu) \rangle = (852\pm41)$"
        annotation_string += "\n"
        annotation_string += r"$- (0.54\pm0.04) \nu$ m"
        plt.text(475, 800, annotation_string,
                 ha='center',
                 bbox=dict(boxstyle="round",
                           ec='black',
                           alpha = 0.85,
                           fc='w',
                           ))
    else:
        #p0: [55.46557013010805, 50.34254274500003, -0.95, -654.7093673745969, 1024.3790596358244]

        annotation_string = r"Fit: $\langle L_\alpha(\nu) \rangle = (1024\pm50)$"
        annotation_string += "\n"
        annotation_string += r"$- (0.65\pm0.06) \nu$ m"    
        plt.text(470, 800 * scale_to_top_1p5k, annotation_string,
                 ha='center',
                 bbox=dict(boxstyle="round",
                           ec='black',
                           alpha=0.90,
                           fc='w',
                           ))

    plt.xlim(50.0, 600.0)
    if(scale_to_top_1p5k == 1.0):
        plt.ylim(500.0, 1100.0)
    else:
        plt.ylim(600.0, 1300.0)
    plt.xlabel("Frequency [MHz]")

    if(scale_to_top_1p5k == 1.0):
        plt.ylabel(r"Bulk Field Attenuation Length, $\langle L_\alpha\rangle$ [m]")
    else:
        plt.ylabel(r"Avg. Field Attenuation Length of Top 1500 m [m]")

    plt.legend(loc = "upper right")
    plt.tight_layout()
    plt.grid()
    if(scale_to_top_1p5k == 1.0):
        plt.savefig("./plots/A05_att_with_errors_final.png",
                    dpi=300)
    else:
        plt.savefig("./plots/A05_att_with_errors_final_1500m.png",
                    dpi=300)

    plt.show()


if __name__ == "__main__":
    #for i_run in range(20):
    main() #i_run)
