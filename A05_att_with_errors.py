import copy
import glob
import scipy.signal
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt


def open_data(file_name, att_correction):
    data = np.load(file_name)
    time = data["template_time"]
    trace = data["template_trace"]
    fs = 1.0 / (time[1] - time[0])
    trace *= np.power(10.0, att_correction / 20.0)
    fft = np.fft.rfft(trace)
    freq = np.fft.rfftfreq(len(trace), 1.0 / fs)

    return time, trace, freq, fft, fs


def prepare_ice_signal(t_gb, ice_time, ice_trace, master_time, Pxx_noise):

    t0, t1 = t_gb

    gb_select = np.logical_and(ice_time > t0,
                               ice_time < t1)

    window_mine = scipy.signal.windows.tukey(np.sum(gb_select),
                                             alpha=0.25)

    ground_bounce = window_mine * ice_trace[gb_select]
    ground_bounce = np.append(ground_bounce,
                              np.zeros(len(master_time) - len(ground_bounce)))
    Pxx_ice = np.abs(np.square(np.fft.rfft(ground_bounce)))

    # Subtract the noise power from the ice signal.
    Pxx_noise *= (t1 - t0)
    Pxx_ice -= Pxx_noise
    Pxx_noise /= (t1 - t0)

    # If the power in the ice signal goes below zero
    # due to noise fluctuating high, set the ice signal
    # to a very small value.
    Pxx_ice[Pxx_ice < 0.0] = 1e-10

    return Pxx_ice


def calculate_att(T_ratio, R, focusing_factor,
                  Pxx_air, Pxx_ice, air_prop, ice_prop):

    # Calculates the attenuation.
    power_ratio = (np.sqrt(Pxx_air) * air_prop) / (np.sqrt(Pxx_ice) * ice_prop)
    corrections = T_ratio * np.sqrt(R * focusing_factor)
    att = ice_prop / np.log(corrections * power_ratio)

    att[Pxx_ice == 1e-10] = 0.0
    att[Pxx_air <= 0.0] = 1e10

    return att


def perform_mc(t_gb, t_noise, gb_duration=10e-6,
               nthrows=1000, time_offset=0.0):

    # Load up the data
    ice_time, ice_trace, ice_freq, ice_fft, ice_fs = open_data("./data_processed/averaged_in_ice_trace.npz",
                                                               att_correction=0.0)
    air_time, air_trace, air_freq, air_fft, air_fs = open_data("./data_processed/averaged_in_air_trace.npz",
                                                               att_correction=46.0)

    # Define a master time close,
    # based on the ice_data time steps.
    master_time = np.arange(gb_duration * ice_fs) / ice_fs
    master_freq = np.fft.rfftfreq(len(master_time), 1.0 / ice_fs)

    air_f = scipy.interpolate.interp1d(air_time, air_trace,
                                       kind='linear', bounds_error=False, fill_value=0.0)

    air_trace = np.array(air_f(master_time + air_time[0]))
    air_time = copy.deepcopy(master_time + air_time[0])

    air_fft = np.fft.rfft(air_trace)
    air_fs = ice_fs

    Pxx_air = np.abs(np.square(np.fft.rfft(air_trace)))
    freqs = copy.deepcopy(master_freq)

    t0_noise, t1_noise = t_noise
    noise = ice_trace[np.logical_and(ice_time > t0_noise,
                                     ice_time < t1_noise)]

    noise_fft = np.fft.rfft(noise)
    noise_freq = copy.deepcopy(ice_freq)

    Pxx_noise = np.abs(np.square(np.fft.rfft(noise)))
    f_noise = scipy.interpolate.interp1d(np.fft.rfftfreq(len(noise),
                                                         1.0 / ice_fs),
                                         Pxx_noise,
                                         kind='linear',
                                         bounds_error=False,
                                         fill_value=0.0)
    Pxx_noise = f_noise(master_freq)
    Pxx_noise /= (t1_noise - t0_noise)

    # Correct to absolute time
    ice_time += time_offset
    Pxx_ice = prepare_ice_signal(t_gb,
                                 ice_time,
                                 ice_trace,
                                 master_time,
                                 Pxx_noise)

    # t0 = 35.55
    # t1 = 36.05
    # R E (0.1, 1.0) uniform in log
    # focusing_factor = 1.61 +/- 0.24
    # ice_prop = 6008.0 +/- 100.0
    # air_prop = 244.0 +/- 1.0
    # T_Ratio = 1.05 +/- 0.05

    R_ = np.random.uniform(np.log10(0.1), np.log10(1.0), nthrows)
    R_ = np.power(10.0, R_)
    focusing_factor_ = np.random.normal(1.61, 0.24, nthrows)
    ice_prop = np.random.normal(6008.0, 100.0, nthrows)
    air_prop = np.random.normal(244.0, 1.0, nthrows)
    T_ratio = np.random.normal(1.00, 0.05, nthrows)

    #######################
    # Starting the toy MC #
    #######################

    att_freqs = np.array([])
    atts = np.array([])

    # linear fit values
    ms = np.zeros(nthrows)
    bs = np.zeros(nthrows)

    for i_throw in range(nthrows):
        att = calculate_att(T_ratio=T_ratio[i_throw],
                            R=R_[i_throw],
                            focusing_factor=focusing_factor_[i_throw],
                            Pxx_air=Pxx_air,
                            Pxx_ice=Pxx_ice,
                            air_prop=air_prop[i_throw],
                            ice_prop=ice_prop[i_throw])

        selection_region = np.logical_and(freqs > 140e6, freqs < 400e6)
        selection_region = np.logical_and(selection_region, att > 100.0)
        selection_region = np.logical_and(selection_region, att < 2000.0)

        freqs_to_fit = freqs[selection_region]
        att_to_fit = att[selection_region]

        try:
            m, b = np.polyfit(freqs_to_fit, att_to_fit, 1)
        except Exception as e:
            print("Error '{0}' occured. Arguments {1}.".format(e.message, e.args))
            print("Fit was most likely passed nan/inf")
            raise

        ms[i_throw] = m
        bs[i_throw] = b

        print(i_throw, ")", m * 1e6, "m / MHz", b, "m")

        atts = np.append(atts, att)
        att_freqs = np.append(att_freqs, freqs)

    return ms, bs, att_freqs, atts


def main():

    nthrows = 1000

    t_gb = (35.55e-6, 36.05e-6)  # Seconds
    t_noise = (22.0e-6, 34.0e-6)  # Seconds

    time_offset = (35.55e-6 - 34.59e-6)  # Seconds

    ms, bs, att_freqs, atts = perform_mc(t_gb=t_gb,
                                         t_noise=t_noise,
                                         nthrows=nthrows,
                                         time_offset=time_offset)

    new_freqs = np.linspace(150e6, 566.6666e6, 26)

    freqs = np.zeros(len(new_freqs) - 1)
    middle_val = np.zeros(len(new_freqs) - 1)
    low_bound = np.zeros(len(new_freqs) - 1)
    high_bound = np.zeros(len(new_freqs) - 1)

    for i_unique_freq, unique_freq in enumerate(new_freqs[:-1]):
        selection_region = np.logical_and(att_freqs > new_freqs[i_unique_freq],
                                          att_freqs < new_freqs[i_unique_freq + 1])
        atts_ = atts[selection_region]
        atts_ = np.sort(atts_)

        cumsum = np.cumsum(np.ones(len(atts_)))
        if(len(cumsum) == 0):
            continue
        cumsum = np.array(cumsum) / float(cumsum[-1])

        # Center of bin
        freqs[i_unique_freq] = (new_freqs[i_unique_freq] +
                                new_freqs[i_unique_freq + 1]) / 2.0

        # If lower bound of att is exactly equal (below threshold),
        # then report a 95\% CL instead of point with errorbars.

        if(atts_[np.argmin(np.abs(cumsum - 0.15))] == 0):
            cumsum_max = np.argmin(np.abs(cumsum - 0.95))

            middle_val[i_unique_freq] = atts_[cumsum_max]
            low_bound[i_unique_freq] = 0.0
            high_bound[i_unique_freq] = atts_[cumsum_max]
        else:
            cumsum_min = np.argmin(np.abs(cumsum - (0.5 - 0.341)))
            cumsum_max = np.argmin(np.abs(cumsum - (0.5 + 0.341)))
            cumsum_middle = np.argmin(np.abs(cumsum - 0.5))

            middle_val[i_unique_freq] = atts_[cumsum_middle]
            low_bound[i_unique_freq] = atts_[cumsum_min]
            high_bound[i_unique_freq] = atts_[cumsum_max]

    np.savez("./data_processed/A05_mc_results",
             ms=ms,
             bs=bs,
             freqs=freqs,
             low_bound=low_bound,
             high_bound=high_bound,
             middle_val=middle_val)


if __name__ == "__main__":

    main()
