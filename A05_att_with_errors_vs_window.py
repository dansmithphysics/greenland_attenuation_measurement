import copy
import glob
import scipy.signal
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
import analysis_funcs
import experiment


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


def perform_mc(exper_constants, nthrows=1000):

    # Load up the data
    ice_time, ice_trace, ice_freq, ice_fft, ice_fs = analysis_funcs.load_file("./data_processed/averaged_in_ice_trace.npz",
                                                                              att_correction=exper_constants.ice_att,
                                                                              time_offset=exper_constants.time_offset,
                                                                              return_fft=True)
    air_time, air_trace, air_freq, air_fft, air_fs = analysis_funcs.load_file("./data_processed/averaged_in_air_trace.npz",
                                                                              att_correction=exper_constants.air_att,
                                                                              time_offset=exper_constants.time_offset,
                                                                              return_fft=True)

    # Define a master time close,
    # based on the ice_data time steps.
    master_time = np.arange(exper_constants.gb_duration * ice_fs) / ice_fs
    master_freq = np.fft.rfftfreq(len(master_time), 1.0 / ice_fs)

    air_f = scipy.interpolate.interp1d(air_time, air_trace,
                                       kind='linear', bounds_error=False, fill_value=0.0)

    air_trace = np.array(air_f(master_time + air_time[0]))
    air_time = copy.deepcopy(master_time + air_time[0])

    air_fft = np.fft.rfft(air_trace)
    air_fs = ice_fs

    Pxx_air = np.abs(np.square(np.fft.rfft(air_trace)))
    freqs = copy.deepcopy(master_freq)

    noise = ice_trace[np.logical_and(ice_time > exper_constants.noise_start,
                                     ice_time < exper_constants.noise_end)]

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
    Pxx_noise /= (exper_constants.noise_end - exper_constants.noise_start)

    Pxx_ice = prepare_ice_signal((exper_constants.gb_start, exper_constants.gb_end),
                                 ice_time,
                                 ice_trace,
                                 master_time,
                                 Pxx_noise)

    #######################
    # Starting the toy MC #
    #######################

    R_ = np.random.uniform(np.log10(exper_constants.m_R_low),
                           np.log10(exper_constants.m_R_high),
                           nthrows)
    R_ = np.power(10.0, R_)
    focusing_factor_ = np.random.normal(exper_constants.m_ff,
                                        exper_constants.m_ff_uncert,
                                        nthrows)
    ice_prop = np.random.normal(exper_constants.m_depth,
                                exper_constants.m_depth_uncert,
                                nthrows)
    air_prop = np.random.normal(exper_constants.m_air_prop,
                                exper_constants.m_air_prop_uncert,
                                nthrows)
    T_ratio = np.random.normal(exper_constants.m_T_ratio,
                               exper_constants.m_T_ratio_uncert,
                               nthrows)

    atts = np.zeros((nthrows, len(Pxx_air)))

    for i_throw in range(nthrows):
        att = calculate_att(T_ratio=T_ratio[i_throw],
                            R=R_[i_throw],
                            focusing_factor=focusing_factor_[i_throw],
                            Pxx_air=Pxx_air,
                            Pxx_ice=Pxx_ice,
                            air_prop=air_prop[i_throw],
                            ice_prop=ice_prop[i_throw])

        atts[i_throw] = att

    return freqs, atts


def main(nthrows=1000, gb_end=36.05e-6):

    exper_constants = experiment.Experiment()
    exper_constants.gb_end = gb_end

    att_freqs, atts = perform_mc(exper_constants,
                                 nthrows=nthrows)

    new_freqs = np.linspace(150e6, 566.6666e6, 26)

    freqs = np.zeros(len(new_freqs) - 1)
    middle_val = np.zeros(len(new_freqs) - 1)
    low_bound = np.zeros(len(new_freqs) - 1)
    high_bound = np.zeros(len(new_freqs) - 1)

    for i_unique_freq, unique_freq in enumerate(new_freqs[:-1]):
        selection_region = np.logical_and(att_freqs > new_freqs[i_unique_freq],
                                          att_freqs < new_freqs[i_unique_freq + 1])
        atts_ = atts[:, selection_region].flatten()
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

    return freqs, low_bound, high_bound, middle_val


if __name__ == "__main__":

    start_time = 35.55e-6

    end_times = np.linspace(start_time + 0.1e-6,
                            start_time + 4.0e-6,
                            5)
    end_times = np.append(end_times, [36.05e-6])
    end_times = np.sort(end_times)

    vals = np.zeros(len(end_times))
    vals_min = np.zeros(len(end_times))
    vals_max = np.zeros(len(end_times))

    freq_select = 3  # 200 MHz

    for i_end_time, end_time in enumerate(end_times):
        print(start_time, end_time, end_time-start_time)

        freqs, low_bound, high_bound, middle_val = main(nthrows=1000, gb_end=end_time)

        vals[i_end_time] = middle_val[freq_select]
        vals_min[i_end_time] = low_bound[freq_select]
        vals_max[i_end_time] = high_bound[freq_select]

    plt.figure(figsize=(5, 4))

    plt.plot((end_times[vals_min != 0] - start_time) / 1e-6,
             vals[vals_min != 0],
             color='red')
    '''
    plt.fill_between((end_times[vals_min != 0] - start_time) / 1e-6,
                     vals_min[vals_min != 0],
                     vals_max[vals_min != 0],
                     color='red',
                     alpha=0.5,
                     label="Modified Time Window Result, 1 $\sigma$ Errors")
    '''
    plt.fill_between((end_times - start_time) / 1e-6,
                     vals_min,
                     vals_max,
                     color='red',
                     alpha=0.5,
                     label="Modified Time Window Result, 1 $\sigma$ Errors")

    plt.scatter(36.1 - start_time / 1e-6,
                711.0,
                color='black')

    plt.errorbar(36.1 - start_time / 1e-6,
                 711.0,
                 yerr=([711.0 - 653.0], [772.0 - 711.0]),
                 label="Nominal Result, 1 $\sigma$ Errors",
                 color='black',
                 ls='none')

    plt.grid()
    plt.xlim(0, 4.0)
    plt.ylim(500.0, 1100.0)

    yticks = np.arange(500, 1101, 100)
    yticks = np.append(yticks, [653.0, 711.0, 772.0])
    yticks = np.sort(yticks)
    yticklabels = [str(int(i)) if i % 100 == 0 else '' for i in yticks]
    plt.yticks(yticks, labels=yticklabels)

    plt.xlabel("Window Length Used in Analysis [$\mu$s]")
    plt.ylabel("Bulk Field Attenuation Length at 200 MHz [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/A05_att_increased_window.png",
                dpi=300)
    plt.show()
