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
        atts[i_throw] = att

    return ms, bs, freqs, atts


def main():

    nthrows = 1000

    exper_constants = experiment.Experiment()

    ms, bs, att_freqs, atts = perform_mc(exper_constants=exper_constants,
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

        atts_, cumsum = analysis_funcs.calculate_uncertainty(atts_)
        
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
            entries_min, entries_mid, entries_max = analysis_funcs.return_confidence_intervals(atts_, cumsum)
            
            middle_val[i_unique_freq] = entries_mid
            low_bound[i_unique_freq] = entries_min
            high_bound[i_unique_freq] = entries_max

    np.savez("./data_processed/A05_mc_results",
             ms=ms,
             bs=bs,
             freqs=freqs,
             low_bound=low_bound,
             high_bound=high_bound,
             middle_val=middle_val)


if __name__ == "__main__":

    main()
