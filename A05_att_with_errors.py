import numpy as np
import matplotlib.pyplot as plt
import analysis_funcs
import experiment


def perform_mc(exper_constants, exper, nthrows=1000, verbose=False):

    atts = np.zeros((nthrows, len(exper.Pxx_air)))

    # linear fit values
    ms = np.zeros(nthrows)
    bs = np.zeros(nthrows)

    for i_throw in range(nthrows):
        exper.mc_set_ice_properties()
        att = exper.calculate_att()
        freqs = exper.gb_freq
        
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

        if(verbose):
            print("%i) \t %.2E m/MHz \t %.2E m" % (i_throw, m * 1e6, b))

        atts[i_throw] = att

    return ms, bs, freqs, atts


def main(exper_constants, exper, nthrows=10000, save_name=None, verbose=False):
    
    ms, bs, att_freqs, atts = perform_mc(exper_constants=exper_constants,
                                         exper=exper,
                                         nthrows=nthrows,
                                         verbose=verbose)

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

    if(save_name is not None):
        np.savez(save_name,
                 ms=ms,
                 bs=bs,
                 freqs=freqs,
                 low_bound=low_bound,
                 high_bound=high_bound,
                 middle_val=middle_val)

    return freqs, low_bound, high_bound, middle_val

if __name__ == "__main__":
    
    ice_file_name = "./data_processed/averaged_in_ice_trace.npz"
    air_file_name = "./data_processed/averaged_in_air_trace.npz"
    
    exper_constants = experiment.ExperimentConstants()
    exper = experiment.Experiment(exper_constants, ice_file_name, air_file_name, verbose=True)

    main(exper_constants, exper, save_name="./data_processed/A05_mc_results")
