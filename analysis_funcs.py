import numpy as np
import scipy.signal


def load_file(file_name, att_correction=0, time_offset=0, return_fft=False):
    """
    Load the pickle file, and adjust the time 
    and amplitude to adjust detector effects and 
    correct for attenuators. 
    """
    
    try:
        ice = np.load(file_name)
    except FileNotFoundError:
        print("File not found: %s" % file_name)
        raise

    t = ice["template_time"]
    trace = ice["template_trace"]

    t += time_offset
    trace *= np.power(10.0, att_correction / 20.0)

    if(return_fft):
        fs = 1.0 / (t[1] - t[0])
        fft = np.fft.rfft(trace)
        freq = np.fft.rfftfreq(len(trace), 1.0 / fs)
        return t, trace, freq, fft, fs
    else:
        return t, trace


def calculate_uncertainty(entries):
    """
    Calculate the approximate CDF    
    """

    entries = np.sort(entries)
    cumsum = np.cumsum(np.ones(len(entries)))

    if(len(cumsum) != 0):
        cumsum = cumsum / float(cumsum[-1])

    return entries, cumsum


def power_integration(time, trace, window_length=250):
    """
    Sliding window integration to 
    suppress noise and highlight signal. 
    """
    
    data_power_mW = np.power(trace, 2.0) / 50.0 * 1e3
    delta_t = (time[1] - time[0]) * 1e9
    time_length = window_length * delta_t

    window = scipy.signal.windows.tukey(window_length, alpha=0.25)

    rolling = np.convolve(data_power_mW * delta_t,
                          window,
                          'valid')

    time = (time[(window_length - 1):] + time[:-(window_length - 1)]) / 2.0

    return time, rolling


def return_confidence_intervals(entries, cdf):
    """
    Find the 68.2% CI given a CDF.
    """

    entries_min = entries[np.argmin(np.abs(cdf - (0.5 - 0.341)))]
    entries_mid = entries[np.argmin(np.abs(cdf - (0.5 - 0.000)))]
    entries_max = entries[np.argmin(np.abs(cdf - (0.5 + 0.341)))]

    return entries_min, entries_mid, entries_max
