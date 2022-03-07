import numpy as np


def load_file(file_name, att_correction=0, time_offset=0, return_fft=False):

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
    entries = np.sort(entries)

    cumsum = np.cumsum(np.ones(len(entries)))

    if(len(cumsum) != 0):
        cumsum = cumsum / float(cumsum[-1])

    return entries, cumsum


def return_confidence_intervals(entries, cumsum):

    entries_min = entries[np.argmin(np.abs(cumsum - (0.5 - 0.341)))]
    entries_mid = entries[np.argmin(np.abs(cumsum - (0.5 - 0.000)))]
    entries_max = entries[np.argmin(np.abs(cumsum - (0.5 + 0.341)))]

    return entries_min, entries_mid, entries_max
