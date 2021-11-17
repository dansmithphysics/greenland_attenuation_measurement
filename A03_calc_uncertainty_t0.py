import glob
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt

noise_times = [25.0e-6, 32.0e-6]
window_length = 250

data = np.load("./data_processed/averaged_in_ice_trace.npz")
data_t = data["template_time"]
data_trace = data["template_trace"]

#####################
#      Raw Data     #
#####################

noise_std = np.std(data_trace[np.logical_and(data_t > noise_times[0], data_t < noise_times[1])])

plt.figure()
plt.title("Ice Echo Data from Bed Rock")

plt.plot(data_t * 1e6, 
         data_trace * 1e3,
         alpha = 1.0, color = 'black', linewidth = 1.0, label = "Data")

plt.axhline(3.0 * noise_std * 1e3, color = 'red', linestyle = '--', label = "3x Noise RMS")
plt.axvline(34.59, color = 'purple', linestyle = '-', label = "Start of G.B.")
plt.axvline(35.05, color = 'purple', linestyle = "-.", label = "End of G.B.")
plt.xlabel("Time Since Trigger $t_0$ [$\mu$s]")
plt.ylabel("Trace [mV]")
plt.xlim(33.0, 38.0)
plt.ylim(-0.7, 0.7)
plt.legend(loc = 'upper right')
plt.grid()

plt.savefig("./plots/A03_calc_uncertainty_t0.png", dpi = 300)

#####################
# Integrated Window #
#####################

data_power_mW = np.power(data_trace * 1e3, 2.0) / 50.0
time_length = window_length * (data_t[1] - data_t[0]) * 1e9
window = scipy.signal.windows.tukey(window_length, alpha = 0.25)

rolling = np.convolve(data_power_mW / time_length,
                      window,
                      'valid')

rolling_t = (data_t[(window_length - 1):] + data_t[:-(window_length - 1)]) / 2.0
    
noise = rolling[np.logical_and(rolling_t > noise_times[0], rolling_t < noise_times[1])]

noise = np.sort(noise)
cumsum = np.cumsum(np.arange(len(noise)))
cumsum = np.array(cumsum) / float(cumsum[-1])
noise_threshold = noise[np.argmin(np.abs(cumsum - 0.95))]

plt.figure()
plt.title("Ice Echo Data from Bed Rock, Integrated in Sliding %i ns Window" % int(250.0 * 1e9 * (data_t[1] - data_t[0])))
plt.semilogy((data_t[:len(rolling)] + data_t[-len(rolling):]) / 2.0* 1e6,
             rolling, alpha = 1.0, color = 'black', linewidth = 1.0, label = "Integrated Data")

plt.axhline(noise_threshold, color = 'red', linestyle = '--', label = "95% CI of Noise")
plt.axvline(34.59, color = 'purple', linestyle = '-', label = "Start of G.B.")
plt.axvline(35.05, color = 'purple', linestyle = "-.", label = "End of G.B.")
plt.xlabel("Time Since Trigger $t_0$ [$\mu$s]")
plt.ylabel("Integrated Power [mW / ns]")
plt.xlim(33.0, 38.0)
plt.ylim(4e-6, 4e-2)
plt.legend(loc = 'upper right')
plt.grid()

plt.savefig("./plots/A03_calc_uncertainty_t0_integrated.png", dpi = 300)

plt.show()
