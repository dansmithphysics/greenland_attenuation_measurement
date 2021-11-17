import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

t0_from_data = 0.035e-6
t_tx = 74.2e-9
t_rx = 74.5e-9
prop_time_delay = 8.0 / 3e8 # measured by eye

window_length = 250

file_names = glob.glob("./data_processed/averaged_in_ice_trace_biref.npz")

data = np.load(file_names[0])
data_t = data['template_time']
data_trace = data['template_trace']

# time corrections
data_t += prop_time_delay
data_t -= t0_from_data
data_t -= t_rx

data_power_mW = np.power(data_trace * 1e3, 2.0) / 50.0
time_length = window_length * (data_t[1] - data_t[0]) * 1e9

window = scipy.signal.windows.tukey(window_length, alpha = 0.25)

rolling = np.convolve(data_power_mW / time_length,
                      window,
                      'valid')
    
data_t = (data_t[(window_length - 1):] + data_t[:-(window_length - 1)]) / 2.0

noise = rolling[np.logical_and(data_t > 25.0e-6, data_t < 32.0e-6)]

entries = np.sort(copy.deepcopy(np.abs(noise)))
cumsum = np.cumsum(np.arange(len(entries)))
cumsum = np.array(cumsum) / float(cumsum[-1])
threshold = entries[np.argmin(np.abs(cumsum - 0.95))]

plt.figure()
plt.semilogy(data_t * 1e6, rolling, color = 'black', linewidth = 1.0)
plt.axvline(35.57, color = 'purple', alpha = 0.85, label = "Arrival of Bedrock Echo")
plt.axhline(threshold, color = 'red', linestyle = "--", label = "95% CI of Noise")

plt.grid()
plt.title("Ice Echo Data From Bed Rock, Integrated in Sliding 100 ns Window \n From Birefringence Data Run")
plt.xlabel(r"Absolute Time Since Pulser [$\mu$s]")
plt.ylabel("Integrated Power [mW / ns]")
plt.xlim(33.0, 38.0)
plt.ylim(1e-6, 1e-3)

x_ticks = np.arange(33, 39, 1)
x_ticks = np.append(x_ticks, 35.57)
x_ticks = np.sort(x_ticks)
x_tick_labels = [str(int(tick)) if int(tick) == tick else str(tick) for tick in x_ticks]
plt.xticks(x_ticks, x_tick_labels)
plt.legend(loc = 'upper left')

plt.savefig("./plots/A03_calc_uncertainty_depth_time_integrate.png", dpi = 300)

#######################################################################
#                                                                     #
# First half of the script is to find the ground bounce time          #
#                                                                     #
# Second half is to calculate the uncertainty from ice model of depth #
#                                                                     #
#######################################################################

def rho(z):
    if(z <= 14.9):
        return 0.917 - 0.594 * np.exp(-1.0 * z / 30.8)
    else:
        return 0.917 - 0.367 * np.exp(-1.0 * (z - 14.9) / 40.5)
def n(z):
    return np.abs(1.0 + 0.845 * rho(z))

# get average index of refraction
n_avg_top_1k, steps_top_1k = 0, 0
z, t = 0, 0
space_step = 0.1 # m
for i in range(1000000):
    z += space_step
    t += space_step / (0.3 / n(z))
    
    if(z < 200.0):
        n_avg_top_1k += n(z)
        steps_top_1k += 1
    else:
        break

n_avg_top_1k = float(n_avg_top_1k / float(steps_top_1k))

def calculate_n(z):
    return (n_avg_top_1k * (1.78 / 1.774865) * 200.0 + 1.78 * (z - 200.0)) / z

nthrows = 20000
entries = []

scales = np.random.normal(1.78, 0.03, nthrows) / 1.78
gb_times = 35.57e-6 * np.ones(len(scales))
    
depths = np.linspace(2600.0, 3400.0, 1000)
for throw in range(nthrows):
    times = np.array([2.0 * depth / (3.0e8 / (calculate_n(depth) * scales[throw])) for depth in depths])
    f = scipy.interpolate.interp1d(times, depths, kind = 'linear')
    entries += [f(gb_times[throw])]
    
entries = np.sort(entries)
cumsum = np.cumsum(np.arange(len(entries)))
cumsum = np.array(cumsum) / float(cumsum[-1])
    
f_cumsum = scipy.interpolate.interp1d(entries, cumsum, kind = 'cubic')
even_entries = np.linspace(np.min(entries), np.max(entries), 100)
even_cumsum = f_cumsum(even_entries)
even_cumsum_grad = np.gradient(even_cumsum)

entries_min = entries[np.argmin(np.abs(cumsum - (0.5 - 0.341)))]
entries_mid = entries[np.argmin(np.abs(cumsum - (0.5 - 0.000)))]
entries_max = entries[np.argmin(np.abs(cumsum - (0.5 + 0.341)))]

print("%f (+%f)(-%f)" % (entries_mid, entries_mid - entries_min, entries_max - entries_mid))

plt.figure()
plt.plot(entries, cumsum)
plt.axvline(entries_min, color = 'red', linestyle = '--')
plt.axvline(entries_mid, color = 'red', linestyle = '--', linewidth = 2.0, label = r"Depth: $3035^{+39}_{-45}$ m")
plt.axvline(entries_max, color = 'red', linestyle = '--')
plt.title("Depth from Time of Flight \n through Ice of $n_{asymp.} = 1.78 \pm 0.03$")
plt.legend(loc = 'upper left')
plt.grid()
plt.xlim(3050 - 300, 3050 + 300)
plt.ylim(0.0, 1.0)
plt.xlabel("Depth [m]")
plt.ylabel("CDF")

plt.savefig("./plots/A04_calc_uncertainty_plots_time_to_depth.png", dpi = 300)

plt.show()
