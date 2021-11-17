import numpy as np
import matplotlib.pyplot as plt
import glob

air_file_names = glob.glob("./data_raw/LPDA_Summit_Measurements/*AIR*LPDA*S11*csv")
ice_file_names = glob.glob("./data_raw/LPDA_Summit_Measurements/*ICE*LPDA*S11*csv")

ice_file_names = np.array([file_name for file_name in ice_file_names if not("ORIG" in file_name or "CROSS" in file_name)])

plt.figure()
for file_name in np.append(air_file_names, ice_file_names):
    print(file_name)
    freq, s11 = np.loadtxt(file_name, delimiter=",", comments = ['!', 'BEGIN', 'END'], unpack = True)

    if("AIR" in file_name):
        color = 'red'
    else:
        color = 'blue'

    plt.plot(freq * 1e-6, np.power(10.0, s11 / 10.0), color = color, alpha = 0.5)        

plt.plot([], [], color = 'red', alpha = 0.5, label = "In Air")    
plt.plot([], [], color = 'blue', alpha = 0.5, label = "In Ice")
    
plt.xlim(0.0, 1000.0)
plt.ylim(0.0, 0.5)
plt.xlabel("Freq. [MHz]")
plt.ylabel("S11 [dB]")
plt.legend()
plt.grid()
plt.savefig("./plots/A03_calc_uncertainty_match.png", dpi = 300)

plt.figure()

maxs = np.zeros(len(s11))
mins = np.ones(len(s11)) * 1000.0

for ice_file_name in ice_file_names:
    for air_file_name in air_file_names:

        freq, ice_s11 = np.loadtxt(ice_file_name, delimiter=",", comments = ['!', 'BEGIN', 'END'], unpack = True)
        freq, air_s11 = np.loadtxt(air_file_name, delimiter=",", comments = ['!', 'BEGIN', 'END'], unpack = True)

        ratio = ((1.0 - np.power(10.0, ice_s11 / 20.0)) / (1.0 - np.power(10.0, air_s11 / 20.0)))

        for i in range(len(ratio)):
            if(ratio[i] > maxs[i]):
                maxs[i] = ratio[i]
            if(ratio[i] < mins[i]):
                mins[i] = ratio[i]
            
plt.fill_between(freq * 1e-6, maxs, mins, color = 'black', alpha = 0.1)

mean = 1.05
up = 0.05
down = -0.05

plt.axhline(mean + up, color = 'black', linestyle = "--", label = r"$T_{ratio} = 1.05\pm0.05$")
plt.axhline(mean + down, color = 'black', linestyle = "--")
plt.axvline(145.0, color = 'red', linestyle = '--', label = "Bandpass Filter")
plt.axvline(575.0, color = 'red', linestyle = '--')
plt.ylim(0.7, 1.3)
plt.xlim(0.0, 750.0)
plt.xlabel("Freq. [MHz]")
plt.ylabel("T$_{ratio}$, (1.0 - S11$_{ice}$) / (1.0 - S11$_{air}$)")
plt.legend()
plt.grid()
plt.savefig("./plots/A03_calc_unceratinty_match_t_ratio_with_errors.png", dpi = 300)

entries = (maxs + mins)[np.logical_and(freq > 150.0e6, freq < 550.0e6)] / 2.0
entries = np.sort(entries)
cumsum = np.cumsum(np.arange(len(entries)))
cumsum = np.array(cumsum) / float(cumsum[-1])

entries_min = entries[np.argmin(np.abs(cumsum - (0.5 - 0.341)))]
entries_mid = entries[np.argmin(np.abs(cumsum - (0.5 - 0.000)))]
entries_max = entries[np.argmin(np.abs(cumsum - (0.5 + 0.341)))]

print("%f (+%f)(-%f)" % (entries_mid, entries_mid - entries_min, entries_max - entries_mid))

plt.figure()
plt.plot(entries, cumsum)
plt.axvline(entries_min, color = 'red', linestyle = '--')
plt.axvline(entries_mid, color = 'red', linestyle = '--', linewidth = 2.0, label = r"Depth: $3045^{+41}_{-45}$ m")
plt.axvline(entries_max, color = 'red', linestyle = '--')

plt.show()
