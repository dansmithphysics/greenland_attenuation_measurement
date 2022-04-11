import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate

avva_et_al_freqs = [75.0, 300.0]
avva_et_al_results = [1149.0, 1022.0]
avva_et_al_ups = [112.0, 230.0]
avva_et_al_dws = [103.0, 253.0]

xs = [820.3853325046614, 16.159105034182716]
ys = [766.7496886674967, 1209.090909090909]

f = scipy.interpolate.interp1d(xs,
                               ys,
                               kind = 'linear',
                               bounds_error = False,
                               fill_value = 'extrapolate')

#p0: [60.07896249685749, 56.5898269295221, -0.95, -679.4616970734627, 1106.421509207586]
# all
# Draft
b = 1024
b_spread = 50.0

m = -0.65
m_spread = 0.06

freqs = np.linspace(0.0, 1000.0, 1000)

att_avg_1500 = m * freqs + b
nu_mc = f(freqs)

plt.figure()
plt.plot(freqs,
         att_avg_1500,
         color = 'black',
         label = r"Bogorodsky Model, $\pm1\sigma$ Errors")

plt.fill_between(freqs,
                 m * freqs + (b + b_spread),
                 m * freqs + (b - b_spread),
                 color = 'black',
                 alpha = 0.5)

# Updated paper

b = 1154
b_spread = 121

m = -0.81
m_spread = 0.14

freqs = np.linspace(0.0, 1000.0, 1000)

att_avg_1500 = m * freqs + b
nu_mc = f(freqs)

plt.plot(freqs,
         att_avg_1500,
         color = 'red',
         label = r"MacGregor Model, $\pm1\sigma$ Errors")


plt.fill_between(freqs,
                 m * freqs + (b + b_spread),
                 m * freqs + (b - b_spread),
                 color = 'red',
                 alpha = 0.5)

plt.axvline(145.0, color = 'red', linestyle = '--', label = "Bandpass Filters of Our Result")
plt.axvline(575.0, color = 'red', linestyle = '--')

plt.plot(freqs,
         nu_mc,
         color = 'green',
         label = "NuRadioMC at exactly 1500 m")

plt.scatter(avva_et_al_freqs,
            avva_et_al_results,
            color = 'purple',
            label = "Avva et al. Result")
plt.errorbar(avva_et_al_freqs,
             avva_et_al_results,
             yerr = ((avva_et_al_ups), (avva_et_al_dws)),
             color = 'purple', ls = 'none')

plt.xlabel("Frequency [MHz]")
plt.ylabel("Average Attenuation Length of Top 1500 m [m]")
plt.grid()
plt.legend(loc = 'lower left')
plt.xlim(00.0, 600.0)
plt.ylim(500.0, 1300.0)
#plt.ylim(600.0, 1475.0)
plt.savefig("./plots/A11_nuradiomc_Avva_DanSmith_att_comparison.png", dpi = 300)
plt.show()
