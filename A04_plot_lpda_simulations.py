import numpy as np
import matplotlib.pyplot as plt
import copy


def dbm(trace):
    return 10.0 * np.log10(np.square(np.abs(trace) * 1e3) / 50.0)  # 1e3 for mV


Z0 = 120.0 * np.pi
ZL = 50.0
c = 0.3  # for ghz

##########################
# NuRadioMC's Simulation #
##########################

nuradiomc_sim = np.load("./data_simulated/createLPDA_100MHz_InfAir.npz")
numc_ff = nuradiomc_sim['ff']
numc_theta = nuradiomc_sim['theta']
numc_phi = nuradiomc_sim['phi']

numc_air = copy.deepcopy(numc_phi)

nuradiomc_sim = np.load("./data_simulated/createLPDA_100MHz_InfFirn_n1.4.npz")
numc_ff = nuradiomc_sim['ff']
numc_theta = nuradiomc_sim['theta']
numc_phi = nuradiomc_sim['phi']

numc_ice = copy.deepcopy(numc_phi)

n = 1.4

numc_air_gain = np.square(numc_air) * Z0 / ZL * 4.0 * np.pi * np.square(numc_ff / c)
numc_ice_gain = np.square(numc_ice) * Z0 / ZL * 4.0 * np.pi * n * np.square(numc_ff / c)

plt.figure()
plt.plot(numc_ff * 1e0, 10.0 * np.log10(numc_air_gain),
         color='purple', label="In Air")
plt.plot(numc_ff * 1e0, 10.0 * np.log10(numc_ice_gain),
         color='blue', label="In Ice")
plt.xlim(0.0, 1.0)
plt.ylim(-10.0, 10.0)
plt.xlabel("Freq. [GHz]")
plt.ylabel("Realized Gain [dB]")
plt.grid()
plt.legend()

numc_air *= np.sqrt(ZL / Z0)
numc_ice *= np.sqrt(ZL / Z0)
plt.figure()
plt.plot(numc_ff * 1e3, np.abs(numc_ice))
plt.plot(numc_ff * 1e3, np.abs(numc_air))
plt.xlim(0.0, 1e3)
plt.xlabel("Freq. [MHz]")
plt.ylabel("Effective Height [m]")
plt.grid()


# for the ratio, to calculate the systematic, we are doing the ratio of Voltages
# so the voltage is simply the square of the effective heights
ratio = np.square(np.abs(numc_ice)) / np.square(np.abs(numc_air))
ratio[np.logical_or(np.isnan(ratio), np.isinf(ratio))] = 0.0

plt.figure()
plt.plot(numc_ff * 1e3, ratio, color='red')
plt.xlim(0.0, 1e3)
plt.ylim(0.0, 1.2)
plt.xlabel("Freq. [MHz]")
plt.ylabel("Simulated In-Ice Voltage over In-Air Voltage")
plt.grid()

plt.savefig("./plots/A04_calc_systematic_gain.png",
            dpi=300)

plt.show()
