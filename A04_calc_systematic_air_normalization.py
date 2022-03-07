import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.interpolate
import scipy.signal
import copy
import experiment


def dbm(trace):
    return 10.0 * np.log10(np.square(np.abs(trace)) / 50.0 * 1e3)  # 1e3 for mW


exper_constants = experiment.Experiment()

exper_constants.Z0 = 120.0 * np.pi
exper_constants.ZL = 50.0
c = exper_constants.c / 1e9

####################
# xFDTD Simulation #
####################

# gain file
gain_file = "./data_simulated/lpda_xfdtd_sims/gain.csv"
vswr_file = "./data_simulated/lpda_xfdtd_sims/vswr.csv"

# GHz
gain_freqs, gain_theta_, gain_phi_, gain_phi, gain_theta = np.loadtxt(gain_file, delimiter=",", unpack=True, skiprows=1)
vswr_freqs, vswr = np.loadtxt(vswr_file, delimiter=",", unpack=True, skiprows=1)
lpda_s11 = (vswr - 1) / (vswr + 1)

lpda_gain_rl = gain_theta * (1.0 - np.square(np.abs(lpda_s11)))
lpda_freqs = copy.deepcopy(gain_freqs)

# my simuation results
mine_h_rl = np.abs(np.sqrt(np.square(c / lpda_freqs) * (exper_constants.ZL / exper_constants.Z0) * (1.0 / (4.0 * np.pi)) * lpda_gain_rl))
mine_h_rl *= np.sqrt(exper_constants.ZL / exper_constants.Z0)

# going to upsample here a lot
mine_h_rl = np.append(mine_h_rl, np.zeros(200000 - len(mine_h_rl)))

master_freqs = np.arange(len(mine_h_rl)) * (lpda_freqs[1] - lpda_freqs[0])
master_times = np.arange(len(np.fft.irfft(mine_h_rl))) / master_freqs[-1] / 2.0

##########################
# NuRadioMC's Simulation #
##########################

nuradiomc_sim = np.load("./data_simulated/createLPDA_100MHz_InfAir.npz")
numc_ff = nuradiomc_sim['ff']
numc_theta = nuradiomc_sim['theta']
numc_phi = nuradiomc_sim['phi']

f_numc = scipy.interpolate.interp1d(numc_ff, numc_phi,
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=0.0)
numc_h = f_numc(master_freqs)
numc_h *= np.sqrt(exper_constants.ZL / exper_constants.Z0)

##############
# FID pulser #
##############

file_name = "./data_raw/Groundbounce_fid_pulser_after_sick_72db/2021_08_08_fid_pulser_with_72db_att_Ch1.csv"
data_fid_time, data_fid_trace = np.loadtxt(file_name, delimiter=",", skiprows=6, usecols=(3, 4), unpack=True)
data_fid_trace *= np.power(10.0, 72.0 / 20.0)
data_fid_time *= 1.0e9

fid_sickness = 2.0  # because the FID trace I am using is low amplitude ...

f_fid = scipy.interpolate.interp1d(data_fid_time - data_fid_time[0], data_fid_trace * fid_sickness,
                                   kind='linear', bounds_error=False, fill_value=0.0)

data_fid_trace = f_fid(master_times)
fid_fft = np.fft.rfft(data_fid_trace)

################
# Frontend Amp #
################

amp_freqs, amp_s21 = np.loadtxt("./data_raw/amp_board_v3.txt",
                                delimiter=",",
                                unpack=True,
                                comments="#")
amp_freqs *= 1e-3  # GHz
amp_s21 = np.power(10.0, amp_s21 / 20.0)
f_amp_s21 = scipy.interpolate.interp1d(amp_freqs, amp_s21,
                                       kind='linear', bounds_error=False, fill_value=0.0)
amp_s21 = (f_amp_s21(master_freqs))

###########
# Filters #
###########

Hz, S_1, S_2, dB_1, dB_2, R_1, R_2, fifty_1, fifty_2 = np.loadtxt("./data_simulated/s_para_files/VHF-145+.S2P",
                                                                  delimiter="\t",
                                                                  unpack=True,
                                                                  comments=["!", "#"],
                                                                  skiprows=1)

hp_filter_freq = Hz * 1e-9
hp_filter = np.power(10.0, dB_1 / 20.0)
f_hp_filter = scipy.interpolate.interp1d(hp_filter_freq, hp_filter,
                                         kind='linear',
                                         bounds_error=False,
                                         fill_value=0.0)
hp_filter = (f_hp_filter(master_freqs))

Hz, S_1, S_2, dB_1, dB_2, R_1, R_2, fifty_1, fifty_2 = np.loadtxt("./data_simulated/s_para_files/VLF-575+___Plus25degC.S2P",
                                                                  delimiter=",",
                                                                  unpack=True,
                                                                  comments=["!", "#"],
                                                                  skiprows=1)

lp_filter_freq = Hz * 1e-3
lp_filter = np.power(10.0, dB_1 / 20.0)
f_lp_filter = scipy.interpolate.interp1d(lp_filter_freq, lp_filter,
                                         kind='linear', bounds_error=False, fill_value=0.0)
lp_filter = (f_lp_filter(master_freqs))

#####################
# Signal Processing #
#####################

r = 244.0
w = 2.0 * np.pi * master_freqs

mine_signal = copy.deepcopy(fid_fft)  # FID Voltage
mine_signal *= mine_h_rl  # tx antenna response
mine_signal *= w  # derivative from transmission
mine_signal /= (2.0 * np.pi * c * r)  # path loss
mine_signal *= (exper_constants.Z0 / exper_constants.ZL)
mine_signal *= mine_h_rl  # rx antenna response
mine_signal *= amp_s21  # rx amplifier
mine_signal *= lp_filter  # LP filter
mine_signal *= hp_filter  # hp filter
mine_signal[np.logical_or(np.isnan(mine_signal), np.isinf(mine_signal))] = 0.0

numc_signal = copy.deepcopy(fid_fft)  # FID Voltage
numc_signal *= numc_h  # tx antenna response
numc_signal *= w  # derivative from transmission
numc_signal /= (2.0 * np.pi * c * r)  # path loss
numc_signal *= (exper_constants.Z0 / exper_constants.ZL)
numc_signal *= numc_h  # rx antenna response
numc_signal *= amp_s21  # rx amplifier
numc_signal *= lp_filter  # LP filter
numc_signal *= hp_filter  # hp filter
numc_signal[np.logical_or(np.isnan(numc_signal), np.isinf(numc_signal))] = 0.0

################
# Load up data #
################

air_to_air_att = 46.0  # dBm
data_time, data_trace = np.loadtxt("./data_raw/2021_08_02_psuedo_air_to_air_with_46db_att_Ch1.csv",
                                   delimiter=",",
                                   skiprows=6,
                                   usecols=(3, 4),
                                   unpack=True)
data_trace *= np.power(10.0, air_to_air_att / 20.0)
data_time *= 1e9
f_air = scipy.interpolate.interp1d(data_time - data_time[0], data_trace,
                                   kind='linear',
                                   bounds_error=False,
                                   fill_value=0.0)
data_air_trace = f_air(master_times)
airtoair_signal = np.abs(np.fft.rfft(data_air_trace))

############
# Plotting #
############

plt.figure(figsize=(5, 4))

f, Pxx_airtoair = scipy.signal.periodogram(data_air_trace,
                                           1.0 / (master_times[1] - master_times[0]) * 1e9,
                                           scaling='density',
                                           return_onesided=True)
Pxx_airtoair /= 50.0  # power on 50 Ohm
Pxx_airtoair *= 1e3  # mW
plt.plot(f * 1e-6, 10.0 * np.log10(Pxx_airtoair), color='black', label="Data")

f, Pxx_mine = scipy.signal.periodogram(np.fft.irfft(mine_signal),
                                       1.0 / (master_times[1] - master_times[0]) * 1e9,
                                       scaling='density',
                                       return_onesided=True)

Pxx_mine /= 50.0  # power on 50 Ohm
Pxx_mine *= 1e3  # mW
plt.plot(f * 1e-6, 10.0 * np.log10(Pxx_mine),
         color='red',
         label="xFDTD Simulation")

f, Pxx_numc = scipy.signal.periodogram(np.fft.irfft(numc_signal),
                                       1.0 / (master_times[1] - master_times[0]) * 1e9,
                                       scaling='density',
                                       return_onesided=True)

Pxx_numc /= 50.0  # power on 50 Ohm
Pxx_numc *= 1e3  # mW
plt.plot(f * 1e-6, 10.0 * np.log10(Pxx_numc), color='purple', label="WIPL-D Simulation")

plt.xlim(0.0, 750.0)
plt.ylim(-85.0, -45.0)
plt.xlabel("Frequency [MHz]")
plt.ylabel("Power at Oscilloscope [dBm / Hz]")
plt.grid()
plt.legend()
plt.savefig("./plots/A04_calc_systematics_air_normalization_no_reflections.png",
            dpi=300)

plt.show()
exit()

##############
# Reflection #
##############

plt.figure()

numc_signal = copy.deepcopy(fid_fft)  # FID Voltage
numc_signal *= numc_h  # tx antenna response
numc_signal *= w  # derivative from transmission
numc_signal *= 1.0j
numc_signal /= (2.0 * np.pi * c * r)  # path loss
numc_signal *= (exper_constants.Z0 / exper_constants.ZL)
numc_signal[0] = 0.0
numc_signal_time = np.fft.fftshift(np.fft.irfft(numc_signal))
f_numc = scipy.interpolate.interp1d(master_times, numc_signal_time,
                                    kind='linear', bounds_error=False, fill_value=0.0)

mine_signal = copy.deepcopy(fid_fft)  # FID Voltage
mine_signal *= mine_h_rl  # tx antenna response
mine_signal *= w  # derivative from transmission
numc_signal *= 1.0j
mine_signal /= (2.0 * np.pi * c * r)  # path loss
mine_signal *= (exper_constants.Z0 / exper_constants.ZL)
mine_signal[0] = 0.0
mine_signal_time = np.fft.fftshift(np.fft.irfft(mine_signal))
f_mine = scipy.interpolate.interp1d(master_times, mine_signal_time,
                                    kind='linear', bounds_error=False, fill_value=0.0)

scale = -1.0

for shift_time in [0.4]:

    numc_signal_time_ = copy.deepcopy(numc_signal_time) + scale * f_numc(master_times - shift_time)
    numc_signal_time_ = np.fft.fftshift(numc_signal_time_)
    numc_signal_ = np.fft.rfft(numc_signal_time_)

    numc_signal_ *= numc_h  # rx antenna response
    numc_signal_ *= amp_s21 * lp_filter * hp_filter  # rx response
    numc_signal_[np.logical_or(np.isnan(numc_signal_), np.isinf(numc_signal_))] = 0.0

    plt.plot(master_freqs * 1e3, dbm(numc_signal_), color='purple', label="NuRadioMC Sim")

    mine_signal_time_ = copy.deepcopy(mine_signal_time) + scale * f_mine(master_times - shift_time)
    mine_signal_time_ = np.fft.fftshift(mine_signal_time_)
    mine_signal_ = np.fft.rfft(mine_signal_time_)

    mine_signal_ *= mine_h_rl  # rx antenna response
    mine_signal_ *= amp_s21 * lp_filter * hp_filter  # rx response
    mine_signal_[np.logical_or(np.isnan(mine_signal_), np.isinf(mine_signal_))] = 0.0

    plt.plot(master_freqs * 1e3, dbm(mine_signal_), color='red', label="xFDTD Sim")

numc_signal *= numc_h  # rx antenna response
numc_signal *= amp_s21 * lp_filter * hp_filter  # rx response
numc_signal[np.logical_or(np.isnan(numc_signal), np.isinf(numc_signal))] = 0.0

mine_signal *= mine_h_rl  # rx antenna response
mine_signal *= amp_s21 * lp_filter * hp_filter  # rx response
mine_signal[np.logical_or(np.isnan(mine_signal), np.isinf(mine_signal))] = 0.0

plt.plot(master_freqs * 1e3, dbm(airtoair_signal), color="black", label="Data")

plt.title("Data vs. Simulation for LPDA Antennas seperated by 244 m \n With Reflection at 0.4 ns")
plt.xlim(0.0, 750.0)
plt.ylim(110.0, 160.0)
plt.xlabel("Freqs. [MHz]")
plt.ylabel("Power at Scope [dBm]")
plt.grid()
plt.legend()

plt.savefig("./plots/A04_calc_systematics_air_normalization_reflections.png",
            dpi=300)

plt.figure()
plt.plot(master_freqs * 1e3, np.power(10.0, (dbm(mine_signal_) - dbm(mine_signal)) / 20.0), color='red', label="xFDTD Sim")
plt.xlim(0.0, 750.0)
plt.ylim(0.0, 1.5)
plt.xlabel("Freqs. [MHz]")
plt.ylabel("Voltage Difference Ratio")
plt.grid()
plt.legend()

np.savez("./data_processed/A04_calc_systematic_air_normalization.npz",
         freq=master_freqs,
         ratio=np.power(10.0, (dbm(mine_signal_) - dbm(mine_signal)) / 20.0))

plt.show()
