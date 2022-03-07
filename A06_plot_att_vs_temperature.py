import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.stats import linregress


# Load bulk measurement
result_data = np.load("./data_processed/A05_mc_results.npz")

result_freqs = result_data['freqs']
result_low_bound = result_data['low_bound']
result_high_bound = result_data['high_bound']
result_middle_val = result_data['middle_val']

# Load up grip data
data_depth, data_temp = np.loadtxt("./data_raw/griptemp.txt",
                                   skiprows=40,
                                   unpack=True,
                                   delimiter="\t")

f_temp_of_depth = scipy.interpolate.interp1d(data_depth,
                                             data_temp,
                                             kind='linear',
                                             bounds_error=False,
                                             fill_value='extrapolate')


# now, lets find those slopes.
data_temp, data_att = np.loadtxt("./data_raw/measured_att_vs_temp.txt",
                                 comments="!",
                                 unpack=True,
                                 delimiter=", ")
data_temp_sum = data_temp[-3:]
data_att_sum = data_att[-3:]

# alright, we want slopes...
slope, intercept, r_value, p_value, std_err = linregress(data_temp_sum, np.log10(data_att_sum))

# so we have an att vs freq
# for each att, we have an integration of the depth / temperature
# so the final attenuation is,
# <L> = integral( L(T(x)) dx ) / integral( dx )
# now that L(T) is known to be of the form:
# log10(L(T(x))) = m * T(x) + b,
# -> L(T(x)) = 10^(m * T(x)) * A
# <L> = A * integral( 10^(m * T(x)) * dx ) / integral( dx )
# and the T of x is from the above function!
# The above equation is not true, it must be the harmonic mean,
# not the arithmetic mean


def att(m, depth):
    return np.power(10.0, m * f_temp_of_depth(depth))


nsteps = 100000

m = slope
depth = 0.0  # m
ddepth = 0.1  # m

int_L = att(m, depth)

for step in range(nsteps):
    depth += ddepth
    int_L += ddepth / att(m, depth)
    if(depth > 3008.0):
        break

# oh you know what, this integral might be a Tiny bit wrong
# because it assumes uniform stepping, which isn't right, spends more time in deep?
# Hm maybe not

avg_L_calc = depth / int_L

selection_region = np.logical_and(result_freqs > 10e6, result_freqs < 500e6)
att_targets_middle = result_middle_val[selection_region]
att_targets_high = result_high_bound[selection_region]
att_targets_low = result_low_bound[selection_region]

depths = np.linspace(0.0, 3008.0, 10000)

# intercepts for the middle value and error bars as derived from bulk attenution
As_middle = att_targets_middle / avg_L_calc
As_high = att_targets_high / avg_L_calc
As_low = att_targets_low / avg_L_calc

# turns out this is just a scaling factor, since the intercepts will cancel out for the same frequency.
# so the scale factor of interest is equal to:

print("scale of <L(top 1500 m)> / <L(all)>:", np.mean(1.0 / att(m, depths)) / np.mean(1.0 / att(m, depths[depths <= 1500.0])))

# so now, we want to convert it to the following:
# L(x) = A * 10^(m * T(x))
# start with the As from before
# then sweep over x's

colors = ['red', 'blue', 'green', 'black']

for i in range(len(result_freqs[selection_region])):
    print(i, result_freqs[selection_region][i] * 1e-6, As_middle[i])

selections = [0, 10]
labels = ["150 MHz", "330 MHz"]
colors = ['black', 'red', 'orange']
hatches = ["//", "\\\\", "|"]
plt.figure(figsize=(5, 4))
ax = plt.subplot(1, 1, 1)
ax.grid()

for i_selection in range(len(selections)):
    for i, A in enumerate(As_middle):
        if(i != selections[i_selection]):
            continue

        L_x_middle = As_middle[i] * att(m, depths)
        L_x_high = As_high[i] * att(m, depths)
        L_x_low = As_low[i] * att(m, depths)

        print("Ratio is:", (np.mean(As_middle[i] * att(m, depths[depths < 1500.0])) /
                            np.mean(As_middle[i] * att(m, depths))))

        ax.axhline(result_middle_val[i] * 1.1997683538791095)

        ax.axhline(np.mean(L_x_middle[depths < 1500.0]), linestyle="--")

        ax.fill_between(depths,
                        L_x_high,
                        L_x_low,
                        alpha=0.95,
                        facecolor="none",
                        edgecolor=colors[i_selection],
                        hatch=hatches[i_selection],
                        label=labels[i_selection])

plt.legend()
ax.set_ylim(300.0, 1100.0)
ax.set_xlim(0.0, 3050.0)
plt.ylabel(r"Field Attenuation Length, $L_\alpha$ [m]")
ax.set_xlabel("Depth [m]")

plt.tight_layout()
plt.savefig("./plots/A06_plot_att_vs_temperature_result.png",
            dpi=300)

plt.show()
