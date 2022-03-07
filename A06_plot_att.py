import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import analysis_funcs


def gaus_2d(X, sig_x, sig_y, rho, x0, y0):
    x, y = X
    norm = 2.0 * np.pi * sig_x * sig_y * np.sqrt(1.0 - rho * rho)
    return 1 / norm * np.exp(- 0.5 / (1.0 - rho * rho) *
                             (np.square((x - x0) / sig_x) -
                              2.0 * rho * (x - x0) * (y - y0) / (sig_x * sig_y) +
                              np.square((y - y0) / sig_y)))


def calculate_fit(ms, bs):

    ms = np.array(ms) * 1e9
    bs = np.array(bs)

    x0 = np.mean(ms)
    y0 = np.mean(bs)

    sig_x = np.std(ms)
    sig_y = np.std(bs)

    rho = -0.95

    X, Y = np.meshgrid(np.linspace(-10e2, -2e2, 201),
                       np.linspace(600.0, 1200.0, 201))
    Z = gaus_2d((X, Y), sig_x, sig_y, rho, x0, y0)
    levels = np.array([gaus_2d((x0 + 3.0 * sig_x, y0 + 3.0 * sig_y), sig_x, sig_y, rho, x0, y0),
                       gaus_2d((x0 + 2.0 * sig_x, y0 + 2.0 * sig_y), sig_x, sig_y, rho, x0, y0),
                       gaus_2d((x0 + 1.0 * sig_x, y0 + 1.0 * sig_y), sig_x, sig_y, rho, x0, y0)])

    plt.figure()
    plt.hist2d(ms, bs,
               range=((-10e2, -2e2), (600.0, 1200.0)),
               bins=(100, 100))
    plt.colorbar()
    plt.plot([], [], color='orange', label="1, 2, 3 $\sigma$ Contours, Initial Guess")
    plt.contour(X, Y, Z, levels, colors='orange')
    plt.ylabel("Intercept [m]")
    plt.xlabel("Slope [m / GHz]")
    plt.legend()

    plt.show()
    exit()


def plot_fit(ms, bs, scale, n_freq_pts=100):

    fit_x = np.linspace(0.0, 1e9, n_freq_pts)

    fit_y = np.zeros(len(fit_x))
    fit_yerr_min = np.zeros(len(fit_x))
    fit_yerr_max = np.zeros(len(fit_x))
    for i_unique_freq, unique_freq in enumerate(fit_x):
        hist_entry = ms * unique_freq + bs

        hist_entry_, cumsum = analysis_funcs.calculate_uncertainty(hist_entry)

        if(len(cumsum) == 0):
            continue

        entries_min, entries_mid, entries_max = analysis_funcs.return_confidence_intervals(hist_entry_, cumsum)

        fit_y[i_unique_freq] = entries_mid
        fit_yerr_min[i_unique_freq] = entries_min
        fit_yerr_max[i_unique_freq] = entries_max

    plt.fill_between(fit_x * 1e-6,
                     fit_yerr_min * scale,
                     fit_yerr_max * scale,
                     alpha=0.3,
                     color='red',
                     label=r"Linear Fit with 1 $\sigma$ Errors")

    plt.plot(fit_x * 1e-6,
             fit_y * scale,
             color='red')


if __name__ == "__main__":

    n_top = 1.4
    n_bot = 1.78
    ff = np.square(n_bot / n_top)

    # Taken from Avva et al.
    # Adjusted to include focusing factor.

    d_ice_avva = 3015.0 * 2.0
    thing_in_natural_log = np.exp(d_ice_avva / 947.0)
    thing_in_natural_log *= np.sqrt(ff)
    result = d_ice_avva / np.log(thing_in_natural_log)

    thing_in_natural_log = np.exp(d_ice_avva / (947.0 + 92.0))
    thing_in_natural_log *= np.sqrt(ff)
    result_err = d_ice_avva / np.log(thing_in_natural_log)
    result_err -= result

    result_data = np.load("./data_processed/A05_mc_results.npz")

    for scale_to_top_1p5k in [1.0, 1.1997683538791095]:

        x = result_data['freqs']
        yerr_min = result_data['low_bound']
        yerr_max = result_data['high_bound']
        y = result_data['middle_val']

        plt.figure(figsize=(5, 4))
        plt.errorbar(x[yerr_min != 0] * 1e-6,
                     y[yerr_min != 0] * scale_to_top_1p5k,
                     yerr=((y[yerr_min != 0] - yerr_min[yerr_min != 0]) * scale_to_top_1p5k,
                           (yerr_max[yerr_min != 0] - y[yerr_min != 0]) * scale_to_top_1p5k),
                     color='black',
                     ls='none',
                     label="Data Result with 1 $\sigma$ Errors")

        plt.errorbar(x[yerr_min == 0] * 1e-6,
                     y[yerr_min == 0] * scale_to_top_1p5k,
                     yerr=(yerr_max[yerr_min == 0] - 550.0) * scale_to_top_1p5k,
                     uplims=True,
                     color='black',
                     ls='none',
                     label="95% CL Upper Limit")

        plt.scatter(x[yerr_min != 0] * 1e-6,
                    y[yerr_min != 0] * scale_to_top_1p5k,
                    color='black')

        plt.scatter(x[yerr_min == 0] * 1e-6,
                    y[yerr_min == 0] * scale_to_top_1p5k,
                    marker="_",
                    color="black")

        plot_fit(result_data['ms'],
                 result_data['bs'],
                 scale=scale_to_top_1p5k)

        plt.errorbar([75.0], [result * scale_to_top_1p5k], yerr=[result_err * scale_to_top_1p5k],
                     color='purple', alpha=0.75)
        plt.scatter([75.0], [result * scale_to_top_1p5k],
                    color='purple', alpha=0.75, label="Avva et al. Result, Corrected")

        plt.axvline(145.0, color='red', linestyle='--', label="Bandpass Filters")
        plt.axvline(575.0, color='red', linestyle='--')

        plt.xlim(50.0, 600.0)
        plt.xlabel("Frequency [MHz]")

        if(scale_to_top_1p5k == 1.0):
            plt.ylim(500.0, 1100.0)
            plt.ylabel(r"Bulk Field Attenuation Length, $\langle L_\alpha\rangle$ [m]")
        else:
            plt.ylim(600.0, 1250.0)
            plt.ylabel(r"Avg. Field Attenuation Length of Top 1500 m [m]")

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.grid()

        if(scale_to_top_1p5k == 1.0):
            plt.savefig("./plots/A05_att_with_errors_final.png",
                        dpi=300)
        else:
            plt.savefig("./plots/A05_att_with_errors_final_1500m.png",
                        dpi=300)

    plt.show()
