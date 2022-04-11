import numpy as np
import matplotlib.pyplot as plt
import experiment
import A05_att_with_errors


def main(exper_constants, ice_file_name, air_file_name, end_times, freq_select):

    vals = np.zeros(len(end_times))
    vals_min = np.zeros(len(end_times))
    vals_max = np.zeros(len(end_times))

    print("Start [s] \t End [s] \t Dur. [s]")
    for i_end_time, end_time in enumerate(end_times):
        print("%.2E \t %.2E \t %.2E" % (exper_constants.gb_start, end_time, end_time-exper_constants.gb_start))

        exper_constants.gb_end = end_time
        exper = experiment.Experiment(exper_constants, ice_file_name, air_file_name)
            
        freqs, low_bound, high_bound, middle_val = A05_att_with_errors.main(exper_constants,            
                                                                            exper,
                                                                            nthrows=1000,
                                                                            verbose=False)

        vals[i_end_time] = middle_val[freq_select]
        vals_min[i_end_time] = low_bound[freq_select]
        vals_max[i_end_time] = high_bound[freq_select]
        
    plt.figure(figsize=(5, 4))

    plt.fill_between((end_times - exper_constants.gb_start) / 1e-6,
                     vals_min,
                     vals_max,
                     color='red',
                     alpha=0.5,
                     label="Modified Time Window Result, 1 $\sigma$ Errors")

    plt.scatter(36.1 - exper_constants.gb_start / 1e-6,
                800.0,
                color='black')

    plt.errorbar(36.1 - exper_constants.gb_start / 1e-6,
                 800.0,
                 yerr=([800.0 - 710.0], [915.0 - 800.0]),
                 label="Nominal Result, 1 $\sigma$ Errors",
                 color='black',
                 ls='none')

    plt.grid()
    plt.xlim(0, 4.0)
    plt.ylim(500.0, 1100.0)

    yticks = np.arange(500, 1101, 100)
    yticks = np.sort(yticks)
    yticklabels = [str(int(i)) if i % 100 == 0 else '' for i in yticks]
    plt.yticks(yticks, labels=yticklabels)

    plt.xlabel("Window Length Used in Analysis [$\mu$s]")
    plt.ylabel("Bulk Field Attenuation Length at 200 MHz [m]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plots/A05_att_increased_window.png",
                dpi=300)
    plt.show()

if __name__ == "__main__":

    exper_constants = experiment.ExperimentConstants()    
    ice_file_name = "./data_processed/averaged_in_ice_trace.npz"
    air_file_name = "./data_processed/averaged_in_air_trace.npz"

    # The times tested for the end of the ground bounce
    end_times = np.linspace(exper_constants.gb_start + 0.1e-6,
                            exper_constants.gb_start + 4.0e-6,
                            15)
    end_times = np.append(end_times, [36.05e-6])
    end_times = np.sort(end_times)

    # Index of frequency used to plot
    freq_select = 2  # 200 MHz
    
    main(exper_constants, ice_file_name, air_file_name, end_times, freq_select)
