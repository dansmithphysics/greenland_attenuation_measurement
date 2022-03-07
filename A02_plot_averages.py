import glob
import numpy as np
import matplotlib.pyplot as plt
import analysis_funcs
import experiment


def load_and_plot(file_name, att_correction, title,
                  time_offset=0,
                  xlim=None, ylim=None, fig_size=None, save_name=None):

    data_t, data_trace = analysis_funcs.load_file(file_name,
                                                  att_correction,
                                                  time_offset)
    
    if fig_size is None:
        plt.figure()
    else:
        plt.figure(figsize=fig_size)

    plt.tight_layout()

    plt.title(title)
    plt.plot(data_t * 1e6, data_trace * 1e3,
             alpha=1.0, color='black', linewidth=1.0)
    plt.xlabel("Absolute Time Since Transmitted Pulse [$\mu$s]")
    plt.ylabel("Voltage [mV]")

    plt.axvspan(35.55, 35.55 + (35.05 - 34.59),
                0.0, 1.0, alpha=0.25, color='purple',
                label="Bedrock Echo")
    plt.legend()

    plt.subplots_adjust(bottom=0.2)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid()
    if save_name is not None:
        plt.savefig(save_name,
                    dpi=300,
                    bbox_inches='tight')


def load_and_plot_sliding_power(file_name, att_correction, title,
                                time_offset=0,
                                xlim=None, ylim=None, fig_size=None,
                                save_name=None, window_length=250):

    data_t, data_trace = analysis_funcs.load_file(file_name,
                                                  att_correction,
                                                  time_offset)

    if fig_size is None:
        plt.figure()
    else:
        plt.figure(figsize=fig_size)

    plt.tight_layout()

    data_t, rolling = analysis_funcs.power_integration(data_t, data_trace, window_length)
        
    plt.title(title)
    plt.semilogy(data_t * 1e6, rolling,
                 alpha=1.0, color='black', linewidth=1.0)

    plt.axvspan(35.55, 35.55 + (35.05 - 34.59),
                0.0, 1.0, alpha=0.25, color='purple',
                label="Bedrock Echo")
    plt.legend()

    plt.subplots_adjust(bottom=0.2)

    plt.xlabel("Absolute Time Since Transmitted Pulse [$\mu$s]")
    plt.ylabel("Integrated Power [mW ns]")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid()
    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    exper_constants = experiment.Experiment()

    load_and_plot("data_processed/averaged_in_ice_trace.npz",
                  att_correction=exper_constants.ice_att,
                  time_offset=exper_constants.time_offset,
                  title="Ice Echo Data",
                  xlim=(-1.0, 40.0),
                  ylim=(-10.0, 10.0),
                  save_name="./plots/A02_plot_averages_inice_zoomed_out.png")

    load_and_plot("data_processed/averaged_in_ice_trace.npz",
                  att_correction=exper_constants.ice_att,
                  time_offset=exper_constants.time_offset,
                  title="",
                  xlim=(10.0, 43.0),
                  ylim=(-1.0, 1.0),
                  fig_size=(8, 3),
                  save_name="./plots/A02_plot_averages_inice.png")

    load_and_plot("data_processed/averaged_in_ice_trace.npz",
                  att_correction=exper_constants.ice_att,
                  time_offset=exper_constants.time_offset,
                  title="",
                  xlim=(34.0, 39.0),
                  ylim=(-0.75, 0.75),
                  fig_size=(3.8, 3),
                  save_name="./plots/A02_plot_averages_inice_ground_bounce.png")

    load_and_plot_sliding_power("data_processed/averaged_in_ice_trace.npz",
                                att_correction=exper_constants.ice_att,
                                time_offset=exper_constants.time_offset,
                                title="",
                                xlim=(10.0, 43.0),
                                ylim=(5e-7, 5e-4),
                                fig_size=(8, 3),
                                save_name="./plots/A02_plot_averages_inice_integrated.png")

    load_and_plot_sliding_power("data_processed/averaged_in_ice_trace.npz",
                                att_correction=exper_constants.ice_att,
                                time_offset=exper_constants.time_offset,
                                title="",
                                xlim=(34.0, 39.0),
                                ylim=(5e-7, 5e-4),
                                fig_size=(3.8, 3),
                                save_name="./plots/A02_plot_averages_inice_ground_bounce_integrated.png")

    load_and_plot("data_processed/averaged_in_air_trace.npz",
                  att_correction=0.0,
                  time_offset=0.0,
                  title="Air-to-Air Result",
                  xlim=(-0.07, 0.4),
                  ylim=(-500.0, 500.0),
                  save_name="./plots/A02_plot_averages_inair_uncorrected.png")

    plt.show()
