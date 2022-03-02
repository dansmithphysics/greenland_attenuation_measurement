import glob
import numpy as np


def load_templates(file_names):
    """
    Loads csv files from oscilloscope data,
    and averages time traces together.

    Returns time and everaged trace.
    """

    template_time, template_trace = [], []

    for i_file_name, file_name in enumerate(file_names):
        data_time, data_trace = np.loadtxt(file_name,
                                           delimiter=",",
                                           skiprows=6,
                                           usecols=(3, 4),
                                           unpack=True)

        if(i_file_name == 0):
            template_time = data_time
            template_trace = data_trace
        else:
            template_trace += data_trace

    template_trace /= float(len(file_names))
    template_trace -= np.mean(template_trace[:100])

    return template_time, template_trace


if(__name__ == "__main__"):

    ###################################
    # Data collected with healthy FID #
    ###################################

    file_names = glob.glob("./data_raw/2021_08_02_0db_run_sharper_trigger/*_Ch1.csv")
    template_time, template_trace = load_templates(file_names)

    np.savez("./data_processed/averaged_in_ice_trace",
             template_time=template_time,
             template_trace=template_trace)

    ###########################################
    # Data collected for Biref, unhealthy FID #
    ###########################################

    file_names = glob.glob("./data_raw/2021_08_09_biref_*/*_Ch1.csv")
    template_time, template_trace = load_templates(file_names)
    np.savez("./data_processed/averaged_in_ice_trace_biref",
             template_time=template_time,
             template_trace=template_trace)

    ###########################################
    # Data collected with healthy FID, in air #
    ###########################################

    file_names = glob.glob("./data_raw/2021_08_02_psuedo_air_to_air_with_46db_att_Ch1.csv")
    template_time, template_trace = load_templates(file_names)
    np.savez("./data_processed/averaged_in_air_trace",
             template_time=template_time,
             template_trace=template_trace)
