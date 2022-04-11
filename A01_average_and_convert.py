import glob
import numpy as np


def load_and_pickle_templates(input_file_names, output_file_name):
    """
    Loads csv files from oscilloscope data,
    and averages time traces together.
    """

    template_time, template_trace = np.array([]), np.array([])

    for i_file_name, file_name in enumerate(input_file_names):
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

    template_trace /= float(len(input_file_names))
    template_trace -= np.mean(template_trace[:100])
    
    np.savez(output_file_name,
             template_time=template_time,
             template_trace=template_trace)


if(__name__ == "__main__"):

    # Data collected with healthy FID
    input_file_names = glob.glob("./data_raw/2021_08_02_0db_run_sharper_trigger/*_Ch1.csv")
    if(len(input_file_names) == 0):
        print("No files found for in-ice data. Are you sure you've downloaded them to ./data_raw/?")
    else:
        output_file_name = "./data_processed/averaged_in_ice_trace"
        load_and_pickle_templates(input_file_names, output_file_name)
    
    # Data collected for Biref, unhealthy FID
    input_file_names = glob.glob("./data_raw/2021_08_09_biref_*/*_Ch1.csv")
    if(len(input_file_names) == 0):
        print("No files found for in-ice biref data. Are you sure you've downloaded them to ./data_raw/?")
    else:
        output_file_name = "./data_processed/averaged_in_ice_trace_biref"
        load_and_pickle_templates(input_file_names, output_file_name)

    # Data collected with healthy FID, in air
    input_file_names = glob.glob("./data_raw/2021_08_02_psuedo_air_to_air_with_46db_att_Ch1.csv")
    if(len(input_file_names) == 0):
        print("No files found for in-air data. Are you sure you've downloaded them to ./data_raw/?")
    else:
        output_file_name = "./data_processed/averaged_in_air_trace"
        load_and_pickle_templates(input_file_names, output_file_name)
