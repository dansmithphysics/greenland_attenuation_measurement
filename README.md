# greenland_ground_bounce
Analysis code for radio echo data collected in Greenland, Summer 2021

## Data

The data taken in the field is available from the RNO-G wiki. The three data sources needed for the analysis and their locations on the wiki are listed below. 

* [In ice data](https://radio.uchicago.edu/wiki/images/0/04/Groundbounce_in_ice_data_2021_08_02.zip)
* [In air data](https://radio.uchicago.edu/wiki/images/f/fd/Groundbounce_in_air_data_46dB_2021_08_02.zip)
* [LPDA S11 Measurement](https://radio.uchicago.edu/wiki/images/2/28/LPDA_Summit_Measurements.zip)

Simulations are also required. The simulations performed on xFDTD are not yet documented elsewhere so are included in this repository [here](data_simulated/lpda_xfdtd_sims/). The simulations performed in NuRadioMC are available [here for the in air simulation](http://arianna.ps.uci.edu/~arianna/data/AntennaModels/createLPDA_100MHz_InfAir/) and [here for the in ice simulation](http://arianna.ps.uci.edu/~arianna/data/AntennaModels/createLPDA_100MHz_InfFirn_n1.4/), and are documented [here](https://github.com/nu-radio/NuRadioReco/wiki/Antenna-models). 

For the air-to-air normalization study, the amplifier response and s-parameter files of filters are also required. They're included in the repository. The amplifier response can be derived from [RNO-G's detector paper, Fig. 17](https://arxiv.org/pdf/2010.12279.pdf). The s-parameter files are all from Minicircuits website and datasheets for [VHF-145+.S2P](https://www.minicircuits.com/pdfs/VHF-145+.pdf) and [VLF-575+](https://www.mouser.com/datasheet/2/1030/VLF-575-1701652.pdf). 

## Analysis Scripts

### A01_average_and_convert.py

Loads the raw data from the oscilliscope and converts it into Numpy's pickled format `.npy`.

### A02_plot_averages.py

Plots the time domain of the data as well as a sliding integration window version of the data. This script creates Fig. 2 from the paper. 

### A02_plot_spectrogram.py

Plots a spectrogram of the raw data, used to check the spectral content of the ground bounce. Figure saved [here](./plots/A02_plot_spectrogram.png).

### A03_calc_uncertainty_depth.py

Calculates the depth and uncertainty on the depth calculation based on the time of arrival of the ground bounce and the uncertainty of ice models. Outputs a figure of the CDF, saved [here](./plots/A04_calc_uncertainty_plots_time_to_depth.png)

### A03_calc_uncertainty_match.py

Calculates the match and uncertainty on the match of the antenna going from air to ice. This script creates Fig. 4 from the paper.

### A03_calc_uncertainty_t0.py

Calculates the arrival time and uncertainty on the arrival time of the ground bounce. The uncertainty is a subdominate effect and did not go towards the uncertainty of the final attenuation measurement.

### A03_calc_uncertainty_focusing_factor

Calculates the focusing factor and associated uncertainty using a few different approaches.

### A04_calc_systematic_air_normalization.py

### A04_plot_lpda_simulations.py

### A05_att_with_errors.py

### A05_att_with_errors_vs_window.py

### A06_plot_att.py

### A06_plot_att_vs_temperature.py

