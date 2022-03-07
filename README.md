# Bulk Attenuation Measurement at Summit Station, Greenland.

Analysis code for the *in situ* broadband measurement of the radio frequency attenuation length at Summit Station, Greenland. The data was collected by Dan Smith, Bryan Hendricks and Christoph Welling in Summer 2021 for the RNO-G collaboration.

As of March 2022, the results are in a paper out for review in the Journal of Glaciology. The current citation is:

*In situ*, broadband measurement of the radio frequency attenuation length at Summit Station, Greenland, J.A. Aguilar *et al.* e-Print: [2201.07846](https://arxiv.org/abs/2201.07846) [astro-ph.IM]

Please direct all code-related questions to [danielsmith@uchicago.edu](mailto:danielsmith@uchicago.edu).

## Data

The data taken in the field is available from the RNO-G wiki. The three data sources needed for the analysis and their locations on the wiki are listed below. 

* [In ice data](https://radio.uchicago.edu/wiki/images/0/04/Groundbounce_in_ice_data_2021_08_02.zip)
* [In air data](https://radio.uchicago.edu/wiki/images/f/fd/Groundbounce_in_air_data_46dB_2021_08_02.zip)
* [LPDA S11 Measurement](https://radio.uchicago.edu/wiki/images/2/28/LPDA_Summit_Measurements.zip)

Simulations are also required. The simulations performed on xFDTD are not yet documented elsewhere so are included in this repository [here](data_simulated/lpda_xfdtd_sims/). The simulations performed in NuRadioMC are available [here for the in air simulation](http://arianna.ps.uci.edu/~arianna/data/AntennaModels/createLPDA_100MHz_InfAir/) and [here for the in ice simulation](http://arianna.ps.uci.edu/~arianna/data/AntennaModels/createLPDA_100MHz_InfFirn_n1.4/), and are documented [here](https://github.com/nu-radio/NuRadioReco/wiki/Antenna-models). 

For the air-to-air normalization study, the amplifier response and s-parameter files of filters are also required. They're included in the repository. The amplifier response, found [here](data_raw/amp_board_v3.txt), can be derived from [RNO-G's detector paper, Fig. 17](https://arxiv.org/pdf/2010.12279.pdf). The s-parameter files are all from Minicircuits website and datasheets for [VHF-145+.S2P](https://www.minicircuits.com/pdfs/VHF-145+.pdf) and [VLF-575+](https://www.mouser.com/datasheet/2/1030/VLF-575-1701652.pdf). 

For the conversion of bulk attenuation to attenuation at any given depth, the temperature of the ice as a function of depth from the GRIP borehole is required. This data is available from the Greenland Ice Core Project server [here](ftp://ftp.ncdc.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/physical/griptemp.txt). Also required is the measured relationship between attenuation and temperature as [Bogorodsky *et al.*](https://doi.org/10.1007/978-94-009-5275-1). This data is in the repository [here](data_raw/measured_att_vs_temp.txt) and can be derived from the data plotted in the paper by [Avva *et al.* here](https://arxiv.org/pdf/1409.5413.pdf).

## Analysis Scripts

The analysis code is roughly arranged in sequential scripts that prepare data (`A01`), plot results (`A02`), and calculate systematic uncertainties (`A03`) and biases (`A04`) before calculating the bulk attenuation (`A05`) and plotting the figures in the paper (`A06`). A description of each script is below.

All scripts were built in Python 3.9.5.

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

Plots the absolute response of the air-to-air path to determine if there are any substantial systematic bias from reflections. There are nonw. This script creats Fig. 5 from the paper. 

### A04_plot_lpda_simulations.py

Plots the NuRadioMC simulation in several formats to determine if there is an appreciable shift in gain from in-air to in-ice simulations. There are none. Outputs a gain figure for the simulations in ice and in air, saved [here](./plots/A04_gain_ice_vs_air.png), and an effective height figure, saved [here](./plots/A04_effective_height_ice_vs_air.png).

### A05_att_with_errors.py

Calculates the bulk attenuation using a Toy MC method to determine the statistical uncertainty of each frequency bin. Outputs to a file named `/data_processed/A05_mc_results.npz` which includes the slopes (`ms`) and intercepts (`bs`) of the linear fit for each Toy MC iteration, the frequencies where attenuation is calculated (`freqs`), and the lower, central, and upper bounds of the attenuation (`low_bound`, `middle_val`, and `high_bound` respectively). 

### A05_att_with_errors_vs_window.py

Calculates and plots the bulk attenuation at 200 MHz as a function of window length, creating Fig. 3 from the paper. Note that the result is slightly different than the paper for yet undetermined reason. The most likely cause is a slight change in noise contribution definition which does not change the mean value of measurement but does change the lower bound. 

### A06_plot_att.py

Processes the output of `A05_att_with_errors.py` to create the attenuation Fig. 6 and Fig. 7 from the paper. Calculations include calculating the uncertainty on the linear fit to create the final fit confidence interval. The bulk attenuation is converted to the average attenution of the top 1500 m of the ice by scaling the bulk result by a factor of `~1.20`, determined by the script `A06_plot_att_vs_temperature.py` to be an appropriate correction to convert between the two. 

### A06_plot_att_vs_temperature.py

Converts the bulk attenuation into an attenuation as function of depth via the process described in the paper in the section named **Discussion and Summary** and in Eq. 8. Creates Fig. 8 from the paper, the attenuation vs. depth at 150 and 330 MHz.

