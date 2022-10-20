Atmospheric attenuation statistics for Alphasat at Louvain-la-Neuve, Belgium for two frequecies 19.7 and 39.402 GHz.

STATISTICS-MEASUREMENT : contains the excess attenuation statistics (CCDF) for the periods:
. 01 Sep 2018 - 30 Aug 2019
. 01 Sep 2019 - 30 Aug 2020
. 01 Sep 2018 - 30 Aug 2020

The data are provided for two frequencies 19.7 GHz (ch4) and 39.402 GHz (ch2).

All of the data are in hdf5 format produced by Python library pandas.

To access the contents of the files and plot the statistics, a Jupyter Notebook (*PLOT.ipynb*) along side *utils.py* file can be used. The *PLOT.ipynb* contais two cells which one is for plotting the statistics and one is for error metric analysis. The user needs to only define the frequency in GHz (*FREQ* variable) inside the cells.

In case of any difficulties please send email to: 
sayed.razavian@uclouvain.be
claude.oestges@uclouvain.be
danielle.vanhoenacker@uclouvain.be

# Achnowledgment
The Universite catholique de Louvain is thanked for the access to the Alphasat ground station data. The daily data can be visualized on the website at *http://130.104.205.199:8080/alphasat/graphs/today.html*. The statistics and simulation data for time periods 01 Sep 2018 to 30 Aug 2020 can be accessed on the *https://github.com/mrazavian/Alphasat-Data-UCLouvain*


# Citation
If you find this project useful, please cite:

Mojtaba Razavian, Claude Oestges, Danielle Vanhoenacker-Janvier, "Synthetic rain models and optical flow algorithms for improving the resolution of rain attenuation time series simulated from Numerical Weather Prediction", AGU Radio Science....