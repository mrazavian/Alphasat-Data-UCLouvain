# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing.sharedctypes import Value

import numpy as np 
import pandas as pd
from scipy.interpolate import interp1d
from pandas.core import series
from pyproj import CRS, Proj, Geod
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

import os
import warnings
import socket
import numbers
import subprocess
import sys 


#%%
def statistic_old(timeseries, name='value', bins=10):
    """
    Parameters:
    -----------
    - timeseries : sequene or numpy.array
    An array of time series. the values of np.nan can also be in the array. 
    if theere are np.nan values in the timeseries, the np.nan values will
    be deleted.

    - bins : int or sequence of scalars or str, optional
    If bins is an int, it defines the number of equal-width bins in the
    given range (10, by default). If bins is a sequence, it defines a 
    monotonically increasing array of bin edges, including the rightmost
    edge, allowing for non-uniform bin widths.

    Returns:
    --------
    - stat : structure
        * pdf : (array [%]) probability density function (Ci/(N*Wi))
        * ccdf : (array [%]) cumplimantary cumulative distribution function
        * bin_edges : (array) Return the bin edges (length(ccdf)+1).
        * bin_width : (array) 
    """
    # remove the nan value
    not_nan = np.logical_not(np.isnan(timeseries)) # find the nan values
    timeseries = timeseries[not_nan]

    # considers negative values to 0
    ind_negative = timeseries<0
    timeseries[ind_negative] = 0

    # calculate the pdf. Note: the bins should be equally spaced
    hist, bin_edges = np.histogram(timeseries, density=True, bins=bins)
        
    # calculate the width of each bin
    bin_dx = ( bin_edges[1:] - bin_edges[:-1] )
    # calculate the area of each bin and then the CCDF
    ts_cdf = np.cumsum(hist * bin_dx)
    ts_ccdf = 1 - ts_cdf

    stats = {"pdf" : np.hstack((np.nan,hist)) * 100,
            "cdf" : np.hstack((np.nan,ts_cdf)) * 100,
            "ccdf" : np.hstack((np.nan,ts_ccdf)) * 100,
            name : bin_edges}
    
    stats_df = pd.DataFrame(stats)
    return stats_df

#%%
def statistic(timeseries, name='value', continuous=False):
    """
    Parameters:
    -----------
    - timeseries : sequene or numpy.array
    An array of time series. the values of np.nan can also be in the array. 
    if theere are np.nan values in the timeseries, the np.nan values will
    be deleted.
    - name: string
        name of the output timeseries
    - continous: Boolean
        if it very reasonable to make the assumption that there is 
        only one occurence of each value in the sample (typically encountered 
        in the case of continuous distributions) then the groupby() + agg('count') 
        is not necessary (since the count is always 1).

    Returns:
    --------
    - stat : pandas.DataFrame()
        * values : timeseries
        * pdf : probability density function (Ci/(N*Wi)) (0<pdf<100) Only if continuous=False
        * cdf : cumulative distribution function (0<cdf<100)
        * ccdf : (array [%]) cumplimantary cumulative distribution function (0<ccdf<100)
    """
    #---remove the nan value--> in pandas the nan values are removed automatically--
    # not_nan = np.logical_not(np.isnan(timeseries)) # find the nan values
    # timeseries = timeseries[not_nan]
    
    # considers negative values to 0
    ind_negative = timeseries<0
    timeseries[ind_negative] = 0

    series = pd.Series(timeseries, name = name)
    df = pd.DataFrame(series)
    
    if continuous:
        df['cdf'] = df.rank(method = 'average', pct = True) * 100
        df['ccdf'] = 100 - df['cdf']
        df['pdf'] = np.nan
        stats_df = df.sort_values(name)
    else:
        stats_df = df.groupby(name)[name].agg('count').pipe(pd.DataFrame).rename(columns = {name: 'frequency'})

        # PDF
        stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency']) * 100

        # CDF
        stats_df['cdf'] = stats_df['pdf'].cumsum()

        # CCDF
        stats_df['ccdf'] = 100 - stats_df['cdf']
        stats_df = stats_df.reset_index()  

    return stats_df

#%%
def plot_statistic(file_names, freq, satellite_name, Ts=60, ax=None, 
        rain_multiexcell=True, rain_wrf=True, gas_ox=False, gas_wv=False, cloud=False, 
        scintillation=False, contribution=True, continuous=False, title=None, ylabel='Excess attenuation [dB]', 
        xlabel='Time [%]', xlim=(0.001, 100), ylim=(0, 40) , style='-',
        version='new', ignore_index=False, availability_scale=False, fig_name=None):
    """plot time series based on the time
    Parameters
    ----------
    -file_name: string, directory, array
        Directories point to *.h5 files
     - rain: Boolean
        if True plot the rain attenuation time-series
    - gas: Boolean
        if True plot the gaseous attenuation time-series
    - cloud: Boolean
        if True plot the cloud attenuation time-series
    - scintillation: Boolean
        if True plot the scintillation attenuation time-series
    - title: string
        title of the figure
    - ylabel: string
    - xlabel: string
    - xlim : 2-tuple/list
        Set the x limits of the current axes.
    - ylim : 2-tuple/list
        Set the y limits of the current axes.
    - style : list or dict
        The matplotlib line style per column.
    - fig_name : string, directory, path
        figure name to be saved  
    
    Returns
    -------
    - fig
    - ax
    - availability : pd.DataFrame
        a value between 0-1 shows the availability of the data
    """  
    
    frames, field_names = field_contribution(file_names, satellite_name, freq, Ts, 
                                    rain_multiexcell, rain_wrf, gas_ox, gas_wv, cloud, 
                                    scintillation, contribution, ignore_index)
    
    availability = pd.DataFrame(index=['{} TOTAL'.format(satellite_name),
                                       '{} VALID'.format(satellite_name),
                                       '{} AVAILABILITY'.format(satellite_name)])

    style = np.atleast_1d(style)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)

    ii = 0
    for name in field_names:
        if version.lower()=='new':
            stats_df = statistic(frames[name].dropna(), name.replace('_timeseries',''), continuous=continuous)
        else:
            stats_df = statistic_old(frames[name].dropna(), name.replace('_timeseries',''), bins=200)
        

        TOT = frames[name].shape[0]
        VAL = frames[name].dropna().shape[0]
        AVL = VAL / TOT
        availability[name] = [TOT, VAL, AVL ]
    
        if availability_scale:
            stats_df['ccdf'] = stats_df['ccdf'] * availability[name][2]

        ax = stats_df.plot(ax=ax, x = 'ccdf', y = [name.replace('_timeseries','')], logx=True, xlim=xlim, 
                    ylim=ylim, title=title, xlabel=xlabel, ylabel=ylabel, style=style[ii])
        
        if np.size(style)>1:
            ii = ii + 1

    # ax.grid(which='both', linestyle=':', color='gray',
    #         linewidth=0.3, alpha=0.5)
    
    if fig_name is not None:
        fig.savefig(fig_name, bbox_inches="tight")

    return fig, ax, availability

#%%       Error Analysis
def error_statistics(file_names, X_ref, Y_ref, mode, freq, satellite_name, Ts=60, 
        rain_multiexcell=True, rain_wrf=True, gas_ox=False, gas_wv=False, cloud=False, 
        scintillation=False, contribution=True, continuous=False,
        version='new', ignore_index=False, availability_scale=False):
    """plot time series based on the time
    Parameters
    ----------
    -file_name: string, directory, array
        Directories point to *.h5 files
     - rain: Boolean
        if True plot the rain attenuation time-series
    - gas: Boolean
        if True plot the gaseous attenuation time-series
    - cloud: Boolean
        if True plot the cloud attenuation time-series
    - scintillation: Boolean
        if True plot the scintillation attenuation time-series
    - title: string
        title of the figure
    - ylabel: string
    - xlabel: string
    - xlim : 2-tuple/list
        Set the x limits of the current axes.
    - ylim : 2-tuple/list
        Set the y limits of the current axes.
    - style : list or dict
        The matplotlib line style per column.
    - fig_name : string, directory, path
        figure name to be saved  
    
    Returns
    -------
    - mu :  mean
    - sig : standard deviation
    - rho : root mean square (np.sqrt(mu**2 + sig**2))
    """  
    frames, field_names = field_contribution(file_names, satellite_name, freq, Ts, 
                                    rain_multiexcell, rain_wrf, gas_ox, gas_wv, cloud, 
                                    scintillation, contribution, ignore_index)
    
    mu = []
    sig = []

    for name in field_names:
        if version.lower()=='new':
            stats_df = statistic(frames[name].dropna(), name.replace('_timeseries',''), continuous=continuous)
        else:
            stats_df = statistic_old(frames[name].dropna(), name.replace('_timeseries',''), bins=200)
        
        fcn =  interp1d(stats_df['ccdf'].to_numpy(),stats_df[name.replace('_timeseries','')].to_numpy(), kind='linear', bounds_error=True)

        Ap = fcn(X_ref)

        if mode==1:
            err = Ap - Y_ref
        elif mode==2:
            err = (Ap - Y_ref) / Y_ref
        elif mode==3:
            V = Ap / Y_ref
            err = np.where(Y_ref < 10, np.log(V) * (Y_ref/10)**0.2, np.log(V))

        mu = np.append( mu, np.mean( np.abs(err)) )
        sig = np.append( sig, np.std( np.abs(err)) )
    
    rho =  np.sqrt(mu**2 +  sig**2)

    return mu, sig, rho, field_names


#%%    field Contribution
def field_contribution(file_names, satellite_name, freq, Ts, 
        rain_multiexcell=True, rain_wrf=True, gas_ox=False, gas_wv=False, cloud=False, 
        scintillation=False, contribution=False, ignore_index=False):
    """This function prepare the field contribution for different 
    time series attenuation
    """
    file_names = np.atleast_1d(file_names)

    data_frame = pd.DataFrame()
    for fname in file_names:
        df = pd.read_hdf(fname, key=satellite_name.upper(), mode='r')
        data_frame = data_frame.append(df, ignore_index=ignore_index)

    data_frame = data_frame.sort_index()

    field_name = []
    if gas_ox:
        field_name = np.append(field_name, 
            ['oxygen_att_timeseries_freq-{}'.format(freq)])
    
    if gas_wv:
        field_name = np.append(field_name, 
            ['water_vapour_att_timeseries_freq-{}'.format(freq)])

    if cloud:
        field_name = np.append(field_name, 
            ['cloud_att_timeseries_freq-{}'.format(freq)])
    
    if scintillation:
        field_name = np.append(field_name, 
            ['scint_att_timeseries_freq-{}'.format(freq)])


    field_rain = []
    rain = False
    if rain_wrf:
        field_rain = ['original_rain_att_timeseries_freq-{}'.format(freq)]
        rain = True
    
    if rain_multiexcell:
        field_rain = np.append(field_rain, ['synthetic_rain_att_NC_timeseries_freq-{}'.format(freq)] )
        rain = True



    if contribution:
        data_frame['Total'] = data_frame[field_name].sum(axis=1,skipna=False) 

        if rain:
            field_name = []
        else:
            field_name = ['Total']

        for frain in field_rain:

            field_name = np.append(field_name, frain+'+others')

            data_frame[ frain+'+others' ] = data_frame[ np.append( frain, ['Total'] ) ].sum(axis=1,skipna=False) 
    else:
        field_name = np.append(field_name, field_rain )


    if not ignore_index:
        Time_selection = pd.date_range(data_frame.index[0],data_frame.index[-1], freq='{}S'.format(Ts))

        data_frame = data_frame.loc[Time_selection,:]
    
    return data_frame, field_name
