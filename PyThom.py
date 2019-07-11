#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:40:22 2019

@author: everson
"""

# imports
import matplotlib.pyplot as plt   # plotting
import numpy as np                # math
import scipy.signal as sig        # signal math
import pandas as pd               # large 2D dataframe operations
import seaborn as sns             # colors!
from io import BytesIO            # reading the TS logbook
import requests                   # grabbing the TS logbook from google sheets
from MDSplus import Connection    # MDSplus
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
hitConn = Connection('landau.hit')
PLOTS_ON = 1

class Shot(object):
    def __init__(self, sN):
        self.shotNum = sN
        pass

    def logInfo(self, parameter_list):
        pass

    def preconditionData(self, parameter_list):
        pass



def get_TS_logbook():

    # get the logbook spreadsheet from google sheets and read it in to a pandas dataframe
    r = requests.get('https://docs.google.com/spreadsheet/ccc?key=1yF7RMpYl_KvZPoEwYiH1D_vVrasaTGLuPvdBoYRRM84&output=csv')
    data = r.content
    df = pd.read_csv(BytesIO(data), usecols=np.arange(1, 10))

    # drop Nan's from gaps between run days and other non-shots
    df = df.dropna(subset=['Shot', 'Plasma Species'])

    # identify and get rid of duplicates if necessary
    df_nodupes = df.drop_duplicates(subset=['Shot'])

    if(df.shape[0] != df_nodupes.shape[0]):
        print("*****WARNING! Duplicates while parsing TS logbook shots:*****")
        dupes = df.loc[df.duplicated(subset=['Shot'])]['Shot'].astype(int)
        print(dupes.to_csv(sep='\t', index=False))
        print("...excluding duplicates from parsed logbook.")

    df = df_nodupes

    df[[df.columns[0]]] = df[[df.columns[0]]].apply(pd.to_numeric, errors='coerce', downcast='integer')

    return(df)

#    df[[df.columns[3]]].plot.hist(bins=50)
#    particular shot row:
#    sn = df.where(df['Shot'] == 190624005).dropna()
#    df.dropna()[[df.columns[3]]].plot.hist(bins=100)


def get_average_vacuum():
    pass


def update_energy_cal():
    """
    Update the regression used to create a calibration factor for converting the photodiode response to an energy measurement
    Parameters
    ----------
    (none)

    Returns
    -------
    """

    
    # 

    LB = get_TS_logbook()
    energy_measured = LB.Energy.dropna()
    inds = energy_measured.index
    shots = LB.Shot.dropna()[inds].to_numpy().reshape(-1,1)
    energy_integrated = pd.Series()
    for ss in shots:
        hitConn.openTree("hitsi3", ss)
        try:
            flux_photodiode = np.array(hitConn.get("\\TS_RUBY"))
            flux_photodiode_t = np.array(hitConn.get("DIM_OF(\\TS_RUBY)"))
        except:
            print("WARNING: Error reading photodiode data from shot", ss)
            pass

        flux_baseline = np.mean(flux_photodiode[0:np.int(np.around(np.size(flux_photodiode,0)/6))])
#        flux_baseline = np.mean(flux_photodiode[-np.int(np.around(np.size(flux_photodiode,0)/8)):])
#        flux_baseline = 0
        flux_photodiode = flux_photodiode - flux_baseline

        energy_integrated = energy_integrated.append(pd.Series([np.trapz(flux_photodiode, flux_photodiode_t)]), ignore_index=True)

#    A = np.transpose(np.array([energy_measured, (np.ones_like(energy_measured))]))
#    m, c = np.linalg.lstsq(A, energy_integrated,rcond=None)[0]
    energy_integrated = energy_integrated.to_numpy().reshape(-1,1) 
    energy_measured = energy_measured.to_numpy().reshape(-1,1) 

    # Model initialization
    regression_model = LinearRegression()
    
    # Fit the data(train the model)
    regression_model.fit(energy_measured, energy_integrated)
    
    # Predict
    E_predicted = regression_model.predict(energy_measured)
    
    # model evaluation
    rmse = mean_squared_error(energy_integrated, E_predicted)
    r2 = r2_score(energy_integrated, E_predicted)
    m = regression_model.coef_
    c = regression_model.intercept_
    
    if(PLOTS_ON == 1):
        # printing values
        print('Slope:', m)
        print('Intercept:', c)
        print('Root mean squared error: ', rmse)
        print('R2 score: ', r2)
    
        
        fig1, ax1 = plt.subplots()
        ax1.set_title("Linear regression")
        ax1.set_xlabel(r"$E_{meter} [J]$")
        ax1.set_ylabel(r"$E_{photodiode} [J]$")    
        ax1.plot(energy_measured, energy_integrated/m, 'o', label='Original data', markersize=10)
        ax1.plot(np.arange(0,10), regression_model.predict(np.arange(0,10).reshape(-1,1))/m, label='Fitted line')
        ax1.plot(np.arange(0,10), np.arange(0,10), color='k', ls='--', linewidth=0.5)
        ax1.legend()
        ax1.grid(ls='--')
        
        print(1/m)
        
        fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figures
        ax2.plot(flux_photodiode_t,flux_photodiode)
        ax3.scatter(energy_measured, energy_integrated/m)
    
update_energy_cal()
    
    
    
    
    