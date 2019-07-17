#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:40:22 2019

@author: everson
"""

# imports
import matplotlib.pyplot as plt   # plotting
import datetime
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
style = 'Freq'  # 'Bayes' for Bayesian analysis or 'Freq' for frequentist/ratio evaluation method

def main(PLOTS_ON,style):
    
    # The logbook is the ledger on which any function used in the analysis code
    #   will write down its relevant parameters
    LB = get_TS_logbook()
    

    # calculate and store the laser energy
    
    # calculate and store the electron energy distribution function, including temperature, density, and electron drift velocity        
    pass

class Shot(object):
    def __init__(self, sN):
        self.shotNum = sN
        pass

    def logInfo(self, parameter_list):
        pass

    def preconditionData(self, parameter_list):
        pass
    

def get_TS_logbook():
    """
    Pull in the logbook information from the TS_Logbook Google sheet, parse it, and return a version of it for use in the TS analysis.
    ----------
    (none)

    Returns
    -------
    """
    # get the logbook spreadsheet from google sheets and read it in to a pandas dataframe
    raw_google_csv = requests.get('https://docs.google.com/spreadsheet/ccc?key=1yF7RMpYl_KvZPoEwYiH1D_vVrasaTGLuPvdBoYRRM84&output=csv')
    data = raw_google_csv.content
    raw_TS_log = pd.read_csv(BytesIO(data), usecols=np.arange(1, 10))

    # drop Nan's from gaps between run days and other non-shots
    raw_TS_log = raw_TS_log.dropna(subset=['Shot', 'Plasma Species']).reset_index()

    # identify and get rid of duplicates if necessary
    raw_TS_log_nodupes = raw_TS_log.drop_duplicates(subset=['Shot'])

    if(raw_TS_log.shape[0] != raw_TS_log_nodupes.shape[0]):
        print("*****WARNING! Duplicates while parsing TS logbook shots:*****")
        duplicates = raw_TS_log.loc[raw_TS_log.duplicated(subset=['Shot'])]['Shot'].astype(int)
        print(duplicates.to_csv(sep='\t', index=False))
        print("...excluding duplicates from parsed logbook.")

    raw_TS_log = raw_TS_log_nodupes
    
    # clean up and start the dataframe to return
    generated_TS_log = raw_TS_log[['Shot','TS_TRIG (ms)','Energy','Plasma Species','Notes']]
    generated_TS_log = generated_TS_log.rename(index=str, columns={'Plasma Species':'Fuel'})
    generated_TS_log = generated_TS_log.rename(index=str, columns={'TS_TRIG (ms)':'TS_TRIG'})
    generated_TS_log['Shot'] = generated_TS_log.Shot.astype('int64')
    generated_TS_log['Fuel'].replace({'-':'V'},inplace=True)
    
    generated_TS_log.reset_index(inplace=True,drop=True)
    
    # parse the laser voltage column
    voltage_parse_dataframe = pd.DataFrame()
    
    for laser_power_string in raw_TS_log['Laser Power (OSC-AMP12-AMP3)']:
        voltages = np.array(laser_power_string.rsplit(sep='-')).astype(np.double)*10
        power_oneshot_dataframe = pd.DataFrame(data=[voltages],columns=['V_osc','V_amps12','V_amp3'])
        
        voltage_parse_dataframe = pd.concat([voltage_parse_dataframe,power_oneshot_dataframe],ignore_index=True,sort=False)
    
    voltage_parse_dataframe.reset_index(inplace=True,drop=True)
    generated_TS_log = pd.concat([generated_TS_log,voltage_parse_dataframe],axis=1)
        
    # parse the active polychromator channels column  
    poly_parse_dataframe = pd.DataFrame()
    
    for active_polys_string in raw_TS_log['Active Channels (polychromator no. in tens place)']:
        polys_channels_list = active_polys_string.rsplit(sep=',')
        total_channels = len(polys_channels_list)
        polys_oneshot_dataframe = pd.DataFrame(data=np.ones([1,total_channels]),columns=polys_channels_list)
        
        poly_parse_dataframe = pd.concat([poly_parse_dataframe,polys_oneshot_dataframe],ignore_index=True,sort=False)
    
    poly_parse_dataframe.fillna(value=0, inplace=True)
    poly_parse_dataframe.rename(columns=lambda x: 'poly'+x[0]+'_'+x[1], inplace=True)
    poly_parse_dataframe.reset_index(inplace=True,drop=True)
    generated_TS_log = pd.concat([generated_TS_log,poly_parse_dataframe],axis=1)
    
    # add the relevant calibration file for this shot
    
    
    
    # calculate and store the geometry data for the shot using the lab jack height and mount positions string
        
    return(generated_TS_log)

#    df[[df.columns[3]]].plot.hist(bins=50)
#    particular shot row:
#    sn = df.where(df['Shot'] == 190624005).dropna()
#    df.dropna()[[df.columns[3]]].plot.hist(bins=100)
    
    
def write_TS_logbook(LB):
    """
    Write a logbook to csv and possibly to the analysis3 tree (later).
    
    Parameters
    ----------
    LB: Logbook in the style of the Pandas dataframe object

    Returns
    -------
    """
    # save the file to a csv
    
    filestr = 'TS_logbook_autogen_' + datetime.date.today().isoformat() + '.csv'
    
    LB.to_csv(filestr)
    
    
def get_average_vacuum():
    pass


def update_energy_cal(LB):
    """
    Update the regression used to create a calibration factor for converting the photodiode response to an energy measurement
    
    Parameters
    ----------
    LB: Logbook in the style of the Pandas dataframe object

    Returns
    -------
    """

    
    # 

#    LB = get_TS_logbook()
    energy_measured = LB.Energy[LB.Fuel=='ETEST']
    energy_integrated = pd.Series()
    for shot in LB.Shot[LB.Fuel=='ETEST']:
        hitConn.openTree("hitsi3", shot)
        try:
            flux_photodiode = np.array(hitConn.get("\\TS_RUBY"))
            flux_photodiode_t = np.array(hitConn.get("DIM_OF(\\TS_RUBY)"))
        except:
            print("WARNING: Error reading photodiode data from shot", shot)
            pass

        flux_baseline = np.mean(flux_photodiode[0:np.int(np.around(np.size(flux_photodiode,0)/6))])
#        flux_baseline = np.mean(flux_photodiode[-np.int(np.around(np.size(flux_photodiode,0)/8)):])
#        flux_baseline = 0
        flux_photodiode = flux_photodiode - flux_baseline

        energy_integrated = energy_integrated.append(pd.Series([np.trapz(flux_photodiode, flux_photodiode_t)]), ignore_index=True)
        
    if(style == 'Freq'):

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
    elif(style == 'Bayes'):
        pass
    else:
        print('****Pick a style that"s either Bayes or Freq****')
    
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
        ax1.plot(energy_measured, energy_integrated/m, 'o', label='Original data', markersize=2)
        ax1.plot(np.arange(0,10), regression_model.predict(np.arange(0,10).reshape(-1,1))/m, label='Fitted line')
        ax1.plot(np.arange(0,10), np.arange(0,10), color='k', ls='--', linewidth=0.5)
        ax1.legend()
        ax1.grid(ls='--')
        
        
        print(1/m)
        
#        fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figures
#        ax2.plot(flux_photodiode_t,flux_photodiode)
#        ax3.scatter(energy_measured, energy_integrated/m)