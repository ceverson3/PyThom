#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:40:22 2019

@author: everson
"""

# imports
import matplotlib.pyplot as plt   # plotting
import matplotlib.ticker as ticker
import datetime
import scipy.signal as signal     # for signal operations like detrending, etc.
import statsmodels.api as sm
import scipy.stats as st
import re                         # for string with wildcard character matching
import numpy as np                # math
from scipy.io import loadmat      # for loading matlab files
import pandas as pd               # large 2D dataframe operations
import seaborn as sns             # colors!
from io import BytesIO            # reading the TS logbook
import requests                   # grabbing the TS logbook from google sheets
from MDSplus import Connection, Tree, Data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pymc3 as pm
import theano
import theano.tensor as tt
from scipy.optimize import approx_fprime
import theano.tests.unittest_tools

PLOTS_ON = 1
FORCE_NEW_NODES = 0
CHECK_RAW = 0
photodiode_baseline_record_fraction = 0.15  # the fraction of the ruby photodiode record to take as the baseline
num_vacuum_shots = 4
FAST_GAIN = 1
E_CHARGE = 1.602176634*10**(-19)  # [coulombs]
SIGMA_TS = 6.63404498777644e-29  # [m^2]
h_PLANCK = 6.62607015e-34  # [J*s]
ELECTRON_MASS = 9.1094*10**(-31)  # [kg]
# K_BOLTZMANN = 8.653760519135803e-05  # [eV/K]
K_BOLTZMANN = 1.3864852*10**(-23)  # [J/K]
R_FEEDBACK = 50000  # feedback resistance, [ohms]
C_SPEED = 3e8  # [m/s]
RUBY_WL = 694.3e-9  # [m]
style = 'BDA'  # 'Bayes' for Bayesian analysis or 'Ratio' for ratio evaluation method
sns.set()   # Set the plotting theme


def plot_drifting_max(t_e, v_d):

# 	l_domain = get_dim('CAL_5_CH_FG_POLY_3_T_1', Tree('analysis3', -1))
    l_domain = np.arange(670e-9, 718.6e-9, 1e-12)
    angle_scat = np.pi/2
    beta = C_SPEED*np.sqrt(ELECTRON_MASS)/(2*RUBY_WL*np.sin(angle_scat/2)*np.sqrt(2*E_CHARGE))
    h = np.exp(-((beta)*(l_domain - RUBY_WL) - v_d*np.sqrt(ELECTRON_MASS/(2*E_CHARGE)))**2/t_e)/np.sqrt(t_e)

    print(l_domain[np.where(h == max(h))[0][0]])

    plt.plot(l_domain, h)
    plt.show()


def look_at(shot=None):
#   190501012
    g_res_dpi = 600
    sns.set_style("ticks")
    data = pd.read_csv('TS_analysis_logbook.csv')

    if shot is None:
        analyzed_shots = [s for s in data['Shot'] if not np.isnan(data.loc[data.Shot == s, 'POLY_3_T_e_fixed'].array[0])]
        shot = np.int(np.random.choice(analyzed_shots))
    print(shot)


    analysis_tree = Tree('analysis3', shot)

    i_tor_y = get_data('i_tor_spaavg', analysis_tree)
    i_tor_t = get_dim('i_tor_spaavg', analysis_tree)
    ind_start = np.where(i_tor_t > 0.00025)[0][0]
    ind_stop = np.where(i_tor_t > 0.004)[0][0]
    i_tor_t = i_tor_t[ind_start:ind_stop]
    i_tor_y = i_tor_y[ind_start:ind_stop]
    i_tor = np.int(data.loc[data.Shot == shot, 'i_tor_max'].array[0]/1000)
    fire_time = data.loc[data.Shot == shot, 'FIRE_TIME'].array[0]
    fuel = data.loc[data.Shot == shot, 'Fuel'].array[0]
    # ft = np.where(i_tor_t >= fire_time)[0][0]
    # data.loc[data.Shot == shot].POLY_5_T_e_drift.hist(alpha=0.3)
    t_e_fixed_kde = {}
    t_e_drift_kde = {}
    v_d_drift_kde = {}

    t_e_fixed = {}
    t_e_fixed_lpd = {}
    t_e_fixed_hpd = {}

    t_e_drift = {}
    t_e_drift_lpd = {}
    t_e_drift_hpd = {}

    v_d_drift = {}
    v_d_drift_lpd = {}
    v_d_drift_hpd = {}

    poly_r = {}

    fig = plt.figure(figsize=(7, 8))
    # fig = plt.figure()
    grid = plt.GridSpec(3, 1, wspace=0.3, hspace=0.3)
    ax1 = fig.add_subplot(grid[1:2, 0])
    ax2 = fig.add_subplot(grid[2:, 0])
    # ax2 = ax1.twinx()
    ax3 = fig.add_subplot(grid[0, 0])

    for pp in np.arange(3,6):
        try:
            t_e_fixed_kde[pp] = get_data('POLY_' + np.str(pp) + '_T_E_FIXED', analysis_tree)
            t_e_drift_kde[pp] = get_data('POLY_' + np.str(pp) + '_T_E_DRIFT', analysis_tree)
            v_d_drift_kde[pp] = get_data('POLY_' + np.str(pp) + '_V_D_DRIFT', analysis_tree)
        except Exception as ex:
            print(ex.msgnam)
            t_e_fixed_kde[pp] = np.NaN
            t_e_drift_kde[pp] = np.NaN
            v_d_drift_kde[pp] = np.NaN

        t_e_fixed[pp] = data.loc[data.Shot == shot]['POLY_'+ np.str(pp) +'_T_e_fixed'].array[0]
        t_e_fixed_lpd[pp] = data.loc[data.Shot == shot]['POLY_'+ np.str(pp) +'_T_e_fixed_hpd95low'].array[0]
        t_e_fixed_hpd[pp] = data.loc[data.Shot == shot]['POLY_'+ np.str(pp) +'_T_e_fixed_hpd95high'].array[0]

        t_e_drift[pp] = data.loc[data.Shot == shot]['POLY_'+ np.str(pp) +'_T_e_drift'].array[0]
        t_e_drift_lpd[pp] = data.loc[data.Shot == shot]['POLY_'+ np.str(pp) +'_T_e_drift_hpd95low'].array[0]
        t_e_drift_hpd[pp] = data.loc[data.Shot == shot]['POLY_'+ np.str(pp) +'_T_e_drift_hpd95high'].array[0]

        v_d_drift[pp] = data.loc[data.Shot == shot]['POLY_'+ np.str(pp) +'_v_d_drift'].array[0]/1000
        v_d_drift_lpd[pp] = data.loc[data.Shot == shot]['POLY_'+ np.str(pp) +'_v_d_drift_hpd95low'].array[0]/1000
        v_d_drift_hpd[pp] = data.loc[data.Shot == shot]['POLY_'+ np.str(pp) +'_v_d_drift_hpd95high'].array[0]/1000

        poly_r[pp] = data.loc[data.Shot == shot]['radius_poly_' + np.str(pp)].array[0]

# TODO add in fuel species
    for pp in np.arange(3,6):
        ax1.errorbar(poly_r[pp], t_e_fixed[pp], np.array([[t_e_fixed[pp] - t_e_fixed_lpd[pp]], [t_e_fixed_hpd[pp] - t_e_fixed[pp]]]), mfc='white', fmt='bo', lw=1, capsize=3)
        ax1.errorbar(poly_r[pp], t_e_drift[pp], np.array([[t_e_drift[pp] - t_e_drift_lpd[pp]], [t_e_drift_hpd[pp] - t_e_drift[pp]]]), mfc='white', fmt='bd', lw=1, capsize=3)
        ax2.errorbar(poly_r[pp], v_d_drift[pp], np.array([[v_d_drift[pp] - v_d_drift_lpd[pp]], [v_d_drift_hpd[pp] - v_d_drift[pp]]]), fmt='gv', lw=1, capsize=3)

    ax3.vlines(fire_time, 0, i_tor, linestyles='dashed')
    ax3.plot(i_tor_t, i_tor_y/1000, linewidth=1)
    ax3.legend([r'$I_{tor}$ [kA]', 'TS meas. time'])
    ax1.set_xlabel('Maj. Radius [m]')
    ax2.set_xlabel('Maj. Radius [m]')
    ax3.set_xlabel('[s]')
    ax1.set_ylabel('Temp. [eV]', color='b')
    ax3.set_ylabel(r'$I_{tor}$ [kA]', color='b')
    ax2.set_ylabel('Velocity [km/s]', color='g')
    ax1.grid(linestyle=':', linewidth=0.5)
    ax2.grid(linestyle=':', linewidth=0.5)
    # ax1.legend([r'$T_e$  ($v_d \equiv 0$)', r'$T_e$  (drift allowed)'], loc='lower left')
    # ax2.legend([r'$v_d$'], loc='upper right')
    ax1.legend([r'$T_e$  ($v_d \equiv 0$)', r'$T_e$  (drift allowed)'])
    ax2.legend([r'$v_d$'])
    ax1.set_xlim(0.2, 0.4)
    ax2.set_xlim(0.2, 0.4)
    ax1.set_ylim(0,)
    plt.title('Shot ' +  np.str(shot) + ', Fuel: ' + fuel)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.08)
    fig.savefig('figures/' + np.str(shot) + 'profiles.svg', format='svg', dpi=g_res_dpi)
    plt.show()

    # another plot (histograms)
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle('Shot #' + np.str(shot) + ' Posterior Distributions', fontsize=16)
    grid = plt.GridSpec(100, 1, wspace=0.4, hspace=0.01)
    ax1 = fig.add_subplot(grid[0:25, 0])
    ax2 = fig.add_subplot(grid[41:66, 0])
    ax3 = fig.add_subplot(grid[75:100, 0])

# TODO add in fuel species
    for pp in np.arange(3,6):
        sns.kdeplot(t_e_fixed_kde[pp], ax=ax1, label=np.str(np.int(poly_r[pp]*100)) + ' cm')
        sns.kdeplot(t_e_drift_kde[pp], ax=ax2, label=np.str(np.int(poly_r[pp]*100)) + ' cm')
        sns.kdeplot(v_d_drift_kde[pp], ax=ax3, label=np.str(np.int(poly_r[pp]*100)) + ' cm')

    # ax3.vlines(fire_time, 0, i_tor, linestyles='dashed')
    # ax3.plot(i_tor_t, i_tor_y/1000, linewidth=1)
    ax1.set_xlim(1.5,15)
    ax1.set_title('Fixed Maxwellian model', fontsize=14)
    # ax1.legend([np.str(np.int(r*100)) + ' cm' for r in list(poly_r.values())])
    ax1.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax1.set_ylabel('P (unnormalized)', fontsize=10)
    
    ax2.set_xlim(1.5,15)
    ax2.set_title('Drifting Maxwellian model', fontsize=14)
    # ax2.legend([np.str(np.int(r*100)) + ' cm' for r in list(poly_r.values())])
    ax2.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax2.set_ylabel('P (unnormalized)', fontsize=10)
                   
    # ax3.legend([np.str(np.int(r*100)) + ' cm' for r in list(poly_r.values())])
    ax3.set_xlabel(r'$v_d$ [m/s]', fontsize=10)
    ax3.set_ylabel('P (unnormalized)', fontsize=10)
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    ax3.set_xlim(-450000,450000)
    fig.savefig('figures/posteriors_' + np.str(shot) + '.svg', format='svg', dpi=g_res_dpi)

    plt.show()


def aggregate(shot=None):
    data = pd.read_csv('TS_analysis_logbook.csv')
    g_res_dpi = 600
    tbins = np.arange(2,30,0.5)
    vbins = 50
    fuel = 'D'
    # 190226004
    radii = np.unique(pd.concat([data.radius_poly_3, data.radius_poly_4, data.radius_poly_5]).dropna())    
    radii = np.r_[radii[0:2], radii[-1]]
    
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle("All Thomson scattering measurements", fontsize=16)
    grid = plt.GridSpec(100, 1, wspace=0.4, hspace=0.01)
    ax1 = fig.add_subplot(grid[0:25, 0])
    ax2 = fig.add_subplot(grid[41:66, 0])
    ax3 = fig.add_subplot(grid[75:100, 0])

    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.radius_poly_3 == radii[ii])].POLY_3_T_e_fixed,
                           data.loc[(data.radius_poly_4 == radii[ii])].POLY_4_T_e_fixed,
                           data.loc[(data.radius_poly_5 == radii[ii])].POLY_5_T_e_fixed])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax1.set_xlim(1.5,30)
    ax1.set_title('Fixed Maxwellian model', fontsize=14)
    ax1.legend(leg)
    ax1.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax1.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.radius_poly_3 == radii[ii])].POLY_3_T_e_drift,
                           data.loc[(data.radius_poly_4 == radii[ii])].POLY_4_T_e_drift,
                           data.loc[(data.radius_poly_5 == radii[ii])].POLY_5_T_e_drift])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax2, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax2, color=sns.color_palette()[ii])  
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax2.set_xlim(1.5,30)
    ax2.set_title('Drifting Maxwellian model', fontsize=14)
    ax2.legend(leg)
    ax2.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax2.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.radius_poly_3 == radii[ii])].POLY_3_v_d_drift,
                           data.loc[(data.radius_poly_4 == radii[ii])].POLY_4_v_d_drift,
                           data.loc[(data.radius_poly_5 == radii[ii])].POLY_5_v_d_drift])
        if vbins is None:
            rhist.hist(alpha=0.5, ax=ax3, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=vbins, alpha=0.5, ax=ax3, color=sns.color_palette()[ii])  
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax3.legend(leg)
    ax3.set_xlabel(r'$v_d$ [m/s]', fontsize=10)
    ax3.set_ylabel('# of shots', fontsize=10)
    ax3.set_xlim(-450000,450000)
    fig.savefig('figures/aggregate_allshots.svg', format='svg', dpi=g_res_dpi)

    
    # new figure
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle("Deuterium plasma", fontsize=16)
    grid = plt.GridSpec(100, 1, wspace=0.4, hspace=0.01)
    ax1 = fig.add_subplot(grid[0:25, 0])
    ax2 = fig.add_subplot(grid[41:66, 0])
    ax3 = fig.add_subplot(grid[75:100, 0])

    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.Fuel == fuel) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_fixed,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_fixed,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_fixed])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax1, color=sns.color_palette()[ii])  
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax1.set_xlim(1.5,30)
    ax1.set_title('Fixed Maxwellian model', fontsize=14)
    ax1.legend(leg)
    ax1.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax1.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.Fuel == fuel) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_drift])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax2, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax2, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax2.set_xlim(1.5,30)
    ax2.set_title('Drifting Maxwellian model', fontsize=14)
    ax2.legend(leg)
    ax2.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax2.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.Fuel == fuel) & (data.radius_poly_3 == radii[ii])].POLY_3_v_d_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_4 == radii[ii])].POLY_4_v_d_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_5 == radii[ii])].POLY_5_v_d_drift])
        if vbins is None:
            rhist.hist(alpha=0.5, ax=ax3, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=vbins, alpha=0.5, ax=ax3, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax3.legend(leg)
    ax3.set_xlabel(r'$v_d$ [m/s]', fontsize=10)
    ax3.set_ylabel('# of shots', fontsize=10)
    ax3.set_xlim(-450000,450000)
    fig.savefig('figures/aggregate_D_only.svg', format='svg', dpi=g_res_dpi)

    # new figure
    fuel = 'D/He'
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle("Deuterium/Helium plasma", fontsize=16)
    grid = plt.GridSpec(100, 1, wspace=0.4, hspace=0.01)
    ax1 = fig.add_subplot(grid[0:25, 0])
    ax2 = fig.add_subplot(grid[41:66, 0])
    ax3 = fig.add_subplot(grid[75:100, 0])

    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.Fuel == fuel) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_fixed,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_fixed,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_fixed])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax1.set_xlim(1.5,30)
    ax1.set_title('Fixed Maxwellian model', fontsize=14)
    ax1.legend(leg)
    ax1.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax1.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.Fuel == fuel) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_drift])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax2, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax2, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax2.set_xlim(1.5,30)
    ax2.set_title('Drifting Maxwellian model', fontsize=14)
    ax2.legend(leg)
    ax2.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax2.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.Fuel == fuel) & (data.radius_poly_3 == radii[ii])].POLY_3_v_d_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_4 == radii[ii])].POLY_4_v_d_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_5 == radii[ii])].POLY_5_v_d_drift])
        if vbins is None:
            rhist.hist(alpha=0.5, ax=ax3, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=vbins, alpha=0.5, ax=ax3, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax3.legend(leg)
    ax3.set_xlabel(r'$v_d$ [m/s]', fontsize=10)
    ax3.set_ylabel('# of shots', fontsize=10)
    ax3.set_xlim(-450000,450000)
    fig.savefig('figures/aggregate_DHe_only.svg', format='svg', dpi=g_res_dpi)


    # new figure
    fuel = 'He'
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle("Helium plasma", fontsize=16)
    grid = plt.GridSpec(100, 1, wspace=0.4, hspace=0.01)
    ax1 = fig.add_subplot(grid[0:25, 0])
    ax2 = fig.add_subplot(grid[41:66, 0])
    ax3 = fig.add_subplot(grid[75:100, 0])

    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.Fuel == fuel) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_fixed,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_fixed,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_fixed])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax1, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax1.set_xlim(1.5,30)
    ax1.set_title('Fixed Maxwellian model', fontsize=14)
    ax1.legend(leg)
    ax1.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax1.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.Fuel == fuel) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_drift])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax2, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax2, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax2.set_xlim(1.5,30)
    ax2.set_title('Drifting Maxwellian model', fontsize=14)
    ax2.legend(leg)
    ax2.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax2.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.Fuel == fuel) & (data.radius_poly_3 == radii[ii])].POLY_3_v_d_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_4 == radii[ii])].POLY_4_v_d_drift,
                   data.loc[(data.Fuel == fuel) & (data.radius_poly_5 == radii[ii])].POLY_5_v_d_drift])
        if vbins is None:
            rhist.hist(alpha=0.5, ax=ax3, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=vbins, alpha=0.5, ax=ax3, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax3.legend(leg)
    ax3.set_xlabel(r'$v_d$ [m/s]', fontsize=10)
    ax3.set_ylabel('# of shots', fontsize=10)
    ax3.set_xlim(-450000,450000)
    fig.savefig('figures/aggregate_He_only.svg', format='svg', dpi=g_res_dpi)

    
    # new figure
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle('Positive toroidal current', fontsize=16)
    grid = plt.GridSpec(100, 1, wspace=0.4, hspace=0.01)
    ax1 = fig.add_subplot(grid[0:25, 0])
    ax2 = fig.add_subplot(grid[41:66, 0])
    ax3 = fig.add_subplot(grid[75:100, 0])

    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.i_tor_max > 0) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_fixed,
                           data.loc[(data.i_tor_max > 0) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_fixed,
                           data.loc[(data.i_tor_max > 0) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_fixed])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax1, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax1.set_xlim(1.5,30)
    ax1.set_title('Fixed Maxwellian model', fontsize=14)
    ax1.legend(leg)
    ax1.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax1.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.i_tor_max > 0) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_drift,
                           data.loc[(data.i_tor_max > 0) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_drift,
                           data.loc[(data.i_tor_max > 0) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_drift])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax2, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax2, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax2.set_xlim(1.5,30)
    ax2.set_title('Drifting Maxwellian model', fontsize=14)
    ax2.legend(leg)
    ax2.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax2.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.i_tor_max > 0) & (data.radius_poly_3 == radii[ii])].POLY_3_v_d_drift,
                           data.loc[(data.i_tor_max > 0) & (data.radius_poly_4 == radii[ii])].POLY_4_v_d_drift,
                           data.loc[(data.i_tor_max > 0) & (data.radius_poly_5 == radii[ii])].POLY_5_v_d_drift])
        if vbins is None:
            rhist.hist(alpha=0.5, ax=ax3, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=vbins, alpha=0.5, ax=ax3, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax3.legend(leg)
    ax3.set_xlabel(r'$v_d$ [m/s]', fontsize=10)
    ax3.set_ylabel('# of shots', fontsize=10)
    ax3.set_xlim(-450000,450000)
    fig.savefig('figures/aggregate_positive_itor_only.svg', format='svg', dpi=g_res_dpi)

 
    # new figure
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle('Negative toroidal current', fontsize=16)
    grid = plt.GridSpec(100, 1, wspace=0.4, hspace=0.01)
    ax1 = fig.add_subplot(grid[0:25, 0])
    ax2 = fig.add_subplot(grid[41:66, 0])
    ax3 = fig.add_subplot(grid[75:100, 0])

    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.i_tor_max < 0) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_fixed,
                           data.loc[(data.i_tor_max < 0) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_fixed,
                           data.loc[(data.i_tor_max < 0) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_fixed])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax1, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax1.set_xlim(1.5,30)
    ax1.set_title('Fixed Maxwellian model', fontsize=14)
    ax1.legend(leg)
    ax1.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax1.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.i_tor_max < 0) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_drift,
                           data.loc[(data.i_tor_max < 0) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_drift,
                           data.loc[(data.i_tor_max < 0) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_drift])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax2, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax2, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax2.set_xlim(1.5,30)
    ax2.set_title('Drifting Maxwellian model', fontsize=14)
    ax2.legend(leg)
    ax2.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax2.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.i_tor_max < 0) & (data.radius_poly_3 == radii[ii])].POLY_3_v_d_drift,
                           data.loc[(data.i_tor_max < 0) & (data.radius_poly_4 == radii[ii])].POLY_4_v_d_drift,
                           data.loc[(data.i_tor_max < 0) & (data.radius_poly_5 == radii[ii])].POLY_5_v_d_drift])
        if vbins is None:
            rhist.hist(alpha=0.5, ax=ax3, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=vbins, alpha=0.5, ax=ax3, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax3.legend(leg)
    ax3.set_xlabel(r'$v_d$ [m/s]', fontsize=10)
    ax3.set_ylabel('# of shots', fontsize=10)
    ax3.set_xlim(-450000,450000)
    fig.savefig('figures/aggregate_negative_itor_only.svg', format='svg', dpi=g_res_dpi)


    # new figure
    time_line = 0.002
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle(r'Spheromak decay ($t_{TS}>$' + np.str(time_line) + ' s)', fontsize=16)
    grid = plt.GridSpec(100, 1, wspace=0.4, hspace=0.01)
    ax1 = fig.add_subplot(grid[0:25, 0])
    ax2 = fig.add_subplot(grid[41:66, 0])
    ax3 = fig.add_subplot(grid[75:100, 0])

    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.FIRE_TIME > time_line) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_fixed,
                           data.loc[(data.FIRE_TIME > time_line) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_fixed,
                           data.loc[(data.FIRE_TIME > time_line) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_fixed])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax1, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax1.set_xlim(1.5,30)
    ax1.set_title('Fixed Maxwellian model', fontsize=14)
    ax1.legend(leg)
    ax1.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax1.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.FIRE_TIME > time_line) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_drift,
                           data.loc[(data.FIRE_TIME > time_line) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_drift,
                           data.loc[(data.FIRE_TIME > time_line) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_drift])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax2, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax2, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax2.set_xlim(1.5,30)
    ax2.set_title('Drifting Maxwellian model', fontsize=14)
    ax2.legend(leg)
    ax2.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax2.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.FIRE_TIME > time_line) & (data.radius_poly_3 == radii[ii])].POLY_3_v_d_drift,
                           data.loc[(data.FIRE_TIME > time_line) & (data.radius_poly_4 == radii[ii])].POLY_4_v_d_drift,
                           data.loc[(data.FIRE_TIME > time_line) & (data.radius_poly_5 == radii[ii])].POLY_5_v_d_drift])
        if vbins is None:
            rhist.hist(alpha=0.5, ax=ax3, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=vbins, alpha=0.5, ax=ax3, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax3.legend(leg)
    ax3.set_xlabel(r'$v_d$ [m/s]', fontsize=10)
    ax3.set_ylabel('# of shots', fontsize=10)
    ax3.set_xlim(-450000,450000)
    fig.savefig('figures/aggregate_decay_period_only.svg', format='svg', dpi=g_res_dpi)

    
    # new figure
    fig = plt.figure(figsize=(7, 8))
    fig.suptitle(r'Spheromak sustainment ($t_{TS}<$' + np.str(time_line) + ' s)', fontsize=16)
    grid = plt.GridSpec(100, 1, wspace=0.4, hspace=0.01)
    ax1 = fig.add_subplot(grid[0:25, 0])
    ax2 = fig.add_subplot(grid[41:66, 0])
    ax3 = fig.add_subplot(grid[75:100, 0])

    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.FIRE_TIME < time_line) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_fixed,
                           data.loc[(data.FIRE_TIME < time_line) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_fixed,
                           data.loc[(data.FIRE_TIME < time_line) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_fixed])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax1, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax1, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax1.set_xlim(1.5,30)
    ax1.set_title('Fixed Maxwellian model', fontsize=14)
    ax1.legend(leg)
    ax1.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax1.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.FIRE_TIME < time_line) & (data.radius_poly_3 == radii[ii])].POLY_3_T_e_drift,
                           data.loc[(data.FIRE_TIME < time_line) & (data.radius_poly_4 == radii[ii])].POLY_4_T_e_drift,
                           data.loc[(data.FIRE_TIME < time_line) & (data.radius_poly_5 == radii[ii])].POLY_5_T_e_drift])
        if tbins is None:
            rhist.hist(alpha=0.5, ax=ax2, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=tbins, alpha=0.5, ax=ax2, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax2.set_xlim(1.5,30)
    ax2.set_title('Drifting Maxwellian model', fontsize=14)
    ax2.legend(leg)
    ax2.set_xlabel(r'$T_e$ [eV]', fontsize=10)
    ax2.set_ylabel('# of shots', fontsize=10)
    
    leg = list([])
    for ii in np.arange(0,len(radii)):
        rhist = pd.concat([data.loc[(data.FIRE_TIME < time_line) & (data.radius_poly_3 == radii[ii])].POLY_3_v_d_drift,
                           data.loc[(data.FIRE_TIME < time_line) & (data.radius_poly_4 == radii[ii])].POLY_4_v_d_drift,
                           data.loc[(data.FIRE_TIME < time_line) & (data.radius_poly_5 == radii[ii])].POLY_5_v_d_drift])
        if vbins is None:
            rhist.hist(alpha=0.5, ax=ax3, color=sns.color_palette()[ii])    
        else:
            rhist.hist(bins=vbins, alpha=0.5, ax=ax3, color=sns.color_palette()[ii])
        leg.append(r'R = ' + np.str(np.int(100*radii[ii])) + ' cm (N = ' + np.str(len(rhist)) + ')')
    ax3.legend(leg)
    ax3.set_xlabel(r'$v_d$ [m/s]', fontsize=10)
    ax3.set_ylabel('# of shots', fontsize=10)    
    ax3.set_xlim(-450000,450000)
    fig.savefig('figures/aggregate_sustainment_period_only.svg', format='svg', dpi=g_res_dpi)


def clear_thomson_tree(shot=None):
    log_book = get_ts_logbook()
    shots_list = [shot for shot in log_book['Shot'] if (log_book.loc[log_book.Shot == shot, 'Fuel'].array[0] == 'D'
                                                               or log_book.loc[log_book.Shot == shot, 'Fuel'].array[0] == 'He'
                                                               or log_book.loc[log_book.Shot == shot, 'Fuel'].array[0] == 'D/He'
                                                               or log_book.loc[log_book.Shot == shot, 'Fuel'].array[0] == 'V')]
    if shot == None:
        return 1
        for shot in shots_list:
    
            tree = Tree('analysis3', shot, 'EDIT')
            try:
                tree.getNode('\\ANALYSIS3::TOP.THOMSON')
                print("Node found shot %d" % shot)
            except:
                continue
            tree.deleteNode('\\ANALYSIS3::TOP.THOMSON')
            print("Node deleted")
            #tree.quit()
            #print("Tree Quit")
            tree.write()
            print("Tree wrote")
            tree.close()
            print("Tree Closed")
    else:
        tree = Tree('analysis3', shot, 'EDIT')
        try:
            tree.getNode('\\ANALYSIS3::TOP.THOMSON')
            print("Node found shot %d" % shot)
        except:
            pass
        tree.deleteNode('\\ANALYSIS3::TOP.THOMSON')
        print("Node deleted")
        #tree.quit()
        #print("Tree Quit")
        tree.write()
        print("Tree wrote")
        tree.close()
        print("Tree Closed")


def analyze_plasma(start=None, stop=None):
    log_book = get_ts_logbook()
    # pshots = [s for s in LB.loc[LB.Fuel == 'D/He' or LB.Fuel == 'D' or log_book.Fuel == 'He', 'Shot'] = s]
    pshots_list = [shot for shot in log_book['Shot'] if (log_book.loc[log_book.Shot == shot, 'Fuel'].array[0] == 'D'
                                                               or log_book.loc[log_book.Shot == shot, 'Fuel'].array[0] == 'He'
                                                               or log_book.loc[log_book.Shot == shot, 'Fuel'].array[0] == 'D/He')]

    if start is None:
        start = 190000000

    if stop is None:
        stop = pshots_list[-1]
    PATH = '/home/everson/Dropbox/PyThom/logs/'
    # PATH = '/Users/chris/Dropbox/PyThom/logs/'
    print('Starting full analysis...')
    for pshot in pshots_list:
        if pshot >= start and pshot <= stop:
            # analyze_shot(shot)
            # with open(PATH + 'PyThom.log', 'a') as logf:
            #     logf.write("Shot {0}: {1}\n".format(str(pshot), ' OK'))
            try:
                analyze_shot(pshot)
                with open(PATH + 'PyThom.log', 'a') as logf:
                    logf.write("Shot {0}: {1}\n".format(str(pshot), ' OK'))
            except Exception as ex:
                with open(PATH + 'PyThom.log', 'a') as logf:
                    logf.write("Shot {0}: error {1}, retrying...\n".format(str(pshot), str(ex)))
                    clear_thomson_tree(shot=pshot)
                    for vs in get_vacuum_shot_list(shot_number=pshot, log_book=log_book, number_vacuum_shots=4):
                        clear_thomson_tree(shot=vs)
                    try:
                        analyze_shot(pshot)
                        with open(PATH + 'PyThom.log', 'a') as logf:
                            logf.write("Shot {0}: {1}\n".format(str(pshot), ' OK'))
                    except Exception as ex:
                        with open(PATH + 'PyThom.log', 'a') as logf:
                            logf.write("Shot {0}: error {1}\n".format(str(pshot), str(ex)))

        else:
            pass



def analyze_shot(shot_number):
    """
    Analyze the Thomson data for shot_number
    ----------
    Does analysis in a frugal way that avoids duplicate math by writing
    any incremental results to the analysis3 tree

    Returns
    -------
    """


    # Load the analysis tree and the log book
    analysis_tree = Tree('analysis3', shot_number, 'EDIT')
    log_book = get_ts_logbook()

    # Get the vacuum shots to use for stray light background subtraction
    vacuum_shot_list = get_vacuum_shot_list(shot_number, log_book, num_vacuum_shots)
    # TODO write the vacuum shots to the logbook

    if FORCE_NEW_NODES == 1:
        clear_thomson_tree(shot_number)
        for shot in vacuum_shot_list:
            clear_thomson_tree(shot)

    # Make sure the right data is in the analysis tree
    build_shot(shot_number)

    # The logbook is the dataframe-style ledger for human-readable summary TS data
    # parameters used in the analysis code are found and written here
    log_book = get_ts_logbook(verbose=1)
    analysis_tree = Tree('analysis3', shot_number, 'EDIT')

    # TODO possibly remove next line
    # Get the photodiode energy trace
    # energy_photodiode_sig = get_data('ENERGY_PD', analysis_tree)
    # energy_photodiode_t = get_dim('ENERGY_PD', analysis_tree)

    # Get a list of the active polychromators by name
    regex = re.compile('POLY_._.')
    channel_list = [string for string in log_book.columns if re.match(regex, string) and log_book.loc[log_book.Shot == shot_number, string].array[0] != 0 and len(string) == 8]
    poly_list = np.unique([np.int(ch[-3]) for ch in channel_list])

    # TODO possibly remove next line
    # Get the geometric parameters for the measurements of shot_number
    # geometry = get_spot_geometry(log_book.loc[log_book.Shot == shot_number, 'Mount_Positions'].array[0], log_book.loc[log_book.Shot == shot_number, 'Jack_Height'].array[0])

    # Lists to define the scattering windows for each polychromator:
    scattering_window_start = list([0,0,0,0,0])
    scattering_window_end = list([-1,-1,-1,-1,-1])

    # Get the variance induced in scaling vacuum shots:
    plasma_laser_energy = get_data('ENERGY_BAYES', analysis_tree)
    var_ple = get_data('ENERGY_VAR', analysis_tree)
    var_vac = 0

    for vshot in vacuum_shot_list:
        vtree = Tree('analysis3', vshot)
        vac_laser_energy = get_data('ENERGY_BAYES', vtree)
        var_vle = get_data('ENERGY_VAR', vtree)
        var_vac = var_vac + ((var_ple)/(vac_laser_energy)**2) + var_vle*((plasma_laser_energy)**2/(vac_laser_energy)**4)
        vtree.close()

    # Switches for saturation detection:
    # poly_saturated_YN = list([0, 0, 0, 0, 0])

    # Analyze all active polychromator channels
    for channel_id in channel_list:

        # An integer to identify the polychromator number
        poly_num = np.int(channel_id[5])
        channel_num = np.int(channel_id[-1])

        # Get the raw polychromator data:
        raw_poly_channel_sig = get_data(channel_id + '_RAW', analysis_tree)
        raw_poly_channel_t = get_dim(channel_id + '_RAW', analysis_tree)
        
        # Get the vacuum shot:
        # Average the closest previous/subsequent comparable vacuum shots with their energies scaled, checking first
        # and putting the vacuum shot in the tree if necessary
        vacuum_avg = get_data(channel_id + '_VAC', analysis_tree)
        vacuum_avg_clean = signal.savgol_filter(signal.detrend(vacuum_avg), 101, 3)        
                
        # if poly_saturated_YN[poly_num - 1] == 0:
        #     if channel_num != 1:
        #         poly_saturated_YN[poly_num - 1] = check_for_saturation(raw_poly_channel_sig)
        if channel_num == 1:
            tree_write_safe(1, channel_id + '_SAT', tree=analysis_tree)
        else:
            tree_write_safe(check_for_saturation(raw_poly_channel_sig), channel_id + '_SAT', tree=analysis_tree)
        
        
        # Detrend and otherwise clean up the raw signals
        clean_poly_channel_sig = signal.savgol_filter(signal.detrend(raw_poly_channel_sig), 101, 3)
        clean_poly_channel_t = raw_poly_channel_t
        dt = np.mean(np.diff(raw_poly_channel_t))
        tree_write_safe(clean_poly_channel_sig, channel_id + '_SIG', dim=clean_poly_channel_t, tree=analysis_tree)




        # Find the signal voltage and the standard deviation signal voltage of the background
        if channel_num == 1:
            signal_voltage = np.max(clean_poly_channel_sig)
            scattering_window_start[poly_num - 1] = np.where(clean_poly_channel_sig > 0.2*signal_voltage)[0][0]
            scattering_window_end[poly_num - 1] = np.where(clean_poly_channel_sig > 0.2*signal_voltage)[0][-1]
        else:
            signal_voltage = np.max(clean_poly_channel_sig[scattering_window_start[poly_num - 1]:scattering_window_end[poly_num - 1]])

        background_voltage = np.std(np.r_[raw_poly_channel_sig[0:scattering_window_start[poly_num - 1] - 200], raw_poly_channel_sig[scattering_window_end[poly_num - 1] + 1000:]])

        # Get the calibration string based on the shot number:
        cal_string = get_cal_string(shot_number, channel_id)

        # Use the signal voltage to find the number of photoelectrons n_pe
        capacitance = get_data(cal_string.replace('TYPE?', 'C'), analysis_tree)
        pulse_width_time = (scattering_window_end[poly_num - 1] - scattering_window_start[poly_num - 1])*dt
        if(cal_string[9] != 'H'):
            amplifier_gain = 75
        else:
            amplifier_gain = 75/2

        n_pe = signal_voltage*pulse_width_time/(amplifier_gain*E_CHARGE*FAST_GAIN*R_FEEDBACK*(1 - np.exp(-pulse_width_time/(R_FEEDBACK*capacitance))))
        n_background = background_voltage*n_pe/signal_voltage

        # Find the same quantities for the stray (vacuum) light:
        stray_voltage = np.max(vacuum_avg_clean[scattering_window_start[poly_num - 1]:scattering_window_end[poly_num - 1]])

        background_stray_voltage = np.std(np.r_[vacuum_avg[0:scattering_window_start[poly_num - 1] - 200], vacuum_avg[scattering_window_end[poly_num - 1] + 1000:]])
        # Use the voltage to find the number of photoelectrons
        n_stray = stray_voltage*pulse_width_time/(amplifier_gain*E_CHARGE*FAST_GAIN*R_FEEDBACK*(1 - np.exp(-pulse_width_time/(R_FEEDBACK*capacitance))))
        n_background_stray = background_stray_voltage*n_stray/stray_voltage

        n_scat = n_pe - n_stray
        if n_scat < 0:
            n_scat = 0

        tree_write_safe(n_scat, channel_id + '_N_SCAT', tree=analysis_tree)

        # Calculate the variances of the measured data:
        var_scale = (n_stray**2)*var_vac
        var_bg = 4*n_background
        var_pe = 4*(n_pe) + var_bg
        var_stray = 4*(n_stray + n_background_stray) + var_scale
        var_scat = var_pe + var_stray
        tree_write_safe(var_scat, channel_id + '_VAR_SCAT', tree=analysis_tree)

        # Calculate the theoretical number of
        var_energy = get_data('ENERGY_VAR', analysis_tree)

        var_poly = var_scat
        if CHECK_RAW == 1:
            try:
                plt.figure()
                sig_for_plot = signal.detrend(raw_poly_channel_sig)
                plt.plot(np.arange(0, len(sig_for_plot)), sig_for_plot)
                bg1 = sig_for_plot[0:scattering_window_start[poly_num - 1] - 200]
                bg2 = sig_for_plot[scattering_window_end[poly_num - 1] + 1000:]
                plt.plot(np.arange(0,len(bg1)), bg1)
                plt.plot(np.arange(len(sig_for_plot) - len(bg2), len(sig_for_plot)), bg2)
                plt.plot(np.arange(0, len(sig_for_plot)), background_voltage*np.ones(len(sig_for_plot)))
                plt.show()
    
                # print(signal_voltage)
                # print(background_voltage)
            except:
                pass
    if style == 'REM':
        pass

    elif style == 'BDA':

        pmdf_fixed = {}
        trace_fixed = {}
        pmdf_drift = {}
        trace_drift = {}

        for poly in poly_list:
            if poly == 2:
                continue
            # elif poly_saturated_YN[poly - 1] == 0:
            else:
                poly_id = 'POLY_' + np.str(poly)

                print('Shot: ' + np.str(shot_number) + ', poly. ' + np.str(poly))
                pmdf_fixed[poly], trace_fixed[poly] = inference_button_fixed(poly_id, analysis_tree)
                pmdf_drift[poly], trace_drift[poly] = inference_button_drift(poly_id, analysis_tree)

                summary_f = pm.summary(trace_fixed[poly])
                summary_d = pm.summary(trace_drift[poly])
                
                summary_f.to_csv('traces/' + np.str(analysis_tree.shot) + poly_id + 'fixed.csv', index=False)
                summary_d.to_csv('traces/' + np.str(analysis_tree.shot) + poly_id + 'drift.csv', index=False)

                tree_write_safe(np.array(pmdf_fixed[poly].t_e), poly_id + '_T_E_FIXED', np.arange(1,len(pmdf_fixed[poly].t_e)+1), tree=analysis_tree)
                tree_write_safe(np.array(pmdf_drift[poly].t_e), poly_id + '_T_E_DRIFT', np.arange(1,len(pmdf_drift[poly].t_e)+1), tree=analysis_tree)
                tree_write_safe(np.array(pmdf_drift[poly].v_d), poly_id + '_V_D_DRIFT', np.arange(1,len(pmdf_drift[poly].v_d)+1), tree=analysis_tree)

                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_fixed'] = summary_f['mean'][0]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_fixed_sd'] = summary_f['sd'][0]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_fixed_mc_error'] = summary_f['mc_error'][0]

                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_fixed_hpd95low'] = summary_f['hpd_2.5'][0]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_fixed_hpd95high'] = summary_f['hpd_97.5'][0]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_fixed_Rhat'] = summary_f['Rhat'][0]

                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_drift'] = summary_d['mean'][0]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_drift_hpd95low'] = summary_d['hpd_2.5'][0]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_drift_hpd95high'] = summary_d['hpd_97.5'][0]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_T_e_drift_Rhat'] = summary_d['Rhat'][0]

                log_book.loc[log_book.Shot == shot_number, poly_id + '_v_d_drift'] = summary_d['mean'][1]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_v_d_drift_hpd95low'] = summary_d['hpd_2.5'][1]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_v_d_drift_hpd95high'] = summary_d['hpd_97.5'][1]
                log_book.loc[log_book.Shot == shot_number, poly_id + '_v_d_drift_Rhat'] = summary_d['Rhat'][1]

            # else:
            #     continue






                # return pmdf_fixed, trace_fixed, pmdf_drift, trace_drift


    else:
        print('Pick either style = BDA or REM')



    # calculate and store the electron energy distribution function,
    # including temperature, density, and electron drift velocity



    # Get the vacuum shot to use for stray light background subtraction, using an average of every vacuum shot from the
    # same day as the plasma shot, or the nearest day with a vacuum shot:




    # Write the new logbook dataframe to the master logbook file
    record_filename = 'TS_analysis_logbook' + datetime.date.today().isoformat() + '.csv'
    latest_filename = 'TS_analysis_logbook.csv'

    log_book.to_csv(record_filename, index=False)
    log_book.to_csv(latest_filename, index=False)

    analysis_tree.close()


def build_shot(shot_number):
    """
    Build the tree and logbook up with relevant Thomson data/parameters for shot_number
    ----------
    Does analysis in a frugal way that avoids duplicate math by writing
    any incremental results to the analysis3 tree

    Returns
    -------
    """
    print('Building shot ' + str(shot_number))


    # TODO remove next line
    # shot_number = 190619016
    record_filename = 'TS_analysis_logbook' + datetime.date.today().isoformat() + '.csv'
    latest_filename = 'TS_analysis_logbook.csv'

    # The logbook is the dataframe-style ledger for human-readable summary TS data
    # parameters used in the analysis code are found and written here
    log_book = get_ts_logbook()

    # Load the data, analysis, and model trees
    data_tree = Tree('hitsi3', shot_number)
    analysis_tree = Tree('analysis3', shot_number, 'EDIT')
    model_tree = Tree('analysis3', -1)

    # Ensure calibration data are in the tree,
    # both the spectral calibrations
    put_cals_in_tree(analysis_tree)

    # and the laser energy calibration


    try:
        laser_energy_calibration_slope = get_data('LASER_E_SLOPE', analysis_tree)
        laser_energy_calibration_int = get_data('LASER_E_INT', analysis_tree)
    except Exception as ex:
        if ex.msgnam == 'NNF' or ex.msgnam == 'NODATA':
            laser_energy_calibration_slope = get_data('LASER_E_SLOPE', model_tree)
            tree_write_safe(laser_energy_calibration_slope, 'LASER_E_SLOPE', tree=analysis_tree)
            laser_energy_calibration_int = get_data('LASER_E_INT', model_tree)
            tree_write_safe(laser_energy_calibration_int, 'LASER_E_INT', tree=analysis_tree)
        else:
             print(ex)

    try:
        laser_energy_calibration_slope_bayes = get_data('LASER_E_SLOPE_B', analysis_tree)
        laser_energy_calibration_int_bayes = get_data('LASER_E_INT_B', analysis_tree)
    except Exception as ex:
        if ex.msgnam == 'NNF' or ex.msgnam == 'NODATA':
            laser_energy_calibration_slope_bayes = get_data('LASER_E_SLOPE_B', model_tree)
            tree_write_safe(laser_energy_calibration_slope_bayes, 'LASER_E_SLOPE_B', tree=analysis_tree)
            laser_energy_calibration_int_bayes = get_data('LASER_E_INT_B', model_tree)
            tree_write_safe(laser_energy_calibration_int_bayes, 'LASER_E_INT_B', tree=analysis_tree)
        else:
            print(ex)

    # Find the laser timing in the context of the full shot using the \TS_OSC_TRIG data tree variable
    try:
        fire_time = get_data('FIRE_TIME', analysis_tree)
    except Exception as ex:
        if ex.msgnam == 'NNF' or ex.msgnam == 'NODATA':
            shot_timing_sig = get_data('TS_OSC_TRIG', data_tree)
            shot_timing_t = get_dim('TS_OSC_TRIG', data_tree)
            fire_time = shot_timing_t[np.where(shot_timing_sig == np.max(shot_timing_sig))[0]][0]
            tree_write_safe(fire_time, 'FIRE_TIME', tree=analysis_tree)
        else:
            print(ex)

    # Add the determined fire time to the log book
    log_book.loc[log_book.Shot == shot_number, 'FIRE_TIME'] = fire_time

    # Calculate and store the laser energy
    try:
        laser_energy = get_data('ENERGY', analysis_tree)
        laser_energy_bayes = get_data('ENERGY_BAYES', analysis_tree)
        laser_energy_var = get_data('ENERGY_VAR', analysis_tree)
    except Exception as ex:
        if ex.msgnam == 'NNF' or ex.msgnam == 'NODATA':
            energy_photodiode_sig = get_data('TS_RUBY', data_tree)
            energy_photodiode_t = get_dim('TS_RUBY', data_tree)
            photodiode_baseline = np.mean(energy_photodiode_sig[0:np.int(np.around(np.size(energy_photodiode_sig, 0)*photodiode_baseline_record_fraction))])
            energy_photodiode_sig = energy_photodiode_sig - photodiode_baseline
            integrated_photodiode = np.trapz(energy_photodiode_sig, energy_photodiode_t)
            laser_energy = (integrated_photodiode - laser_energy_calibration_int)/laser_energy_calibration_slope
            laser_energy_bayes = np.mean((integrated_photodiode - laser_energy_calibration_int_bayes)/laser_energy_calibration_slope_bayes)
            laser_energy_var = np.var((integrated_photodiode - laser_energy_calibration_int_bayes)/laser_energy_calibration_slope_bayes)/laser_energy_bayes**2
            tree_write_safe(energy_photodiode_sig, 'ENERGY_PD', dim=energy_photodiode_t, tree=analysis_tree)
            tree_write_safe(laser_energy, 'ENERGY', tree=analysis_tree)
            tree_write_safe(laser_energy_bayes, 'ENERGY_BAYES', tree=analysis_tree)
            tree_write_safe(laser_energy_var, 'ENERGY_VAR', tree=analysis_tree)
        else:
            print(ex)

    # kde = sm.nonparametric.KDEUnivariate((integrated_photodiode - laser_energy_calibration_int_bayes)/laser_energy_calibration_slope_bayes)
    # kde.fit()
    # kde.support
    # kde.density
    # # p = pm.kdeplot((integrated_photodiode - laser_energy_calibration_int_bayes)/laser_energy_calibration_slope_bayes)
    # # x,y = p.get_lines()[0].get_data()

    # Add the determined laser energy to the log book
    log_book.loc[log_book.Shot == shot_number, 'Energy'] = laser_energy
    log_book.loc[log_book.Shot == shot_number, 'Energy_Bayes'] = laser_energy_bayes

    # Get the photodiode energy trace
    energy_photodiode_sig = get_data('ENERGY_PD', analysis_tree)
    energy_photodiode_t = get_dim('ENERGY_PD', analysis_tree)

    # Add the time of peak laser power to the log book
    peak_laser_time = energy_photodiode_t[np.where(energy_photodiode_sig == np.max(energy_photodiode_sig))[0][0]]
    log_book.loc[log_book.Shot == shot_number, 'Pulse_Time'] = peak_laser_time

    # Get the vacuum shots to use for stray light background subtraction
    vacuum_shot_list = get_vacuum_shot_list(shot_number, log_book, num_vacuum_shots)
    # TODO write the vacuum shots to the logbook

    # Get a list of the active polychromators by name
    regex = re.compile('POLY_._.')
    channel_list = [string for string in log_book.columns if re.match(regex, string) and log_book.loc[log_book.Shot == shot_number, string].array[0] != 0 and len(string) == 8]
    
    # Get the geometric parameters for the measurements of shot_number
    geometry = get_spot_geometry(log_book.loc[log_book.Shot == shot_number, 'Mount_Positions'].array[0], log_book.loc[log_book.Shot == shot_number, 'Jack_Height'].array[0])

    # Write the new logbook dataframe to the master logbook file
    log_book.to_csv(record_filename, index=False)
    log_book.to_csv(latest_filename, index=False)

    # Analyze all active polychromator channels
    for channel_id in channel_list:
        # print(str(shot_number) + ' ' + str(channel_id))

        # Put the geometry data for each polychromator in the analysis tree if it isn't already:
        for tag_name in geometry.columns.array[1:]:
            data = geometry.loc[geometry.polychromator == int(channel_id.rsplit(sep='_')[-2]), tag_name].array[0]
            tree_write_safe(data, channel_id[:-2] + '_' + tag_name, tree=analysis_tree)

        # Put the raw polychromator data in the analysis tree if it isn't already:
        try:
            raw_poly_channel_sig = get_data(channel_id + '_RAW', analysis_tree)
        except Exception as ex:
            raw_poly_channel_sig = get_data('TS_POLY' + channel_id[-3:], data_tree)
            raw_poly_channel_t = get_dim('TS_POLY' + channel_id[-3:], data_tree)
            if len(raw_poly_channel_sig) > 9952:
                raw_poly_channel_sig = raw_poly_channel_sig[24:-24]
                raw_poly_channel_t = raw_poly_channel_t[24:-24]

            tree_write_safe(raw_poly_channel_sig, channel_id + '_RAW', dim=raw_poly_channel_t, tree=analysis_tree)

        # Set up a vacuum shot if this is a plasma shot:
        if log_book.loc[log_book.Shot == shot_number, 'Fuel'].array[0] != 'V' and log_book.loc[log_book.Shot == shot_number, 'Fuel'].array[0] != '-':
            try:
                poly_channel_vacuum_avg = get_data(channel_id + '_VAC', analysis_tree)
            except Exception as ex:
                vacuum_sig = {}
                vacuum_t = {}
                ctr = 0
                for vac_shot in vacuum_shot_list:
                    try:
                        vac_tree = Tree('analysis3', vac_shot)
                        vacuum_sig[ctr] = get_data(channel_id + '_RAW', vac_tree)
                        vacuum_t[ctr] = get_dim(channel_id + '_RAW', vac_tree)
                        vac_energy = get_data('ENERGY_BAYES', vac_tree)
                    except Exception as ex:
                        vac_tree.close()
                        build_shot(vac_shot)
                        log_book = get_ts_logbook()
                        # print(str(vac_shot) + ' ' + channel_id + '_RAW')
                        vac_tree = Tree('analysis3', vac_shot)
                        vacuum_sig[ctr] = get_data(channel_id + '_RAW', vac_tree)
                        vacuum_t[ctr] = get_dim(channel_id + '_RAW', vac_tree)
                        vac_energy = get_data('ENERGY_BAYES', vac_tree)

                    vac_tree.close()
                    peak_vac_time = log_book.loc[log_book.Shot == vac_shot, 'Pulse_Time'].array[0]
                    # vacuum_sig[ctr] = signal.detrend(slide_trace(peak_vac_time, peak_laser_time, vacuum_sig[ctr], vacuum_t[ctr]))/vac_energy
                    vacuum_sig[ctr] = slide_trace(peak_vac_time, peak_laser_time, vacuum_sig[ctr], vacuum_t[ctr])/vac_energy
                    ctr = ctr + 1

                poly_channel_vacuum_avg = signal.savgol_filter(signal.detrend(np.array([trace*laser_energy for trace in vacuum_sig.values()]).mean(axis=0)), 101, 3)
                tree_write_safe(poly_channel_vacuum_avg, channel_id + '_VAC', dim=vacuum_t[0], tree=analysis_tree)

    # Load the data, analysis, and model trees
    data_tree.close()
    analysis_tree.close()
    model_tree.close()

    # Write the new logbook dataframe to the master logbook file
    log_book.to_csv(record_filename, index=False)
    log_book.to_csv(latest_filename, index=False)


def get_ts_logbook(verbose=None, force_update=None):
    """
    Pull in the logbook information from the TS_Logbook Google sheet, parse it, and return a version of it
    for use in the TS analysis.
    ----------
    (none)

    Returns
    -------
    """

    # get the logbook spreadsheet from google sheets and read it in to a pandas dataframe
    raw_google_csv = requests.get(
        'https://docs.google.com/spreadsheet/ccc?key=1yF7RMpYl_KvZPoEwYiH1D_vVrasaTGLuPvdBoYRRM84&output=csv')
    data = raw_google_csv.content
    raw_ts_log = pd.read_csv(BytesIO(data), usecols=np.arange(1, 10))

    # drop Nan's from gaps between run days and other non-shots
    raw_ts_log = raw_ts_log.dropna(subset=['Shot', 'Plasma Species']).reset_index()

    # identify and get rid of duplicates if necessary
    raw_ts_log_nodupes = raw_ts_log.drop_duplicates(subset=['Shot'])

    if raw_ts_log.shape[0] != raw_ts_log_nodupes.shape[0]:
        print("*****WARNING! Duplicates while parsing TS logbook shots:*****")
        duplicates = raw_ts_log.loc[raw_ts_log.duplicated(subset=['Shot'])]['Shot'].astype(int)
        print(duplicates.to_csv(sep='\t', index=False))
        print("...excluding duplicates from parsed logbook.")

    raw_ts_log = raw_ts_log_nodupes

    # if the logbook has already been analyzed today just use that, otherwise update the logbook and then read it in:
    latest_filename = 'TS_analysis_logbook.csv'
    try:
        latest_ts_log = pd.read_csv(latest_filename)
        if latest_ts_log['Shot'].iloc[-1] == raw_ts_log['Shot'].iloc[-1] and force_update is None:
            if verbose == 1:
                print('Using previous analysis logbook...')
            return latest_ts_log
        else:
            latest_ts_log = pd.concat([latest_ts_log, raw_ts_log], ignore_index=True, sort=False).drop_duplicates(subset=['Shot'])
    except FileNotFoundError:
        latest_filename = 'TS_logbook_autogen.csv'
        try:
            latest_ts_log = pd.read_csv(latest_filename)
            if latest_ts_log['Shot'].iloc[-1] == raw_ts_log['Shot'].iloc[-1] and force_update is None:
                if verbose == 1:
                    print('Using previously pulled logbook...')
                return latest_ts_log
            else:
                pass
        except FileNotFoundError:
            pass
    print('Updating logbook...')

    # clean up and start the dataframe to return
    generated_ts_log = raw_ts_log[['Shot', 'TS_TRIG (ms)', 'Energy', 'Plasma Species', 'Notes', 'Fiber Mount Positions (in order polys. 12345)', 'Lab jack height (in.)']]
    generated_ts_log = generated_ts_log.rename(index=str, columns={'Plasma Species': 'Fuel'})
    generated_ts_log = generated_ts_log.rename(index=str, columns={'TS_TRIG (ms)': 'TS_TRIG'})
    generated_ts_log = generated_ts_log.rename(index=str, columns={'Fiber Mount Positions (in order polys. 12345)': 'Mount_Positions'})
    generated_ts_log = generated_ts_log.rename(index=str, columns={'Lab jack height (in.)': 'Jack_Height'})
    generated_ts_log['Shot'] = generated_ts_log.Shot.astype('int64')
    # generated_ts_log['Fuel'].replace({'-': 'V'}, inplace=True)

    generated_ts_log.reset_index(inplace=True, drop=True)

    # parse the laser voltage column
    voltage_parse_dataframe = pd.DataFrame()

    for laser_power_string in raw_ts_log['Laser Power (OSC-AMP12-AMP3)']:
        voltages = np.array(laser_power_string.rsplit(sep='-')).astype(np.double)*10
        power_oneshot_dataframe = pd.DataFrame(data=[voltages], columns=['V_osc', 'V_amps12', 'V_amp3'])

        voltage_parse_dataframe = pd.concat([voltage_parse_dataframe, power_oneshot_dataframe], ignore_index=True, sort=False)

    voltage_parse_dataframe.reset_index(inplace=True, drop=True)
    generated_ts_log = pd.concat([generated_ts_log, voltage_parse_dataframe], axis=1)

    # parse the active polychromator channels column
    polychromator_parse_dataframe = pd.DataFrame()

    for active_polychromators_string in raw_ts_log['Active Channels (polychromator no. in tens place)']:
        polychromators_channels_list = active_polychromators_string.rsplit(sep=',')
        total_channels = len(polychromators_channels_list)
        polychromators_oneshot_dataframe = pd.DataFrame(data=np.ones([1, total_channels]), columns=polychromators_channels_list)

        polychromator_parse_dataframe = pd.concat([polychromator_parse_dataframe, polychromators_oneshot_dataframe], ignore_index=True, sort=False)

    polychromator_parse_dataframe.fillna(value=0, inplace=True)
    polychromator_parse_dataframe.rename(columns=lambda x: 'POLY_'+x[0]+'_'+x[1], inplace=True)
    polychromator_parse_dataframe.reset_index(inplace=True, drop=True)
    generated_ts_log = pd.concat([generated_ts_log, polychromator_parse_dataframe], axis=1)

    # calculate and store the geometry data for the shot using the lab jack height and mount positions string
    geometry_parse_dataframe = pd.DataFrame()

    for ind in np.arange(raw_ts_log.shape[0]):
        fiber_mount_string = raw_ts_log['Fiber Mount Positions (in order polys. 12345)'].iloc[ind]
        jack_height = raw_ts_log['Lab jack height (in.)'].iloc[ind]
        geometry_oneshot_dataframe = get_spot_geometry(fiber_mount_string, jack_height)
        r_oneshot_dataframe = pd.DataFrame(data=[geometry_oneshot_dataframe['R'].array], columns=['radius_poly_' + str(p) for p in geometry_oneshot_dataframe['polychromator']])
        geometry_parse_dataframe = pd.concat([geometry_parse_dataframe, r_oneshot_dataframe.astype(np.float64)], ignore_index=True, sort=False)

    generated_ts_log = pd.concat([generated_ts_log, geometry_parse_dataframe], axis=1)
    number_of_shots = ind + 1

    # calculate and store the peak I_tor (sihi smoothed)
    hit_conn = Connection('landau.hit')
    i_tor_max = np.zeros((number_of_shots, 1))
    ctr = 0
    for shot in generated_ts_log['Shot']:
        hit_conn.openTree("hitsi3", shot)
        try:
            i_tor = np.array(hit_conn.get("\\I_TOR_SPAAVG"))
            if np.max(np.abs(i_tor)) == np.max(i_tor):
                i_tor_max[ctr] = np.max(i_tor)
            else:
                i_tor_max[ctr] = np.min(i_tor)
        except Exception as ex:
            print('No i_tor data for shot '+str(shot))
            i_tor_max[ctr] = np.nan
        ctr = ctr + 1

    max_i_tor_dataframe = pd.DataFrame(data=i_tor_max, columns=['i_tor_max'])
    generated_ts_log = pd.concat([generated_ts_log, max_i_tor_dataframe], axis=1)
    
    latest_filename = 'TS_logbook_autogen.csv'
    record_filename = 'TS_logbook_autogen_' + datetime.date.today().isoformat() + '.csv'
    generated_ts_log.to_csv(record_filename, index=False)
    generated_ts_log.to_csv(latest_filename, index=False)

    return generated_ts_log


def get_spot_geometry(fiber_mount_string_in, jack_height_in):
    """
    Gives geometry info for the spot specified by the lab jack height and the mount position of the fibers
    ----------
    Parameters
    ----------
    fiber_mount_string_in
    jack_height_in

    Returns
    -------
    return_dataframe: dataframe of spot geometry data including
        spot length
        r, phi, z coordinates of the measurement spots
        the uncertainties r_pos, r_neg, z_pos, z_neg
        view_angle from the spot to the collection optics
        f_number from the spot to the collection optics
        solid_angle from the spot to the collection optics
    """

    # the laser line is a cord at radius 21 cm from the goemetric axis, and the
    # point at which a radius line intersects the laser at a right angle is
    # where the Thomson measuring/alignment string showed 52 cm. So:
    r_laser = 21    # laser "impact parameter"
    origin_measurements = 52
    uncertainty = 0.5

    fiber_mount_list = list(fiber_mount_string_in)
    mount_positions = pd.Series(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])
    all_jack_heights = pd.DataFrame(data=[0.927, 1.000, 1.125, 1.250, 1.375, 1.500, 1.625, 1.750, 1.875], columns=['jack_height'])

    # each row in these dataframes corresponds to a different lab jack height:
    # each column gives the distance in cm from the origin of the TS target onto which the fibers were backlit
    # to the high or low edge of the spot (in the up/down lab frame of reference).
    spot_low_edge = pd.DataFrame(data=[[67.5, 70, 72, 74.5, 76.75, 79, 81.25, 83.5, 86, 88.25, 90.75],
                                       [67.25, 69.5, 72, 74, 76.5, 78.75, 81, 83.25, 85.5, 88, 90],
                                       [66, 68.5, 71, 73.25, 75.5, 77.75, 80, 82.5, 84.5, 87, 89.5],
                                       [65, 67.75, 70, 72.25, 74.5, 77, 79, 81.5, 83.75, 85, 88],
                                       [64.25, 66.75, 69, 71.5, 73.5, 76, 78.25, 80.5, 82.75, 85, 87],
                                       [63.5, 66, 68.25, 70.5, 73, 75, 77.5, 79.5, 82, 84, 86],
                                       [62.5, 65, 67.5, 69.5, 72, 74.25, 76.5, 78.75, 81, 83.5, 85],
                                       [61.5, 64, 66.5, 68.75, 71, 73.5, 75.5, 78, 80.25, 82.5, 84.5],
                                       [60.5, 63, 65.5, 68, 70.25, 72.5, 75, 77, 79.5, 81, 83]], columns=mount_positions)

    spot_low_edge = pd.concat([spot_low_edge, all_jack_heights], axis=1)
    spot_high_edge = pd.DataFrame(data=[[69.5, 71.5, 73.5, 76, 77.25, 80.5, 82.75, 85, 87.5, 89.75, 92.25],
                                        [68.75, 71, 73.5, 75.5, 78, 80.25, 82.5, 84.75, 87, 89.5, 92],
                                        [68, 70, 72.5, 74.75, 77, 79.25, 81.5, 84, 86, 88.5, 91],
                                        [67, 69.25, 71.5, 73.75, 76, 78.5, 80.5, 83, 85.25, 87.5, 90],
                                        [66, 68.25, 70.5, 73, 75, 77.5, 79.75, 82, 84.25, 86.75, 89],
                                        [65, 67.5, 69.75, 72, 74.25, 76.5, 79, 81, 83.5, 85.75, 88],
                                        [64, 66.5, 69, 71, 73.5, 75.75, 78, 80.25, 82.5, 85, 87.25],
                                        [63, 65.5, 68, 70.25, 72.5, 75, 77, 79.5, 81.75, 84, 86.5],
                                        [62, 64.5, 67, 69.5, 71.75, 74, 76, 78.5, 81, 83, 85.5]], columns=mount_positions)

    spot_high_edge = pd.concat([spot_high_edge, all_jack_heights], axis=1)

    geom_dataframe = pd.DataFrame(columns=['polychromator', 'LENGTH', 'R', 'PHI', 'Z',
                                           'R_POS', 'R_NEG', 'Z_POS', 'Z_NEG',
                                           'THETA', 'SOLID_ANGLE', 'K_R', 'K_PHI', 'K_Z'])

    polychromator = 0
    for fiber_mount_character in fiber_mount_list:
        polychromator = polychromator + 1
        if fiber_mount_character == 'X':
            continue

        spot_bottom = spot_low_edge[fiber_mount_character][spot_low_edge.jack_height == jack_height_in].array[0] - origin_measurements
        spot_top = spot_high_edge[fiber_mount_character][spot_high_edge.jack_height == jack_height_in].array[0] - origin_measurements

        length = spot_top - spot_bottom
        raw_center = spot_bottom + length/2
        r = np.sqrt(r_laser**2 + raw_center**2)

        r_pos = (spot_top - raw_center)*((raw_center)/r)
        r_neg = (raw_center - spot_bottom)*((raw_center)/r)

        z = 0*r    # measurements are made at midplane
        z_pos = z + uncertainty
        z_neg = z + uncertainty

        window_center = 26.10
        window_z = 35.245
        spot_window_offset = window_center - raw_center
        r_scattering_vector = np.sqrt(spot_window_offset**2 + window_z**2)
        view_angle = np.arctan(abs(spot_window_offset)/window_z)

        if spot_window_offset < 0:
            angle_scattering_vector_k_ruby = np.pi/2 - view_angle
        else:
            angle_scattering_vector_k_ruby = np.pi/2 + view_angle

        theta = angle_scattering_vector_k_ruby
        window_diameter_cm = 3.75*2.54
        phi = np.arctan(raw_center/r_laser) + np.pi/4
        area_lens = np.pi*(window_diameter_cm/2)**2
        solid_angle = area_lens/r_scattering_vector**2
        f_number = r_scattering_vector/(window_diameter_cm)

        # scattering vector:
        k_s_r = np.cos(theta)*(np.cos(np.pi - np.arcsin(r_laser/r)))
        k_s_phi = -1*np.cos(theta)*(r_laser/r)
        k_s_z = np.sin(theta)

        # incident vector:
        k_i_r = np.cos(np.pi - np.arcsin(r_laser/r))
        k_i_phi = -1*(r_laser/r)
        k_i_z = 0

        # print(np.sqrt(k_s_r**2 + k_s_phi**2 + k_s_z**2))
        # print(np.sqrt(k_i_r**2 + k_i_phi**2 + k_i_z**2))

        # total doppler shift vector k = k_s - k_i:
        k_r = k_s_r - k_i_r
        k_phi = k_s_phi - k_i_phi
        k_z = k_s_z - k_i_z
        k_l = np.sqrt(k_r**2 + k_phi**2 + k_z**2)
        k_r = k_r/k_l
        k_phi = k_phi/k_l
        k_z = k_z/k_l

        # report in [m]
        geom_dataframe = pd.concat([geom_dataframe, pd.DataFrame(data=[[polychromator, length/100, r/100, phi, z/100,
                                                                       r_pos/100, r_neg/100, z_pos/100, z_neg/100,
                                                                       theta, solid_angle, k_r, k_phi, k_z]],
                                                                 columns=['polychromator', 'LENGTH', 'R', 'PHI', 'Z',
                                                                          'R_POS', 'R_NEG', 'Z_POS', 'Z_NEG',
                                                                          'THETA', 'SOLID_ANGLE', 'K_R', 'K_PHI', 'K_Z'])],
                                                                 ignore_index=True, sort=False)

        # TODO: add TS scattering parameter to the returned values

    return(geom_dataframe)


def update_energy_cal():
    """
    Update the regression used to create a calibration factor for converting the
    photodiode response to an energy measurement

    Parameters
    ----------
    log_book: Logbook in the style of the Pandas dataframe object

    Returns
    -------
    """

    hit_conn = Connection('landau.hit')

    log_book = get_ts_logbook()
    energy_measured = log_book.Energy[log_book.Fuel == 'ETEST']
    energy_integrated = pd.Series()
    for shot in log_book.Shot[log_book.Fuel == 'ETEST']:
        hit_conn.openTree("hitsi3", shot)
        try:
            flux_photodiode = np.array(hit_conn.get("\\TS_RUBY"))
            flux_photodiode_t = np.array(hit_conn.get("DIM_OF(\\TS_RUBY)"))
        except EOFError:
            print("WARNING: Error reading photodiode data from shot", shot)
            # return -1
            pass

        flux_baseline = np.mean(flux_photodiode[0:np.int(np.around(np.size(flux_photodiode, 0)*photodiode_baseline_record_fraction))])
        flux_photodiode = flux_photodiode - flux_baseline

        energy_integrated = energy_integrated.append(pd.Series([np.trapz(flux_photodiode, flux_photodiode_t)]), ignore_index=True)



    # A = np.transpose(np.array([energy_measured, (np.ones_like(energy_measured))]))
    # m, c = np.linalg.lstsq(A, energy_integrated,rcond=None)[0]
    energy_integrated = energy_integrated.to_numpy().reshape(-1, 1)
    energy_measured = energy_measured.to_numpy().reshape(-1, 1)

    # Model initialization
    regression_model = LinearRegression()

    # Fit the data
    regression_model.fit(energy_measured, energy_integrated)

    # Predict
    energy_predicted = regression_model.predict(energy_measured)

    # model evaluation
    rmse = mean_squared_error(energy_integrated, energy_predicted)
    r2 = r2_score(energy_integrated, energy_predicted)
    m = regression_model.coef_[0][0]
    b = regression_model.intercept_[0]

    if PLOTS_ON == 1:
        # printing values
        print('Slope:', m)
        print('Intercept:', b)
        print('Root mean squared error: ', rmse)
        print('R2 score: ', r2)

        fig1, ax1 = plt.subplots()
        ax1.set_title("Linear regression")
        ax1.set_xlabel(r"$E_{meter} [J]$")
        ax1.set_ylabel(r"$E_{photodiode} [J]$")
        ax1.plot(energy_measured, energy_integrated, 'o', label='Original data', markersize=2)
        ax1.plot(np.arange(0, 10), regression_model.predict(np.arange(0, 10).reshape(-1, 1)), label='Fitted line')
        # ax1.plot(np.arange(0, 10), np.arange(0, 10), color='k', ls='--', linewidth=0.5)
        ax1.legend()
        ax1.grid(ls='--')

    tree_write_safe(m, 'LASER_E_SLOPE')
    tree_write_safe(b, 'LASER_E_INT')

    with pm.Model() as linear_model:
        # Intercept
        intercept = pm.Normal('intercept', mu=0, sd=5)
        # intercept = pm.Uniform('intercept',lower=0, upper=1)

        # Slope
        # slope = pm.Normal('slope', mu=0, sd=10)
        slope = pm.Uniform('slope',lower=0, upper=1)

        # Standard deviation
        sigma = pm.HalfNormal('sigma', sd=10)

        # Estimate of mean
        mean = intercept + slope*energy_measured

        # Observed values
        Y_obs = pm.Normal('Y_obs', mu=mean, sd=sigma, observed=energy_integrated)

        # Sampler
        step = pm.NUTS(target_accept=0.95)

        # Posterior distribution
        linear_trace = pm.sample(2000, step, tune=4000)
        # linear_trace = pm.sample(1000, step, tune=2000)
        pm.summary(linear_trace)

    if PLOTS_ON == 1:
        pm.traceplot(linear_trace, figsize=(12, 12))
        pm.plot_posterior(linear_trace, figsize=(12, 10), text_size=20, credible_interval=0.95, round_to=12)
        # pm.forestplot(linear_trace)

        plt.figure(figsize=(8, 8))
        pm.plot_posterior_predictive_glm(linear_trace, samples=100, eval=np.linspace(0, 10, 100), linewidth=1,
                                         color='red', alpha=0.05, label='Bayesian Posterior Fits',
                                         lm=lambda x, sample: sample['intercept'] + sample['slope'] * x)
        plt.scatter(energy_measured[:500], energy_integrated[:500], s=12, alpha=0.8, c='blue', label='Observations')

        # bayes_prediction = (1e-07 - linear_trace['Intercept'])/linear_trace['slope']
        # plt.figure(figsize = (8, 8))
        # sns.kdeplot(bayes_prediction, label = 'Bayes Posterior Prediction')
        # plt.vlines(x = (1e-07 - c)/m,
        #            ymin = 0, ymax = 2.5,
        #            label = 'OLS Prediction',
        #            colors = 'red', linestyles='--')
        print(pm.summary(linear_trace))

    tree_write_safe(linear_trace['slope'], 'LASER_E_SLOPE_B')
    tree_write_safe(linear_trace['intercept'], 'LASER_E_INT_B')

#        fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figures
#        ax2.plot(flux_photodiode_t,flux_photodiode)
#        ax3.scatter(energy_measured, energy_integrated/m)


def tree_write_safe(data_to_write, tag_name, dim=None, tree=None):
    """
    If the node exists, write the data to that node, including its independent variable (often time or wavelength).
    If the node does not exist, create it and then write the data.

    Parameters
    ----------
    data_to_write
    dim, the independent variable of data_to_write if necessary
    tag_name
    tree

    Returns
    -------
    """

    if tree is None:
        # t = MDSplus.Tree('analysis3',-1).createPulse(shot=888)    # to create a new practice tree
        # TODO: change the "888" in the line below to "-1" when this code is ready for primetime.
        tree = Tree('analysis3', -1, 'EDIT')
        tree_was = None
    else:
        tree_was = tree

    try:
        node_to_write = tree.getNode('\\' + tag_name)
    except Exception as ex:
        if ex.msgnam == 'NNF':
            add_node_safe(tag_name, tree)
        else:
            print(ex)
        node_to_write = tree.getNode('\\' + tag_name)

    # then write the data based on what type or "usage" it is (MDSplus terminology)
    node_usage = node_to_write.getUsage()
    node_units = thomson_tree_lookup['Units'][thomson_tree_lookup['Tag'] == tag_name].values[0]
    node_dim_units = thomson_tree_lookup['dim_Units'][thomson_tree_lookup['Tag'] == tag_name].values[0]

    if node_usage == 'SIGNAL' and dim is None:
        write_string = "BUILD_SIGNAL(BUILD_WITH_UNITS($1,'" + node_units + "'), $1)"
        expr = Data.compile(write_string, data_to_write)
        node_to_write.putData(expr)

    elif node_usage == 'SIGNAL':
        write_string = "BUILD_SIGNAL(BUILD_WITH_UNITS($1,'" + node_units + "'), $1, BUILD_WITH_UNITS($2,'" + node_dim_units + "'))"
        expr = Data.compile(write_string, data_to_write, dim)
        node_to_write.putData(expr)

    elif node_usage == 'NUMERIC':
        node_to_write.putData(data_to_write)
        node_to_write.setUnits(str(node_units))

    else:
        print('**** DATA TYPE NOT RECOGNIZED ****')
        return(-1)

    tree.write()
    if tree_was is None:
        tree.close()


def add_node_safe(tag_name_in, tree):
    """
    Add a node specified by the input tag to the analysis3 tree (default tree is a test tree, shot 888) and include all
    of the parameters as specified in the matching node of the model tree
    Parameters
    ----------
    tag_name_in: unique tag name to determine node
    tree: tree to add node to

    Returns
    -------
    """

    try:
        node_string = '\\' + thomson_tree_lookup['Path'][thomson_tree_lookup['Tag'] == tag_name_in].values[0]
    except Exception as ex:
        if str(ex.args) == "('index 0 is out of bounds for axis 0 with size 0',)":
            print('!*!*!*!*! INVALID TAG NAME !*!*!*!*!*! \nCheck global variable thomson_tree_lookup or tag_name_in in function add_node_safe().')
        else:
            print('***ERROR in add_node_safe()***')

    node_usage = thomson_tree_lookup['Usage'][thomson_tree_lookup['Tag'] == tag_name_in].values[0]

    # then add appropriate nodes (recursive?) until all parent (type 'STRUCTURE') nodes are built
    try:
        tree.addNode(node_string, node_usage).addTag(tag_name_in)
        tree.write()
    except Exception as ex:
        if ex.msgnam == 'NNF':
            print('Parent node for ' + node_string + ' not in tree, creating...')
            add_parent(node_string, tree)
            tree.addNode(node_string, node_usage).addTag(tag_name_in)
        elif ex.msgnam == 'ALREADY_THERE':
            print("Node " + node_string + " already exists in the tree: " + str(tree))
            pass
        else:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            return -1

    # print(tag_name_in)
    # print(node_string)
    # print(tree)
    node = tree.getNode(node_string)
    node.setUsage(node_usage)

    tree.write()


def add_parent(node_name_in, tree):
    """
    Recursively adds the parent and grandparent nodes necessary up to the immediate parent of node_name_in.

    Parameters
    ----------
    node_name_in: node name full path off of which the last node is stripped to form parent_string
    tree: tree to add node to

    Returns
    -------
    """
    node_name_in = node_name_in.replace(':','.').replace('..','::')
    parent_string = node_name_in.rsplit(sep='.', maxsplit=1)[0]

    try:
        tree.addNode(parent_string, 'STRUCTURE')
        tree.write()
    except Exception as ex:
        if ex.msgnam == 'NNF':
            add_parent(parent_string, tree)
        tree.addNode(parent_string, 'STRUCTURE')
        tree.write()


def put_cals_in_tree(tree=None):
    """
    A script to put calibration traces into the (model) tree to be retreived later when necessary

    Parameters
    ----------
    tree (optional)

    Returns
    -------
    """
    cal_poly_list = ['CAL_5_CH_FG_POLY_3_T_1', 'CAL_5_CH_FG_POLY_3_T_2', 'CAL_5_CH_FG_POLY_3_T_3', 'CAL_5_CH_FG_POLY_3_T_4', 'CAL_5_CH_FG_POLY_3_T_5',
                     'CAL_5_CH_FG_POLY_3_V_1', 'CAL_5_CH_FG_POLY_3_V_2', 'CAL_5_CH_FG_POLY_3_V_3', 'CAL_5_CH_FG_POLY_3_V_4', 'CAL_5_CH_FG_POLY_3_V_5',
                     'CAL_5_CH_FG_POLY_4_T_1', 'CAL_5_CH_FG_POLY_4_T_2', 'CAL_5_CH_FG_POLY_4_T_3', 'CAL_5_CH_FG_POLY_4_T_4', 'CAL_5_CH_FG_POLY_4_T_5',
                     'CAL_5_CH_FG_POLY_4_V_1', 'CAL_5_CH_FG_POLY_4_V_2', 'CAL_5_CH_FG_POLY_4_V_3', 'CAL_5_CH_FG_POLY_4_V_4', 'CAL_5_CH_FG_POLY_4_V_5',
                     'CAL_5_CH_FG_POLY_5_T_1', 'CAL_5_CH_FG_POLY_5_T_2', 'CAL_5_CH_FG_POLY_5_T_3', 'CAL_5_CH_FG_POLY_5_T_4', 'CAL_5_CH_FG_POLY_5_T_5',
                     'CAL_5_CH_FG_POLY_5_V_1', 'CAL_5_CH_FG_POLY_5_V_2', 'CAL_5_CH_FG_POLY_5_V_3', 'CAL_5_CH_FG_POLY_5_V_4', 'CAL_5_CH_FG_POLY_5_V_5',
                     'CAL_5_CH_HG_POLY_3_T_1', 'CAL_5_CH_HG_POLY_3_T_2', 'CAL_5_CH_HG_POLY_3_T_3', 'CAL_5_CH_HG_POLY_3_T_4', 'CAL_5_CH_HG_POLY_3_T_5',
                     'CAL_5_CH_HG_POLY_3_V_1', 'CAL_5_CH_HG_POLY_3_V_2', 'CAL_5_CH_HG_POLY_3_V_3', 'CAL_5_CH_HG_POLY_3_V_4', 'CAL_5_CH_HG_POLY_3_V_5',
                     'CAL_5_CH_HG_POLY_4_T_1', 'CAL_5_CH_HG_POLY_4_T_2', 'CAL_5_CH_HG_POLY_4_T_3', 'CAL_5_CH_HG_POLY_4_T_4', 'CAL_5_CH_HG_POLY_4_T_5',
                     'CAL_5_CH_HG_POLY_4_V_1', 'CAL_5_CH_HG_POLY_4_V_2', 'CAL_5_CH_HG_POLY_4_V_3', 'CAL_5_CH_HG_POLY_4_V_4', 'CAL_5_CH_HG_POLY_4_V_5',
                     'CAL_5_CH_HG_POLY_5_T_1', 'CAL_5_CH_HG_POLY_5_T_2', 'CAL_5_CH_HG_POLY_5_T_3', 'CAL_5_CH_HG_POLY_5_T_4', 'CAL_5_CH_HG_POLY_5_T_5',
                     'CAL_5_CH_HG_POLY_5_V_1', 'CAL_5_CH_HG_POLY_5_V_2', 'CAL_5_CH_HG_POLY_5_V_3', 'CAL_5_CH_HG_POLY_5_V_4', 'CAL_5_CH_HG_POLY_5_V_5',
                     'CAL_3_CH_POLY_1_T_1', 'CAL_3_CH_POLY_1_T_2', 'CAL_3_CH_POLY_1_T_3',
                     'CAL_3_CH_POLY_1_V_1', 'CAL_3_CH_POLY_1_V_2', 'CAL_3_CH_POLY_1_V_3',
                     'CAL_3_CH_POLY_2_T_1', 'CAL_3_CH_POLY_2_T_2', 'CAL_3_CH_POLY_2_T_3',
                     'CAL_3_CH_POLY_2_V_1', 'CAL_3_CH_POLY_2_V_2', 'CAL_3_CH_POLY_2_V_3',
                     'CAL_3_CH_POLY_3_T_1', 'CAL_3_CH_POLY_3_T_2', 'CAL_3_CH_POLY_3_T_3',
                     'CAL_3_CH_POLY_3_V_1', 'CAL_3_CH_POLY_3_V_2', 'CAL_3_CH_POLY_3_V_3',
                     'CAL_3_CH_POLY_4_T_1', 'CAL_3_CH_POLY_4_T_2', 'CAL_3_CH_POLY_4_T_3',
                     'CAL_3_CH_POLY_4_V_1', 'CAL_3_CH_POLY_4_V_2', 'CAL_3_CH_POLY_4_V_3']
    
    # slow_to_fast_cal = []
    capacitance_5ch = 10**(-12)*np.array([[np.NAN,np.NAN,np.NAN,np.NAN,np.NAN],
                                          [np.NAN,np.NAN,np.NAN,np.NAN,np.NAN],
                                          [1.1852,1.1895,1.1665,1.2006,1.1805],
                                          [1.1891,1.1586,1.1927,1.1836,1.1642],
                                          [1.1985,1.2131,1.2138,1.2433,1.1757]])
    
    capacitance_3ch = 10**(-12)*np.array([[1.1985,1.2131,1.2138],
                                          [1.1757,1.1999,1.1872],
                                          [1.1852,1.1895,1.1665],
                                          [1.1891,1.1586,1.1927]])
    
    max_transmission = list([0,0,0])
    # pulse time from Kiyong's fast/slow calibration:
    t_0 = 30e-9

    if tree is None:
        tree = Tree('analysis3', -1, 'EDIT')

        for poly_tag in cal_poly_list:
            info_list = poly_tag.rsplit(sep='_')
            if info_list[1] == '3':
                # path_to_load = '/Users/chris/Dropbox/HIT_TS/Calibrations/3_CH/P' + info_list[-3] + '.mat'
                path_to_load = '/home/everson/Dropbox/HIT_TS/Calibrations/3_CH/P' + info_list[-3] + '.mat'
                capacitance = capacitance_3ch
                max_ind = 0
            else:
                # path_to_load = '/Users/chris/Dropbox/HIT_TS/Calibrations/5_CH_' + info_list[3] + '/P' + info_list[-3] + '.mat'
                path_to_load = '/home/everson/Dropbox/HIT_TS/Calibrations/5_CH_' + info_list[3] + '/P' + info_list[-3] + '.mat'
                capacitance = capacitance_5ch
                if info_list[3] == 'FG':
                    max_ind = 1
                else:
                    max_ind = 2


            loaded_mat = loadmat(path_to_load, struct_as_record=False)
            poly_string = 'p' + info_list[-3]
            poly_num = np.int(info_list[-3])
            channel_num = np.int(info_list[-1])
            tau = 50000*capacitance[poly_num - 1, channel_num - 1]
            v_slow_to_fast_factor = (1/2)*((1 - np.exp(-t_0/tau))/(1 - np.exp(-t_0/2.46e-7)))
            transmission = (loaded_mat[poly_string][0][0].t)*v_slow_to_fast_factor
            if np.max(transmission) > max_transmission[max_ind]:
                max_transmission[max_ind] = np.max(transmission)
            wavelength = loaded_mat[poly_string][0][0].l
            variance = loaded_mat[poly_string][0][0].v
            channel_index = int(info_list[-1]) - 1
            
            
            
            if info_list[-2] == 'V':
                tree_write_safe(variance[channel_index], poly_tag, dim=wavelength[0], tree=tree)
            else:
                tree_write_safe(transmission[channel_index], poly_tag, dim=wavelength[0], tree=tree)
                c_tag = poly_tag.replace('T','C')
                tree_write_safe(capacitance[poly_num - 1, channel_num - 1], c_tag, tree=tree)

        for poly_tag in cal_poly_list:
            if poly_tag[-3] == 'T':
                data_t = get_data(poly_tag, tree)
                wavelength = get_dim(poly_tag, tree)
                if poly_tag[9] == 'F':
                    max_ind = 1
                elif poly_tag[9] == 'H':
                    max_ind = 2
                else:
                    max_ind = 0
                data_t = data_t/max_transmission[max_ind]
                tree_write_safe(data_t, poly_tag, dim=wavelength, tree=tree)
            else:
                pass

        tree.close()

    else:

        for poly_tag in cal_poly_list:
            info_list = poly_tag.rsplit(sep='_')
            analysis_tree = Tree('analysis3', -1)

            if info_list[1] == '3':
                capacitance = capacitance_3ch
            else:
                capacitance = capacitance_5ch
            poly_string = '\\' + poly_tag
            poly_num = np.int(info_list[-3])
            channel_num = np.int(info_list[-1])
            data_to_write = get_data(poly_tag, analysis_tree)
            data_to_write_t = get_dim(poly_tag, analysis_tree)

            tree_write_safe(data_to_write, poly_tag, dim=data_to_write_t, tree=tree)

            if poly_tag[-3] == 'T':
                c_tag = poly_tag.replace('T', 'C')
                data_to_write_c = get_data(c_tag, analysis_tree)
                tree_write_safe(data_to_write_c, c_tag, tree=tree)


def show_tree_node(tag_name, tree):
    pass


def get_data(tag_name, tree):
    data = tree.getNode('\\' + tag_name).getData().data()
    return data


def get_dim(tag_name, tree):
    t = tree.getNode('\\' + tag_name).dim_of().data()
    return t


def get_cal_string(shot_number, channel_id):
    # An integer to identify the polychromator number
    poly_num = np.int(channel_id[5])
    channel_num = np.int(channel_id[-1])
    return_string = 'CAL_'
    if shot_number < 181008002:
        return_string = return_string + '3_CH_POLY_' + str(poly_num) + '_TYPE?_' + str(channel_num)
    elif shot_number < 190403014:
        if poly_num < 3:
            return_string = return_string + '3_CH_POLY_' + str(poly_num) + '_TYPE?_' + str(channel_num)
        else:
            return_string = return_string + '5_CH_FG_POLY_' + str(poly_num) + '_TYPE?_' + str(channel_num)
    else:
        return_string = return_string + '5_CH_HG_POLY_' + str(poly_num) + '_TYPE?_' + str(channel_num)

    return return_string


def quantum_efficiency(l_meters):
    # Avalanche Photodiode quantum efficiency as determined by Kiyong Lee
    # EG&G model C30956E Si-APD detectors
    # takes in lambda in [m]
    # outputs QE in fraction of 1
    l_nm = l_meters*10**9
    Q = (-9.167*10**-5)*l_nm**2 + 0.23855*l_nm - 46.722
    Q = Q/100

    return Q


def window_correction(l_meters, minavgmax):
    # Spectral signature of the window, as measured by SPRED3
    # takes in lambda in [m] **ONLY VALID FOR WAVELENGTHS BETWEEN 675 AND 710 nm**
    # outputs transmission in fraction of 1
    l_nm = l_meters*10**9
    if minavgmax == -1:
        T_ret = -0.00044628*l_nm + 0.95572
    elif minavgmax == 0:
        T_ret = 0.00049617*l_nm + 0.33436
    elif minavgmax == 1:
        T_ret = 0.0013479*l_nm - 0.15607
    else:
        print('***ERROR in window_correction***')
        return(np.NaN)

    return T_ret

def polarizer_correction(l_meters, minavgmax):
    # Spectral signature of the polarizing filter, as measured by SPRED3
    # takes in lambda in [m] **ONLY VALID FOR WAVELENGTHS BETWEEN 675 AND 710 nm**
    # outputs transmission in fraction of 1
    l_nm = l_meters*10**9
    if minavgmax == -1:
        T_ret = 5.1111e-05*l_nm + 0.18994
    elif minavgmax == 0:
        T_ret = 0.00021592*l_nm + 0.091641
    elif minavgmax == 1:
        T_ret = 0.00034135*l_nm + 0.010499
    else:
        print('***ERROR in polarizer_correction***')
        return(np.NaN)

    return T_ret

def get_vacuum_shot_list(shot_number, log_book, number_vacuum_shots):
    """
    Finds nearest vacuum shots for a given Thomson shot.

    Parameters
    ----------
    shot_num
    log_book

    Returns
    -------
    vac_list: a list of vacuum shots
    """
    ts_today = np.int(str(shot_number)[0:6])
    ts_days = np.unique(np.array([np.int(shots[0:6]) for shots in log_book['Shot'].astype(str)]))
    earlier_days = ts_days[np.where(ts_days < ts_today)]
    later_days = ts_days[np.where(ts_days > ts_today)]
    other_days = np.empty((earlier_days.size + later_days.size,), dtype=earlier_days.dtype)

    other_days = interleave(earlier_days[::-1], later_days)
    days_to_check = np.insert(other_days, 0, ts_today)

    vac_list = [str(shot) for shot in log_book['Shot'] if str(shot)[:6] == str(ts_today) and log_book.loc[log_book.Shot == shot, 'Fuel'].array[0] == 'V']

    ctr = 1
    while len(vac_list) < number_vacuum_shots and ctr < len(days_to_check):
        if len(vac_list) == 0:
            ind_insert = 0
        else:
            ind_insert = 1

        vac_list = np.insert(vac_list, ind_insert, [str(shot) for shot in log_book['Shot'] if str(shot)[:6] == str(days_to_check[ctr])
                                                                                    and log_book.loc[log_book.Shot == shot, 'Fuel'].array[0] == 'V'
                                                                                    and log_book.loc[log_book.Shot == shot, 'Mount_Positions'].array[0] ==
                                                                                        log_book.loc[log_book.Shot == shot_number, 'Mount_Positions'].array[0]
                                                                                    and log_book.loc[log_book.Shot == shot, 'Jack_Height'].array[0] ==
                                                                                        log_book.loc[log_book.Shot == shot_number, 'Jack_Height'].array[0]])
        
        ctr = ctr + 1
    vac_list = [np.int(shot_str) for shot_str in vac_list]
    return vac_list[0:number_vacuum_shots]


def slide_trace(current_peak_pulse_time, target_peak_pulse_time, trace, trace_t):
    dt = np.diff(trace_t).mean()
    slid_trace = np.roll(trace, np.int(np.round((target_peak_pulse_time - current_peak_pulse_time)/dt)))
    return slid_trace

def interleave(list1, list2):
    newlist = []
    a1 = len(list1)
    a2 = len(list2)

    for i in range(max(a1, a2)):
        if i < a1:
            newlist.append(list1[i])
        if i < a2:
            newlist.append(list2[i])

    return newlist


def check_for_saturation(sig):
    num_points = len(sig)
    sat_window = np.int(np.around(num_points/200))
    if sat_window < 3:
        sat_window = 3

    for start_point in np.arange(0, num_points - sat_window + 1):
        var_now = np.var(sig[start_point:start_point + sat_window])
        if var_now < 1e-20:
            return(1)
        else:
            pass

    return(0)





# thomson analysis tree variable definitions:
thomson_tree_lookup = pd.DataFrame(data=[['CAL_3_CH_POLY_1_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_1:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_1_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_1:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_1_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_1:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_1_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_1:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_1_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_1:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_1_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_1:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_2_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_2:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_2_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_2:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_2_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_2:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_2_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_2:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_2_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_2:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_2_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_2:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_3_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_3:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_3_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_3:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_3_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_3:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_3_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_3:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_3_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_3:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_3_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_3:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_4_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_4:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_4_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_4:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_4_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_4:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_3_CH_POLY_4_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_4:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_4_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_4:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_4_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_4:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_3_CH_POLY_1_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_1:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_1_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_1:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_1_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_1:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_2_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_2:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_2_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_2:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_2_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_2:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_3_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_3:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_3_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_3:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_3_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_3:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_4_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_4:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_4_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_4:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_3_CH_POLY_4_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.3_CHANNEL.POLY_4:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_3_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_T_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:TRANS_4', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_T_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:TRANS_5', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_V_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:VARIANCE_4', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_V_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:VARIANCE_5', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_3_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_3_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_3_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_3_C_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:CFB_4', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_3_C_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_3:CFB_5', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_4_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_T_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:TRANS_4', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_T_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:TRANS_5', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_V_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:VARIANCE_4', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_V_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:VARIANCE_5', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_4_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_4_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_4_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_4_C_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:CFB_4', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_4_C_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_4:CFB_5', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_5_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_T_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:TRANS_4', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_T_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:TRANS_5', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_V_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:VARIANCE_4', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_V_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:VARIANCE_5', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_FG_POLY_5_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_5_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_5_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_5_C_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:CFB_4', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_FG_POLY_5_C_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_FG.POLY_5:CFB_5', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_3_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_T_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:TRANS_4', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_T_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:TRANS_5', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_V_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:VARIANCE_4', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_V_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:VARIANCE_5', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_3_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_3_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_3_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_3_C_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:CFB_4', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_3_C_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_3:CFB_5', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_4_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_T_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:TRANS_4', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_T_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:TRANS_5', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_V_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:VARIANCE_4', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_V_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:VARIANCE_5', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_4_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_4_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_4_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_4_C_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:CFB_4', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_4_C_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_4:CFB_5', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_5_T_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:TRANS_1', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_T_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:TRANS_2', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_T_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:TRANS_3', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_T_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:TRANS_4', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_T_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:TRANS_5', 'SIGNAL', 'arb', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_V_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:VARIANCE_1', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_V_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:VARIANCE_2', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_V_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:VARIANCE_3', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_V_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:VARIANCE_4', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_V_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:VARIANCE_5', 'SIGNAL', '', 'm'],
                                         ['CAL_5_CH_HG_POLY_5_C_1', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:CFB_1', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_5_C_2', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:CFB_2', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_5_C_3', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:CFB_3', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_5_C_4', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:CFB_4', 'NUMERIC', 'F', ''],
                                         ['CAL_5_CH_HG_POLY_5_C_5', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.SPECTRAL.5_CHANNEL_HG.POLY_5:CFB_5', 'NUMERIC', 'F', ''],
                                         ['POLY_1_LENGTH', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:LENGTH', 'NUMERIC', 'm', ''],
                                         ['POLY_1_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:R', 'NUMERIC', 'm', ''],
                                         ['POLY_1_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:Z', 'NUMERIC', 'm', ''],
                                         ['POLY_1_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:PHI', 'NUMERIC', 'm', ''],
                                         ['POLY_1_R_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:R_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_1_R_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:R_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_1_Z_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:Z_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_1_Z_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:Z_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_1_THETA', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:THETA', 'NUMERIC', 'm', ''],
                                         ['POLY_1_K_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:K_R', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_1_K_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:K_PHI', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_1_K_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:K_Z', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_1_T_E_FIXED', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.BAYES:T_E_FIXED', 'SIGNAL', 'eV', ''],
                                         ['POLY_1_T_E_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.BAYES:T_E_DRIFT', 'SIGNAL', 'eV', ''],
                                         ['POLY_1_V_D_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.BAYES:V_D_DRIFT', 'SIGNAL', 'm/s', ''],
                                         ['POLY_1_SOLID_ANGLE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.GEOMETRY:SOLID_ANGLE', 'NUMERIC', 'm', ''],
                                         ['POLY_2_LENGTH', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:LENGTH', 'NUMERIC', 'm', ''],
                                         ['POLY_2_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:R', 'NUMERIC', 'm', ''],
                                         ['POLY_2_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:Z', 'NUMERIC', 'm', ''],
                                         ['POLY_2_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:PHI', 'NUMERIC', 'm', ''],
                                         ['POLY_2_R_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:R_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_2_R_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:R_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_2_Z_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:Z_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_2_Z_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:Z_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_2_THETA', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:THETA', 'NUMERIC', 'm', ''],
                                         ['POLY_2_K_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:K_R', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_2_K_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:K_PHI', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_2_K_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:K_Z', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_2_SOLID_ANGLE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.GEOMETRY:SOLID_ANGLE', 'NUMERIC', 'm', ''],
                                         ['POLY_2_T_E_FIXED', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.BAYES:T_E_FIXED', 'SIGNAL', 'eV', ''],
                                         ['POLY_2_T_E_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.BAYES:T_E_DRIFT', 'SIGNAL', 'eV', ''],
                                         ['POLY_2_V_D_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.BAYES:V_D_DRIFT', 'SIGNAL', 'm/s', ''],
                                         ['POLY_3_LENGTH', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:LENGTH', 'NUMERIC', 'm', ''],
                                         ['POLY_3_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:R', 'NUMERIC', 'm', ''],
                                         ['POLY_3_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:Z', 'NUMERIC', 'm', ''],
                                         ['POLY_3_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:PHI', 'NUMERIC', 'm', ''],
                                         ['POLY_3_R_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:R_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_3_R_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:R_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_3_Z_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:Z_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_3_Z_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:Z_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_3_THETA', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:THETA', 'NUMERIC', 'm', ''],
                                         ['POLY_3_K_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:K_R', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_3_K_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:K_PHI', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_3_K_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:K_Z', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_3_SOLID_ANGLE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.GEOMETRY:SOLID_ANGLE', 'NUMERIC', 'm', ''],
                                         ['POLY_3_T_E_FIXED', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.BAYES:T_E_FIXED', 'SIGNAL', 'eV', ''],
                                         ['POLY_3_T_E_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.BAYES:T_E_DRIFT', 'SIGNAL', 'eV', ''],
                                         ['POLY_3_V_D_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.BAYES:V_D_DRIFT', 'SIGNAL', 'm/s', ''],
                                         ['POLY_4_LENGTH', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:LENGTH', 'NUMERIC', 'm', ''],
                                         ['POLY_4_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:R', 'NUMERIC', 'm', ''],
                                         ['POLY_4_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:Z', 'NUMERIC', 'm', ''],
                                         ['POLY_4_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:PHI', 'NUMERIC', 'm', ''],
                                         ['POLY_4_R_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:R_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_4_R_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:R_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_4_Z_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:Z_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_4_Z_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:Z_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_4_THETA', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:THETA', 'NUMERIC', 'm', ''],
                                         ['POLY_4_K_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:K_R', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_4_K_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:K_PHI', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_4_K_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:K_Z', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_4_SOLID_ANGLE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.GEOMETRY:SOLID_ANGLE', 'NUMERIC', 'm', ''],
                                         ['POLY_4_T_E_FIXED', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.BAYES:T_E_FIXED', 'SIGNAL', 'eV', ''],
                                         ['POLY_4_T_E_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.BAYES:T_E_DRIFT', 'SIGNAL', 'eV', ''],
                                         ['POLY_4_V_D_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.BAYES:V_D_DRIFT', 'SIGNAL', 'm/s', ''],
                                         ['POLY_5_LENGTH', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:LENGTH', 'NUMERIC', 'm', ''],
                                         ['POLY_5_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:R', 'NUMERIC', 'm', ''],
                                         ['POLY_5_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:Z', 'NUMERIC', 'm', ''],
                                         ['POLY_5_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:PHI', 'NUMERIC', 'm', ''],
                                         ['POLY_5_R_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:R_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_5_R_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:R_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_5_Z_POS', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:Z_POS', 'NUMERIC', 'm', ''],
                                         ['POLY_5_Z_NEG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:Z_NEG', 'NUMERIC', 'm', ''],
                                         ['POLY_5_THETA', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:THETA', 'NUMERIC', 'm', ''],
                                         ['POLY_5_K_R', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:K_R', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_5_K_PHI', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:K_PHI', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_5_K_Z', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:K_Z', 'NUMERIC', 'm^-1', ''],
                                         ['POLY_5_SOLID_ANGLE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.GEOMETRY:SOLID_ANGLE', 'NUMERIC', 'm', ''],
                                         ['POLY_5_T_E_FIXED', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.BAYES:T_E_FIXED', 'SIGNAL', 'eV', ''],
                                         ['POLY_5_T_E_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.BAYES:T_E_DRIFT', 'SIGNAL', 'eV', ''],
                                         ['POLY_5_V_D_DRIFT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.BAYES:V_D_DRIFT', 'SIGNAL', 'm/s', ''],
                                         ['POLY_1_1_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.RAW:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_2_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.RAW:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_3_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.RAW:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_4_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.RAW:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_1_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.RAW:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_2_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.RAW:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_3_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.RAW:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_4_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.RAW:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_5_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.RAW:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_1_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.RAW:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_2_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.RAW:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_3_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.RAW:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_4_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.RAW:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_5_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.RAW:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_1_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.RAW:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_2_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.RAW:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_3_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.RAW:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_4_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.RAW:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_5_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.RAW:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_1_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.RAW:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_2_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.RAW:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_3_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.RAW:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_4_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.RAW:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_5_RAW', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.RAW:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_1_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAC:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_2_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAC:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_3_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAC:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_4_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAC:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_5_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAC:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_1_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAC:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_2_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAC:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_3_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAC:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_4_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAC:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_5_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAC:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_1_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAC:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_2_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAC:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_3_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAC:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_4_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAC:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_5_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAC:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_1_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAC:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_2_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAC:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_3_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAC:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_4_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAC:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_5_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAC:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_1_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAC:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_2_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAC:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_3_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAC:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_4_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAC:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_5_VAC', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAC:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_1_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SIG:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_2_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SIG:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_3_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SIG:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_4_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SIG:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_5_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SIG:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_1_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SIG:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_2_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SIG:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_3_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SIG:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_4_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SIG:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_2_5_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SIG:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_1_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SIG:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_2_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SIG:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_3_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SIG:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_4_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SIG:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_3_5_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SIG:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_1_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SIG:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_2_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SIG:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_3_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SIG:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_4_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SIG:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_4_5_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SIG:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_1_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SIG:CHANNEL_1', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_2_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SIG:CHANNEL_2', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_3_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SIG:CHANNEL_3', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_4_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SIG:CHANNEL_4', 'SIGNAL', 'V', 's'],
                                         ['POLY_5_5_SIG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SIG:CHANNEL_5', 'SIGNAL', 'V', 's'],
                                         ['POLY_1_1_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_1_2_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_1_3_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_1_4_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_1_5_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.SAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_2_1_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_2_2_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_2_3_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_2_4_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_2_5_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.SAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_3_1_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_3_2_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_3_3_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_4_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_3_5_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.SAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_4_1_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_4_2_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_4_3_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_4_4_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_4_5_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.SAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_5_1_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_5_2_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_5_3_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_5_4_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_5_5_SAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.SAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_1_1_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_1_1_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_1_1_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_1_1_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_1_1_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_1_1_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_1_1_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_1_1_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_1_2_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_1_2_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_1_2_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_1_2_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_1_2_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_1_2_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_1_2_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_1_2_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_1_3_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_1_3_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_1_3_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_1_3_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.N_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_1_3_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_1_3_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_1_3_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_1_3_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_1.VAR_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_2_1_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_2_1_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_2_1_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_2_1_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_2_1_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_2_1_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_2_1_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_2_1_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_2_2_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_2_2_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_2_2_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_2_2_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_2_2_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_2_2_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_2_2_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_2_2_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_2_3_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_2_3_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_2_3_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_2_3_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.N_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_2_3_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_2_3_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_2_3_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_2_3_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_2.VAR_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_1_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_3_1_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_3_1_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_3_1_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_3_1_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_3_1_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_3_1_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_3_1_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_3_2_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_3_2_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_3_2_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_3_2_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_3_2_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_3_2_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_3_2_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_3_2_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_3_3_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_3_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_3_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_3_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_3_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_3_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_3_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_3_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_3_4_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_PE:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_3_4_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_STRAY:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_3_4_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_BG:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_3_4_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_SCAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_3_4_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_PE:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_3_4_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_STRAY:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_3_4_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_BG:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_3_4_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_SCAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_3_5_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_PE:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_3_5_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_STRAY:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_3_5_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_BG:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_3_5_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.N_SCAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_3_5_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_PE:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_3_5_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_STRAY:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_3_5_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_BG:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_3_5_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_3.VAR_SCAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_4_1_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_4_1_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_4_1_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_4_1_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_4_1_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_4_1_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_4_1_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_4_1_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_4_2_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_4_2_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_4_2_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_4_2_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_4_2_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_4_2_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_4_2_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_4_2_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_4_3_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_4_3_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_4_3_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_4_3_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_4_3_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_4_3_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_4_3_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_4_3_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_4_4_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_PE:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_4_4_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_STRAY:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_4_4_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_BG:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_4_4_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_SCAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_4_4_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_PE:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_4_4_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_STRAY:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_4_4_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_BG:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_4_4_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_SCAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_4_5_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_PE:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_4_5_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_STRAY:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_4_5_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_BG:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_4_5_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.N_SCAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_4_5_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_PE:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_4_5_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_STRAY:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_4_5_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_BG:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_4_5_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_4.VAR_SCAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_5_1_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_5_1_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_5_1_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_5_1_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_5_1_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_PE:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_5_1_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_STRAY:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_5_1_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_BG:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_5_1_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_SCAT:CHANNEL_1', 'NUMERIC', '', ''],
                                         ['POLY_5_2_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_5_2_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_5_2_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_5_2_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_5_2_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_PE:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_5_2_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_STRAY:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_5_2_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_BG:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_5_2_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_SCAT:CHANNEL_2', 'NUMERIC', '', ''],
                                         ['POLY_5_3_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_5_3_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_5_3_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_5_3_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_5_3_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_PE:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_5_3_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_STRAY:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_5_3_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_BG:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_5_3_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_SCAT:CHANNEL_3', 'NUMERIC', '', ''],
                                         ['POLY_5_4_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_PE:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_5_4_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_STRAY:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_5_4_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_BG:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_5_4_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_SCAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_5_4_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_PE:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_5_4_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_STRAY:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_5_4_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_BG:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_5_4_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_SCAT:CHANNEL_4', 'NUMERIC', '', ''],
                                         ['POLY_5_5_N_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_PE:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_5_5_N_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_STRAY:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_5_5_N_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_BG:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_5_5_N_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.N_SCAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_5_5_VAR_PE', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_PE:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_5_5_VAR_STRAY', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_STRAY:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_5_5_VAR_BG', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_BG:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['POLY_5_5_VAR_SCAT', 'ANALYSIS3::TOP.THOMSON.MEASUREMENTS.POLY_5.VAR_SCAT:CHANNEL_5', 'NUMERIC', '', ''],
                                         ['LASER_E_SLOPE', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.LAS_E_SLP', 'NUMERIC', 'V/J', ''],
                                         ['LASER_E_INT', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.LAS_E_INT', 'NUMERIC', 'V', ''],
                                         ['LASER_E_SLOPE_B', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.LAS_E_SLP_B', 'SIGNAL', 'V/J', ''],
                                         ['LASER_E_INT_B', 'ANALYSIS3::TOP.THOMSON.CALIBRATIONS.LAS_E_INT_B', 'SIGNAL', 'V', ''],
                                         ['ENERGY', 'ANALYSIS3::TOP.THOMSON.LASER:ENERGY', 'NUMERIC', 'J', ''],
                                         ['ENERGY_BAYES', 'ANALYSIS3::TOP.THOMSON.LASER:ENERGY_BAYES', 'NUMERIC', 'J', ''],
                                         ['ENERGY_VAR', 'ANALYSIS3::TOP.THOMSON.LASER:ENERGY_VAR', 'NUMERIC', 'J^2', ''],
                                         ['ENERGY_PD', 'ANALYSIS3::TOP.THOMSON.LASER:ENERGY_PD', 'SIGNAL', 'V', 's'],
                                         ['VACUUM', 'ANALYSIS3::TOP.THOMSON.LASER:VACUUM', 'SIGNAL', 'V', 's'],
                                         ['TRIG_TIME', 'ANALYSIS3::TOP.THOMSON.LASER:TRIG_TIME', 'NUMERIC', 's', ''],
                                         ['FIRE_TIME', 'ANALYSIS3::TOP.THOMSON.LASER:FIRE_TIME', 'NUMERIC', 's', '']],
                                         columns=['Tag', 'Path', 'Usage', 'Units', 'dim_Units'])


def inference_button_fixed(poly_id, tree):
    """
    Perform the Bayesian analysis using the fixed Maxwellian distribution
    
    Parameters
    -----------
    poly_id: something like "POLY_3" to identify which polychromator we're analyzing
    tree: the active analysis tree
    
    
    Returns
    -------
    pmdf, trace: the posterior trace as a dataframe, and the trace itself (which ArViz has trouble with for some reason)
    """
    # For testing:
    # N_in_real = np.array([-31164.3, 3040.38, 1613.666, 460.4418, 219.852])
    # N_in_test = np.array([0, 5.30220575e-11, 2.32446419e-11, 3.03997827e-11, 1.06100367e-11])
    # N_in_test = N_in_real
    # var_in = np.array([3202870.7274, 17794.926, 8707.380, 4142.40, 2409.076])

    analysis_tree = tree

    energy_in = get_data('ENERGY_BAYES', analysis_tree)

    tau = {}
    l_domain = {}
    cal_var = {}
    var_scat = list([])
    N_scat = list([])

    node = analysis_tree.getNode('THOMSON.MEASUREMENTS.' + poly_id + '.N_SCAT')
    angle_scat = get_data(poly_id + '_THETA', analysis_tree)

    channels = np.arange(1, node.getNumDescendants() + 1)
    var_str = poly_id + '_Q_VAR_SCAT'
    sat_ctr = 0
    for channel in channels:
        channel_index = channel - sat_ctr
        if get_data(poly_id + '_' + np.str(channel) + '_SAT', analysis_tree) == 0:
            
            cal_str = get_cal_string(analysis_tree.shot, poly_id + '_' + np.str(channel))
    
            cal_var[channel_index] = get_data(cal_str.replace('TYPE?', 'V'), analysis_tree)
            l_domain[channel_index] = get_dim(cal_str.replace('TYPE?', 'T'), analysis_tree)
            corr_fac = polarizer_correction(l_domain[channel_index], minavgmax=0)*window_correction(l_domain[channel_index], minavgmax=0)
            tau[channel_index] = corr_fac*quantum_efficiency(l_domain[channel_index])*get_data(cal_str.replace('TYPE?', 'T'), analysis_tree)
            var_scat.append(get_data(var_str.replace('Q', np.str(channel)), analysis_tree))
            N_scat.append(get_data(poly_id + '_' + np.str(channel) + '_N_SCAT', analysis_tree))
        else:
            sat_ctr = sat_ctr + 1

    var_in = np.array(var_scat)
    N_in = np.array(N_scat)

    N = N_in/N_in[0]
    var = var_in/(N_in**2)

    # create our Op
    logl = LogLikeWithGrad(fixed_maxwellian_loglike, N, l_domain[2], np.sqrt(var), tau, cal_var, angle_scat)

    # use PyMC3 to sampler from log-likelihood
    with pm.Model() as opmodel:

        # Priors
        t_e = pm.Uniform('t_e', lower=-10, upper=100)
        # n_e = pm.Uniform('n_e', lower=1e16, upper=1e23)
        # c_geom = pm.Uniform('c_geom', lower=0, upper=10)

        # convert parameters to a tensor vector
        # theta = tt.as_tensor_variable([t_e, n_e, c_geom])
        theta = tt.as_tensor_variable([t_e])

        # Sampler

        # step = pm.NUTS(target_accept=0.9)
        # step = pm.NUTS(target_accept=0.5)
        # step = pm.Metropolis()
        # step = None

        # use a DensityDist
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        # trace = pm.sample(10000, step, tune=5000, discard_tuned_samples=True)
        # fixed_trace = pm.sample(3000, tune=9000, discard_tuned_samples=True)
        fixed_trace = pm.sample(tune=5000, draws=20000, discard_tuned_samples=True, step=pm.Metropolis())

        # plot the traces
        pm.summary(fixed_trace)

        pmdf = pm.trace_to_dataframe(fixed_trace)
        # pm.save_trace(trace, 'home/everson/Dropbox/PyThom/traces/fixed' + np.str(analysis_tree.shot) + poly_id)

        # pm.summary(trace)
        print(pm.summary(fixed_trace).to_string())

        # samples_pymc3_2 = np.vstack((trace['t_e'])).T

        if PLOTS_ON == 1:

            # t_e_x = np.arange(0, 100, 0.001)
            # t_e_y = st.norm.u(t_e_x, loc=10, scale=1000)

            plt.figure()
            plt.plot(np.arange(1,len(fixed_trace['t_e'])+1),fixed_trace['t_e'])
            plt.figure()
            sns.kdeplot(fixed_trace.t_e)
            # plt.plot(t_e_x, t_e_y)



        return pmdf, fixed_trace


def inference_button_drift(poly_id, tree):
    """
    Perform the Bayesian analysis using the drifting Maxwellian distribution

    Parameters
    -----------
    poly_id: something like "POLY_3" to identify which polychromator we're analyzing
    tree: the active analysis tree

    Returns
    -------
    pmdf, trace: the posterior trace as a dataframe, and the trace itself (which ArViz has trouble with for some reason)
    """
    # For testing:
    # N_in_real = np.array([-31164.3, 3040.38, 1613.666, 460.4418, 219.852])
    # N_in_test = np.array([0, 5.30220575e-11, 2.32446419e-11, 3.03997827e-11, 1.06100367e-11])
    # N_in_test = N_in_real
    # var_in = np.array([3202870.7274, 17794.926, 8707.380, 4142.40, 2409.076])

    analysis_tree = tree

    energy_in = get_data('ENERGY_BAYES', analysis_tree)

    tau = {}
    l_domain = {}
    cal_var = {}
    var_scat = list([])
    N_scat = list([])

    node = analysis_tree.getNode('THOMSON.MEASUREMENTS.' + poly_id + '.N_SCAT')
    angle_scat = get_data(poly_id + '_THETA', analysis_tree)

    channels = np.arange(1, node.getNumDescendants() + 1)
    var_str = poly_id + '_Q_VAR_SCAT'

    sat_ctr = 0
    for channel in channels:
        channel_index = channel - sat_ctr
        if get_data(poly_id + '_' + np.str(channel) + '_SAT', analysis_tree) == 0:
            
            cal_str = get_cal_string(analysis_tree.shot, poly_id + '_' + np.str(channel))
    
            cal_var[channel_index] = get_data(cal_str.replace('TYPE?', 'V'), analysis_tree)
            l_domain[channel_index] = get_dim(cal_str.replace('TYPE?', 'T'), analysis_tree)
            corr_fac = polarizer_correction(l_domain[channel_index], minavgmax=0)*window_correction(l_domain[channel_index], minavgmax=0)
            tau[channel_index] = corr_fac*quantum_efficiency(l_domain[channel_index])*get_data(cal_str.replace('TYPE?', 'T'), analysis_tree)
            var_scat.append(get_data(var_str.replace('Q', np.str(channel)), analysis_tree))
            N_scat.append(get_data(poly_id + '_' + np.str(channel) + '_N_SCAT', analysis_tree))
        else:
            sat_ctr = sat_ctr + 1

    var_in = np.array(var_scat)
    N_in = np.array(N_scat)

    N = N_in/N_in[0]
    var = var_in/(N_in**2)



    # create our Op
    logl = LogLikeWithGrad(drift_maxwellian_loglike, N, l_domain[2], np.sqrt(var), tau, cal_var, angle_scat)

    # use PyMC3 to sampler from log-likelihood
    with pm.Model() as opmodel:

        # Priors
        t_e = pm.Uniform('t_e', lower=-10, upper=100)
        v_d = pm.Uniform('v_d', lower=-5e6, upper=5e6)
        # n_e = pm.Uniform('n_e', lower=1e16, upper=1e23)
        # c_geom = pm.Uniform('c_geom', lower=0, upper=10)

        # convert parameters to a tensor vector
        # theta = tt.as_tensor_variable([t_e, n_e, c_geom])
        theta = tt.as_tensor_variable([t_e, v_d])

        # Sampler

        # step = pm.NUTS(target_accept=0.9)
        # step = pm.NUTS(target_accept=0.5)
        # step = pm.Metropolis()
        # step = None

        # use a DensityDist
        pm.DensityDist('likelihood', lambda v: logl(v), observed={'v': theta})
        # trace = pm.sample(10000, step, tune=5000, discard_tuned_samples=True)
        # drift_trace = pm.sample(2000, tune=9000, discard_tuned_samples=True)
        drift_trace = pm.sample(tune=5000, draws=20000, discard_tuned_samples=True, step=pm.Metropolis())
        # plot the traces
        # pm.summary(trace)

        pmdf = pm.trace_to_dataframe(drift_trace)

        print(pm.summary(drift_trace).to_string())
        
        # pm.save_trace(trace, 'home/everson/Dropbox/PyThom/traces/drift' + np.str(analysis_tree.shot) + poly_id)
        # sns.kdeplot(pmdf.t_e)

        # samples_pymc3_2 = np.vstack((trace['t_e'])).T
        if PLOTS_ON == 1:
            plt.figure()
            plt.plot(np.arange(1,len(drift_trace['t_e'])+1),drift_trace['t_e'])
            plt.figure()
            sns.kdeplot(drift_trace.t_e)
            plt.figure()
            plt.plot(np.arange(1,len(drift_trace['v_d'])+1),drift_trace['v_d'])
            plt.figure()
            sns.kdeplot(drift_trace.v_d)

        return pmdf, drift_trace


    
# define the model
def fixed_maxwellian(theta, tau, l_domain, cal_var, angle_scat):

    # t_e, n_e, c_geom = theta  # unpack parameters
    t_e = theta  # unpack parameters
    beta = C_SPEED*np.sqrt(ELECTRON_MASS)/(2*RUBY_WL*np.sin(angle_scat/2)*np.sqrt(2*E_CHARGE))
    l_unc = np.random.normal(0, 2e-10)
    tau_h_int = list([])
    var_spec = list([])
    h_fm = np.exp(-(beta**2)*(l_domain + l_unc - RUBY_WL)**2/t_e)/np.sqrt(t_e)
    for ii in np.arange(1,len(tau) + 1):
        # ret.append((SIGMA_TS/h_PLANCK)*np.sqrt(ELECTRON_MASS/(2*np.pi*E_CHARGE))*1e19*1e-5*np.trapz(tau[ii]*np.exp(-(beta**2)*(l - RUBY_WL)**2/t_e)/np.sqrt(t_e), l))
        tau_h_int.append(np.trapz(tau[ii]*h_fm, l_domain))
        var_spec.append(np.trapz((cal_var[ii])*(h_fm)**2, l_domain))

    return np.array(tau_h_int/(tau_h_int[0] + 1e-30)), var_spec


# define the likelihood function
def fixed_maxwellian_loglike(theta, l_domain, data, sigma, tau, cal_var, angle_scat):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """
    data = np.array(data)
    model, var_spec = fixed_maxwellian(theta, tau, l_domain, cal_var, angle_scat)
    # model = np.flipud(model)
    # print(data)
    llh = -0.5*np.log((1/(np.sqrt(2*np.pi)**len(sigma)*np.prod(np.sqrt(sigma**2 + var_spec)))))*np.sum((data - model)**2/((sigma**2 + var_spec)))

    # print(llh)
    return llh



# define the model
def drift_maxwellian(theta, tau, l_domain, cal_var, angle_scat):

    # t_e, n_e, c_geom = theta  # unpack parameters
    t_e, v_d = theta  # unpack parameters
    l_unc = np.random.normal(0, 2e-10)
    dl = np.mean(np.diff(l_domain))
    beta = C_SPEED*np.sqrt(ELECTRON_MASS)/(2*RUBY_WL*np.sin(angle_scat/2)*np.sqrt(2*E_CHARGE))
    tau_h_int = list([])
    var_spec = list([])
    h_dm = np.exp(-((beta)*(l_domain + l_unc - RUBY_WL) - v_d*np.sqrt(ELECTRON_MASS/(2*E_CHARGE)))**2/t_e)/np.sqrt(t_e)
    for ii in np.arange(1,len(tau) + 1):
        # ret.append((SIGMA_TS/h_PLANCK)*np.sqrt(ELECTRON_MASS/(2*np.pi*E_CHARGE))*1e19*1e-5*np.trapz(tau[ii]*np.exp(-(beta**2)*(l - RUBY_WL)**2/t_e)/np.sqrt(t_e), l))
        tau_h_int.append(np.trapz(tau[ii]*h_dm, l_domain))
        var_spec.append(np.trapz((cal_var[ii])*(h_dm)**2, l_domain))
    return np.array(tau_h_int/(tau_h_int[0] + 1e-30)), var_spec


# define the likelihood function
def drift_maxwellian_loglike(theta, l_domain, data, sigma, tau, cal_var, angle_scat):
    """
    A Gaussian log-likelihood function for a model with parameters given in theta
    """
    data = np.array(data)
    model, var_spec = drift_maxwellian(theta, tau, l_domain, cal_var, angle_scat)
    # model = np.flipud(model)
    # print(data)
    llh = -0.5*np.log((1/(np.sqrt(2*np.pi)**len(sigma)*np.prod(np.sqrt(sigma**2 + var_spec)))))*np.sum((data - model)**2/((sigma**2 + var_spec)))
    # print(llh)
    return llh


class LogLikeWithGrad(tt.Op):

    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma, tau, cal_var, angle_scat):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma
        self.tau = tau
        self.cal_var = cal_var
        self.angle_scat = angle_scat

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.likelihood, self.data, self.x, self.sigma, self.tau, self.cal_var, self.angle_scat)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta, = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma, self.tau, self.cal_var, self.angle_scat)

        outputs[0][0] = np.array(logl) # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs  # our parameters
        return [g[0]*self.logpgrad(theta)]


class LogLikeGrad(tt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike, data, x, sigma, tau, cal_var, angle_scat):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma
        self.tau = tau
        self.cal_var = cal_var
        self.angle_scat = angle_scat

    def perform(self, node, inputs, outputs):
        theta, = inputs

        # define version of likelihood function to pass to derivative function
        def lnlike(values):
            return self.likelihood(values, self.x, self.data, self.sigma, self.tau, self.cal_var, self.angle_scat)

        # calculate gradients
        # grads = gradients(theta, lnlike)
        grads = approx_fprime(theta, lnlike, epsilon=1)
        # grads = approx_fprime(theta, lnlike, epsilon=100)
        

        outputs[0][0] = grads


if __name__ == '__main__':
    # analyze_plasma(start=190501005, stop=190501008)
    analyze_shot(190516017)
    pass





