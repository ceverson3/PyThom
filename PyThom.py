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
from MDSplus import Connection
from MDSplus import Tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

PLOTS_ON = 1
style = 'Freq'  # 'Bayes' for Bayesian analysis or 'Freq' for frequentist/ratio evaluation method
sns.set()   # Set the plotting theme


def analyze_shot(shot_number):
    """
    Analyze the Thomson data for shot_number
    ----------
    Does analysis in a frugal way that avoids duplicate analysis by writing to tree
    any incremental results to the analysis3 tree

    Returns
    -------
    """
    # The logbook is the dataframe-style ledger for human-readable summary TS data
    #   parameters used in the analysis code are found and written here
    log_book = get_ts_logbook()

    #

    # calculate and store the laser energy

    # calculate and store the electron energy distribution function,
    # including temperature, density, and electron drift velocity
    pass


def get_ts_logbook(force_update=None):
    """
    Pull in the logbook information from the TS_Logbook Google sheet, parse it, and return a version of it
    for use in the TS analysis.
    If
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
    record_filename = 'TS_logbook_autogen_' + datetime.date.today().isoformat() + '.csv'
    latest_filename = 'TS_logbook_autogen.csv'

    try:
        latest_ts_log = pd.read_csv(latest_filename)
        if latest_ts_log['Shot'].iloc[-1] == raw_ts_log['Shot'].iloc[-1] and force_update == None:
            print('Using previously updated logbook...')
            return latest_ts_log
        else:
            pass
    except FileNotFoundError:
        pass
    print('Updating logbook...')

    # clean up and start the dataframe to return
    generated_ts_log = raw_ts_log[['Shot', 'TS_TRIG (ms)', 'Energy', 'Plasma Species', 'Notes']]
    generated_ts_log = generated_ts_log.rename(index=str, columns={'Plasma Species': 'Fuel'})
    generated_ts_log = generated_ts_log.rename(index=str, columns={'TS_TRIG (ms)': 'TS_TRIG'})
    generated_ts_log['Shot'] = generated_ts_log.Shot.astype('int64')
    generated_ts_log['Fuel'].replace({'-': 'V'}, inplace=True)

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
    polychromator_parse_dataframe.rename(columns=lambda x: 'poly'+x[0]+'_'+x[1], inplace=True)
    polychromator_parse_dataframe.reset_index(inplace=True, drop=True)
    generated_ts_log = pd.concat([generated_ts_log, polychromator_parse_dataframe], axis=1)

    # calculate and store the geometry data for the shot using the lab jack height and mount positions string
    geometry_parse_dataframe = pd.DataFrame()

    for ind in np.arange(raw_ts_log.shape[0]):
        fiber_mount_string = raw_ts_log['Fiber Mount Positions (in order polys. 12345)'].iloc[ind]
        jack_height = raw_ts_log['Lab jack height (in.)'].iloc[ind]
        geometry_oneshot_dataframe = get_spot_geometry(fiber_mount_string, jack_height)
        r_oneshot_dataframe = pd.DataFrame(data=[geometry_oneshot_dataframe['r'].array],
                                           columns=['radius_poly_' + str(p) for p in geometry_oneshot_dataframe['polychromator']])
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
        except:
            print('No i_tor data for shot '+str(shot))
            i_tor_max[ctr] = np.nan
        ctr = ctr + 1

    max_i_tor_dataframe = pd.DataFrame(data=i_tor_max, columns=['i_tor_max'])
    generated_ts_log = pd.concat([generated_ts_log, max_i_tor_dataframe], axis=1)

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

    geom_dataframe = pd.DataFrame(columns=['polychromator', 'length', 'r', 'phi', 'z',
                                           'r_pos', 'r_neg', 'z_pos', 'z_neg',
                                           'view_angle', 'f_number', 'solid_angle'])

    polychromator = 0
    for fiber_mount_character in fiber_mount_list:
        polychromator = polychromator + 1
        if fiber_mount_character == 'X':
            continue

        spot_bottom = spot_low_edge[fiber_mount_character][spot_low_edge.jack_height == jack_height_in].to_numpy() - origin_measurements
        spot_top = spot_high_edge[fiber_mount_character][spot_high_edge.jack_height == jack_height_in].to_numpy() - origin_measurements

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
        angle_scattering_vector_k_ruby = np.arctan(abs(spot_window_offset)/window_z)

        if spot_window_offset < 0:
            view_angle = np.pi/2 - angle_scattering_vector_k_ruby
        else:
            view_angle = np.pi/2 + angle_scattering_vector_k_ruby

        window_diameter_cm = 3.75*2.54
        phi = np.arctan(raw_center/r_laser) + np.pi/4
        area_lens = np.pi*(window_diameter_cm/2)**2
        solid_angle = area_lens/r_scattering_vector**2
        f_number = r_scattering_vector/(window_diameter_cm)

        geom_dataframe = pd.concat([geom_dataframe, pd.DataFrame(data=[[polychromator, length/100, r/100, phi, z/100,
                                                                       r_pos/100, r_neg/100, z_pos/100, z_neg/100,
                                                                       view_angle, f_number, solid_angle]],
                                                                 columns=['polychromator', 'length', 'r', 'phi', 'z',
                                                                          'r_pos', 'r_neg', 'z_pos', 'z_neg',
                                                                          'view_angle', 'f_number', 'solid_angle'])],
                                                                 ignore_index=True, sort=False)

        # TODO: add TS scattering parameter to the returned values

    return(geom_dataframe)


def update_energy_cal(log_book):
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

    #    LB = get_TS_logbook()
    energy_measured = log_book.Energy[log_book.Fuel == 'ETEST']
    energy_integrated = pd.Series()
    for shot in log_book.Shot[log_book.Fuel == 'ETEST']:
        hit_conn.openTree("hitsi3", shot)
        try:
            flux_photodiode = np.array(hit_conn.get("\\TS_RUBY"))
            flux_photodiode_t = np.array(hit_conn.get("DIM_OF(\\TS_RUBY)"))
        except EOFError:
            print("WARNING: Error reading photodiode data from shot", shot)
            return -1
            pass

        flux_baseline = np.mean(flux_photodiode[0:np.int(np.around(np.size(flux_photodiode, 0)/6))])
        #        flux_baseline = np.mean(flux_photodiode[-np.int(np.around(np.size(flux_photodiode,0)/8)):])
        #        flux_baseline = 0
        flux_photodiode = flux_photodiode - flux_baseline

        energy_integrated = energy_integrated.append(pd.Series([np.trapz(flux_photodiode, flux_photodiode_t)]), ignore_index=True)

    if style == 'Freq':

        # A = np.transpose(np.array([energy_measured, (np.ones_like(energy_measured))]))
        # m, c = np.linalg.lstsq(A, energy_integrated,rcond=None)[0]
        energy_integrated = energy_integrated.to_numpy().reshape(-1, 1)
        energy_measured = energy_measured.to_numpy().reshape(-1, 1)

        # Model initialization
        regression_model = LinearRegression()

        # Fit the data(train the model)
        regression_model.fit(energy_measured, energy_integrated)

        # Predict
        energy_predicted = regression_model.predict(energy_measured)

        # model evaluation
        rmse = mean_squared_error(energy_integrated, energy_predicted)
        r2 = r2_score(energy_integrated, energy_predicted)
        m = regression_model.coef_
        c = regression_model.intercept_
    elif style == 'Bayes':
        pass
    else:
        print('****Pick a style that"s either Bayes or Freq****')

    if PLOTS_ON == 1:
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
        ax1.plot(np.arange(0, 10), regression_model.predict(np.arange(0, 10).reshape(-1, 1))/m, label='Fitted line')
        ax1.plot(np.arange(0, 10), np.arange(0, 10), color='k', ls='--', linewidth=0.5)
        ax1.legend()
        ax1.grid(ls='--')

        print(1/m)

#        fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figures
#        ax2.plot(flux_photodiode_t,flux_photodiode)
#        ax3.scatter(energy_measured, energy_integrated/m)






