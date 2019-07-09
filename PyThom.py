#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:40:22 2019

@author: everson
"""
# imports
import matplotlib.pyplot as plt # plotting
import numpy as np              # math
import pandas as pd             # large 2D dataframe operations
import seaborn as sns           # colors!
from io import BytesIO          # reading the TS logbook 
import requests                 # grabbing the TS logbook from google sheets


class Shot:
    def __init__(self,sN):
        self.shotNum = sN
        pass

    def logInfo(self, parameter_list):
        pass

    def preconditionData(self, parameter_list):
        pass

def get_TS_logbook():
#     get the logbook spreadsheet from google sheets and read it in to a pandas dataframe
    r = requests.get('https://docs.google.com/spreadsheet/ccc?key=1yF7RMpYl_KvZPoEwYiH1D_vVrasaTGLuPvdBoYRRM84&output=csv')
    data = r.content
    df = pd.read_csv(BytesIO(data),usecols=np.arange(1,9))
    df = df.dropna(subset=['Shot','Plasma Species'])
    df_nodupes = df.drop_duplicates(subset=['Shot'])
    
    if(df.shape[0] != df_nodupes.shape[0]):
        print("*****     WARNING! Duplicates detected while parsing TS logbook shots:     *****")
        dupes = df.loc[df.duplicated(subset=['Shot'])]['Shot'].astype(int)
        print(dupes.to_csv(sep='\t', index=False))
        print("...excluding duplicates from parsed logbook.")
    
    df = df_nodupes
    
    df[[df.columns[0]]] = df[[df.columns[0]]].apply(pd.to_numeric,errors='coerce',downcast='integer')
    
    return(df)
    
#    df[[df.columns[3]]].plot.hist(bins=50)
#     particular shot row:
#    sn = df.where(df['Shot'] == 190624005).dropna()
    
    
    
#    df.dropna()[[df.columns[3]]].plot.hist(bins=100)