#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:20:07 2019

@author: chris
"""

# imports
import matplotlib.pyplot as plt # plotting
import numpy as np              # math
import pandas as pd             # large 2D dataframe operations
import seaborn as sns           # colors!
from io import BytesIO          # reading the TS logbook 
import requests                 # grabbing the TS logbook from google sheets

# get the logbook spreadsheet from google sheets and read it in to a pandas dataframe
r = requests.get('https://docs.google.com/spreadsheet/ccc?key=1yF7RMpYl_KvZPoEwYiH1D_vVrasaTGLuPvdBoYRRM84&output=csv')
data = r.content
df = pd.read_csv(BytesIO(data),usecols=np.arange(1,9))



a = df.head()
print(df.shape)
print(a)
#df.dropna()[[df.columns[3]]].plot.hist(bins=100)