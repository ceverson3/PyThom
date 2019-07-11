#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:41:30 2019

@author: everson
"""

import MDSplus
import numpy as np

shots = np.loadtxt('hitsi3_all_shots.txt',dtype=np.int)


for shot in shots:



	t = MDSplus.Tree('analysis3',shot,'EDIT')
	t.addNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1','SIGNAL').addTag("N_C1")
	t.addNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.COMMENT','TEXT')
	t.addNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.FILT_BW','NUMERIC').addTag("C1_FILT_BW")
	t.addNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.FILT_CENTER','NUMERIC').addTag("C1_FILT_CENTER")
	t.addNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.SCENE_FREQ','NUMERIC').addTag("C1_FREQ")
	t.addNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.P_ERR','SIGNAL').addTag("N_C1_P_ERR")
	t.addNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.N_ERR','SIGNAL').addTag("N_C1_N_ERR")
	t.addNode(r'\ANALYSIS3::TOP.DENSITY.FIR.REF_FREQ','NUMERIC').addTag("REF_FREQ")
	t.addNode(r'\ANALYSIS3::TOP.DENSITY.FIR.REF_FILT_BW','NUMERIC').addTag("REF_FILT_BW")

	# t.deleteNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1')
	# t.deleteNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.COMMENT')
	# t.deleteNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.FILT_BW')
	# t.deleteNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.FILT_CENTER')
	# t.deleteNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.SCENE_FREQ')
	# t.deleteNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.P_ERR')
	# t.deleteNode(r'\ANALYSIS3::TOP.DENSITY.FIR.N_AVG_C1.N_ERR')
	# t.deleteNode(r'\ANALYSIS3::TOP.DENSITY.FIR.REF_FREQ')
	# t.deleteNode(r'\ANALYSIS3::TOP.DENSITY.FIR.REF_FILT_BW')




	t.write()
