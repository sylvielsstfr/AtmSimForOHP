# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import math as m
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from hapi import *


atm_ground_file='atm_OHP_ground.csv'
atm_groundsum_file='atm_OHP_groundsum.csv'

WLMIN=200.  # nm
WLMAX=1200. # nm

NUMIN=1e7/WLMAX  # cm-1
NUMAX=1e7/WLMIN  # cm-1

Pa_to_Atm = 0.00000986923267
hPa_to_Atm=Pa_to_Atm*100  # libRadTran provides pressure in hPa
Atm_to_Pa = 1.01325*1e5 
Atm_to_hPa = 1.01325*1e3 


class Hitran:
    """
    """
    def __init__(self):
        db_begin('data')
        fetch_by_ids('O2',[36,37,38],NUMIN,NUMAX)
        fetch_by_ids('H2O',[1,2,3,4,5,6],NUMIN,NUMAX)
    def getO2(self,P,T):
        #fetch_by_ids('O2',[36,37,38],NUMIN,NUMAX)
        nu,coef = absorptionCoefficient_Lorentz(SourceTables='O2',OmegaStep=0.01,Environment={'p': P,'T': T})
        return nu,coef
    def getH2O(self,P,T):
        #fetch_by_ids('H2O',[1,2,3,4,5,6],NUMIN,NUMAX)
        nu,coef = absorptionCoefficient_Lorentz(SourceTables='H2O',OmegaStep=0.01,Environment={'p': P,'T': T})
        return nu,coef



def PlotCoef(nu1,c1,nu2,c2,nu3,c3,titlename):
    plt.figure()
    plt.plot(nu1,c1,'k-',label='us')
    plt.plot(nu2,c2,'b-',label='mw')
    plt.plot(nu3,c3,'r-',label='ms')
    
    plt.title(titlename)
    plt.xlabel('wavenumber in $cm^{-1}$')
    plt.ylabel('absorption coefficient $cm^{2}/molecule $')
    plt.legend(loc='best')
    
    

#####################################################################
# The program simulation start here
#   Variation of precipitable water
####################################################################

if __name__ == "__main__":
    
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    
 
    
    
    # retrieve atmospheric data on Temeprature, Pressure and Air density, and species densities 
    #----------------------------------------------------------------------------------------------
    df=pd.read_csv(atm_ground_file)


    print(df)
    
    d_O2_us=df['O2'][0]  # molec/cm^3 : H2O density  for US standard atmosphere at ground
    d_O2_mw=df['O2'][1] # molec/cm^3 : H2O density  for Mid latitude winter atmosphere at ground
    d_O2_ms=df['O2'][2] # molec/cm^3 : H2O density  for Mid latitude summer atmosphere at ground


    P_us=df['P'][0] # Pressure at ground in hPa  for US standard atmosphere at ground
    P_mw=df['P'][1] # Pressure at ground in hPa for Mid latitude winter atmosphere at ground
    P_ms=df['P'][2] # Pressure at ground in hPa for Mid latitude summer atmosphere at ground
    
    T_us=df['T'][0] # Temperature at ground in hPa  for US standard atmosphere at ground
    T_mw=df['T'][1] # Temperature at ground in hPa for Mid latitude winter atmosphere at ground
    T_ms=df['T'][2] # Temperature at ground in hPa for Mid latitude summer atmosphere at ground
    
    # Convert Pressure from hPa to atmosphere
    Patm_us = P_us*hPa_to_Atm
    Patm_mw = P_mw*hPa_to_Atm
    Patm_ms = P_ms*hPa_to_Atm


  
    # Initialisation of Hapi/Hitran
    # ---------------------------------
    hitr=Hitran()
    
    
    nu_us_o2,coef_us_o2=hitr.getO2(Patm_us,T_us)
    nu_mw_o2,coef_mw_o2=hitr.getO2(Patm_mw,T_mw)
    nu_ms_o2,coef_ms_o2=hitr.getO2(Patm_ms,T_ms)
    
    PlotCoef(nu_us_o2,coef_us_o2,nu_mw_o2,coef_mw_o2,nu_ms_o2,coef_ms_o2,'absorption coefficient for O2')
    
    
    
    nu_us_h2o,coef_us_h2o=hitr.getH2O(Patm_us,T_us)
    nu_mw_h2o,coef_mw_h2o=hitr.getH2O(Patm_mw,T_mw)
    nu_ms_h2o,coef_ms_h2o=hitr.getH2O(Patm_ms,T_ms)
    
    PlotCoef(nu_us_h2o,coef_us_h2o,nu_mw_h2o,coef_mw_h2o,nu_ms_h2o,coef_ms_h2o,'absorption coefficient for H2O')
    
    
    
    
    
    
    
    
   