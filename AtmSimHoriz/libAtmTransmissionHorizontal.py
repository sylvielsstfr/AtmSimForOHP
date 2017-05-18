# -*- coding: utf-8 -*-
"""
libAtmTransmissionHorizontal.py
================================

- Author Syvie Dagoret-Campagne
- Affiliation : LAL/IN2P3/CNRS
- date : May 15th 2017
- update :

library to compute Horizontal Air transparency at OHP.
It uses 
- Hitran absorption and cross-section.
- libRadTran air densities for US,MW,MS atmospheric profiles

"""

import os
import math as m
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from astropy.constants import N_A,R,g0
from astropy import units as u
from hapi import *


atm_ground_file='atm_OHP_ground.csv'
atm_groundsum_file='atm_OHP_groundsum.csv'

WLMIN=200.  # nm
WLMAX=1200. # nm

NUMIN=1e7/WLMAX  # cm-1
NUMAX=1e7/WLMIN  # cm-1
NURES=100.0 # cm-1

Pa_to_Atm = 0.00000986923267
hPa_to_Atm=Pa_to_Atm*100  # libRadTran provides pressure in hPa
Atm_to_Pa = 1.01325*1e5 
Atm_to_hPa = 1.01325*1e3 


# For the input data file on Cross-Section, profile the path of the file
NO2_XSecFiles=['NO2_220.0_0.0_15002.0-42002.3_00.xsc','NO2_294.0_0.0_15002.0-42002.3_00.xsc']
O3Huggins_XSecFiles=['O3_200.0_0.0_29164.0-40798.0_04.xsc','O3_220.0_0.0_29164.0-40798.0_04.xsc',
'O3_240.0_0.0_29164.0-40798.0_04.xsc','O3_260.0_0.0_29164.0-40798.0_04.xsc',
'O3_280.0_0.0_29164.0-40798.0_04.xsc', 'O3_300.0_0.0_29164.0-40798.0_04.xsc']
O3Chappuis_XSecFiles='O3Chapuis/O3chapuis.txt'


libradtran_file_oh='LibRadTran/data/RT_OH_pp_us_sa_rt_z10_wv40.OUT'
libradtran_file_ls='LibRadTran/data/RT_LS_pp_us_sa_rt_z10_wv40.OUT'

libradtran_file_oh_q5='LibRadTran/data/UVSPEC_REPTRAN_SOLAR_ALT06_COARSE_qpwv5_10.out'
libradtran_file_oh_q10='LibRadTran/data/UVSPEC_REPTRAN_SOLAR_ALT06_COARSE_qpwv10_10.out'
libradtran_file_oh_q50='LibRadTran/data/UVSPEC_REPTRAN_SOLAR_ALT06_COARSE_qpwv50_10.out'
libradtran_file_oh_q90='LibRadTran/data/UVSPEC_REPTRAN_SOLAR_ALT06_COARSE_qpwv90_10.out'
libradtran_file_oh_q95='LibRadTran/data/UVSPEC_REPTRAN_SOLAR_ALT06_COARSE_qpwv95_10.out'



# Few constants
M_air= 28.965338*u.g/u.mol
M_air_dry=28.9644*u.g/u.mol
M_h2o=18.016*u.g/u.mol

P0=101325.*u.Pa;   # /*!< Pa : pressure at see level */
T0=288.15*u.K;   #/*!< sea level temperature */  
L=0.0065*u.K/u.m  # refroidissement en fonction de l'altitude


# OHP
distance=200*u.m
altitude=650*u.m
Distance_source_tel=200*u.m/(1*u.cm)

VAOD_quantiles=np.array([0.03,0.0455,0.09,0.244,0.322]) # MERRA2

class Hitran:
    """
    Class Hitran : class tool to retrieve absorption lines in Hitran Database using
    Hapi interface
    """
    fetch_O2_flag=False
    fetch_H2O_flag=False
    fetch_CO2_flag=False
    fetch_NO2_flag=False
    fetch_O3_Huyggins_flag=False
    fetch_O3_Chapuis_flag=False
    
   
    
    def __init__(self):
        db_begin('data')
        #fetch_by_ids('O2',[36,37,38],NUMIN,NUMAX)
        #fetch_by_ids('H2O',[1,2,3,4,5,6],NUMIN,NUMAX)
    def getO2(self,P,T):
        if not self.fetch_O2_flag:
            #fetch_by_ids('O2',[36,37,38],NUMIN,NUMAX)
            self.fetch_O2_flag=True
        nu,coef = absorptionCoefficient_Lorentz(SourceTables='O2',OmegaStep=0.01,Environment={'p': P,'T': T})
        return nu,coef
    def getH2O(self,P,T):
        if not self.fetch_H2O_flag:
            #fetch_by_ids('H2O',[1,2,3,4,5,6],NUMIN,NUMAX)
            self.fetch_H2O_flag=True
        nu,coef = absorptionCoefficient_Lorentz(SourceTables='H2O',OmegaStep=0.01,Environment={'p': P,'T': T})
        return nu,coef
    def getCO2(self,P,T):
        if not self.fetch_CO2_flag:
            #fetch_by_ids('CO2',[7,8,9,10,11,12,13,14,15],NUMIN,NUMAX)
            self.fetch_CO2_flag=True
        nu,coef = absorptionCoefficient_Lorentz(SourceTables='CO2',OmegaStep=0.01,Environment={'p': P,'T': T})
        return nu,coef
    def getNO2(self,P,T):
        if not self.fetch_NO2_flag:
            #fetch_by_ids('CO2',[7,8,9,10,11,12,13,14,15],NUMIN,NUMAX)
            self.fetch_NO2_flag=True        
        file=os.path.join('NO2Xsec',NO2_XSecFiles[1])
        df=pd.read_table(file,delimiter=' ',usecols=[1,2,3,4,5,6,7,8,9,10]) 
        arr=df.values.flatten()
        arr = arr[~np.isnan(arr)]
        wavenum=np.linspace(NUMIN,NUMAX,arr.shape[0])
        nu,coef =  wavenum,arr
        return nu,coef
    
        
    
    def getO3Huggins(self,P,T):  
        file=os.path.join('O3Xsec',O3Huggins_XSecFiles[5]) # T=300k
        df=pd.read_table(file,delimiter=' ',usecols=[1,2,3,4,5,6,7,8,9,10]) 
        arr=df.values.flatten()
        arr = arr[~np.isnan(arr)]
        NUMIN_O3=29164.0
        NUMAX_O3=40798.0
        wavenum=np.linspace(NUMIN_O3,NUMAX_O3,arr.shape[0])
        nu,coef =  wavenum,arr
        return nu,coef
    
    def getO3Chappuis(self,P,T): 
        data_rto3=np.loadtxt(O3Chappuis_XSecFiles,skiprows=6)
        wl_o3=data_rto3[:,1]
        xsecChapuis_o3=data_rto3[:,2]*1e-21
        return wl_o3,xsecChapuis_o3
    
#--------------------------------------------------------------------------------------    
    
class Rayleigh:
    """
    class Rayleigh :
    tools to compute the rayleigh scattering
    """
    def Pressure(self,h):
        P=P0*np.exp(g0*M_air_dry/R/L*np.log(1-L*h/T0))
        return P
    def XDepth(self,altitude,costh=1):
        """
        Function : XDepth(altitude,costh)
        Provide the column depth in gr / cm^2 equivalent of airmass in physical units
        - Input :  altitude : input altitude in meters
        - Input :  costh    : input cosimus of zenith angle 
        - Output :  XDepth  : output column depth in gr per cm squared
        """
        h=altitude
        XD=self.Pressure(h)/g0/costh
        return XD
        
        
    def RayOptDepth(self,wavelength, altitude=0*u.m, costh=1):
        """
        Function RayOptDepth(double wavelength, double altitude, double costh)
        Provide Rayleigh optical depth
         
        - Input  wavelength : input wavelength in nm
        - Input  altitude : input altitude in meters
        - Input   costh    : input cosimus of zenith angle 
        - Output  OptDepth  : output optical depth no unit, for Rayleigh
        """

        h=altitude

        A=(self.XDepth(h,costh)/(3102.*u.g/(u.cm*u.cm)))
        B=np.exp(-4.*np.log(wavelength/(400.*u.nm)))  
        C= 1-0.0722*np.exp(-2*np.log(wavelength/(400.*u.nm)))

        OD=A*B/C
        #double OD=XDepth(altitude,costh)/2970.*np.power((wavelength/400.),-4);
        return OD
        
    def RayOptDepthXD(self,wavelength, xdepth):
        """
        Function RayOptDepthXD(wavelength, xdepth)
        Provide Rayleigh optical depth
         
        - Input  wavelength : input wavelength in nm
        - Input   xdepth   : depth
        - Output  OptDepth  : output optical depth no unit, for Rayleigh
        """
        A=xdepth/(3102.*u.g/(u.cm*u.cm))
        B=np.exp(-4.*np.log(wavelength/(400.*u.nm)))  
        C= 1-0.0722*np.exp(-2*np.log(wavelength/(400.*u.nm)))
        OD=A*B/C       
        return OD
#-----------------------------------------------------------------------------------
#         AEROSOLS
#--------------------------------------------------------------------------------------      

def VAeroOptDepth(wavelength,tau_aerosols_550=0.10,alpha_ang=1.22) :
        """
        VAeroOptDepth(wavelength, alpha)

        Provide Vertical Aerosols optical depth
        - Input : wavelength : input wavelength in nm
        - Input : alpha : Angstrom exponent
        - Output : OptDepth  : output optical depth no unit for aerosols
    
        """

        OD=tau_aerosols_550*np.exp(-alpha_ang*np.log(wavelength/(550*u.nm)))
        return OD
  
def HAeroOptDepth(wavelength,L,HP,tau_aerosols_550=0.10,alpha_ang=1.22) :
        """
        VAeroOptDepth(wavelength, alpha)

        Provide Vertical Aerosols optical depth
        - Input : wavelength : input wavelength in nm
        - Input : alpha : Angstrom exponent
        - Output : OptDepth  : output optical depth no unit for aerosols
    
        """    
    
        VAOD=((L/HP).decompose())*VAeroOptDepth(wavelength,tau_aerosols_550,alpha_ang)
        
        return VAOD

#--------------------------------------------------------------------------------------




#        
PlotFlag=False


def PlotCoef(nu1,c1,nu2,c2,nu3,c3,titlename):
    plt.figure()
    plt.plot(nu1,c1,'k-',label='us')
    plt.plot(nu2,c2,'b-',label='mw')
    plt.plot(nu3,c3,'r-',label='ms')
    
    plt.title(titlename)
    plt.xlabel('wavenumber in $cm^{-1}$')
    plt.ylabel('absorption coefficient $cm^{2}/molecule $')
    plt.legend(loc='best')
    
def PlotXSec(nu1,c1,nu2,c2,nu3,c3,titlename):
    plt.figure()
    plt.plot(nu1,c1,'k-',label='us')
    plt.plot(nu2,c2,'b-',label='mw')
    plt.plot(nu3,c3,'r-',label='ms')
    
    plt.title(titlename)
    plt.xlabel('wavenumber in $cm^{-1}$')
    plt.ylabel('cross-section $cm^{2}/molecule $')
    plt.legend(loc='best')
    
def PlotXSec2(wl1,c1,wl2,c2,wl3,c3,titlename):
    plt.figure()
    plt.plot(wl1,c1,'k-',label='us')
    plt.plot(wl2,c2,'b-',label='mw')
    plt.plot(wl3,c3,'r-',label='ms')
    
    plt.title(titlename)
    plt.xlabel('wavelength in nm')
    plt.ylabel('cross-section $cm^{2}/molecule $')
    plt.legend(loc='best')
    
def PlotRayleighTransmission(wl,T_us,T_mw,T_ms,T_vod):
    plt.figure()
    plt.plot(wl,T_us,'r',label='horiz us')
    plt.plot(wl,T_mw,'b',label='horiz mw')
    plt.plot(wl,T_ms,'g',label='horiz ms')
    plt.plot(wl,T_vod,'k',label='vertical')
    plt.title('Rayleigh transmission at StarDice@OHP')
    plt.ylabel('transmission')
    plt.xlabel('wavelength (nm)')
    plt.legend()
    
def PlotRawTransmittance(nu_t_us,trans_us,nu_t_mw,trans_mw,nu_t_ms,trans_ms,titlename):
    plt.figure()
    plt.plot(nu_t_ms,trans_ms,'g',lw=0.5,label='ms')
    plt.plot(nu_t_us,trans_us,'b',lw=0.5,label='us')
    plt.plot(nu_t_mw,trans_mw,'r',lw=0.5,label='mw')
    plt.title(titlename)
    plt.xlabel('wavenumber in $cm^{-1}$')
    plt.ylabel('raw transmittance ')
    plt.legend()      
    
def PlotSmoothTransmittance(nu_t_us,trans_us,nu_t_mw,trans_mw,nu_t_ms,trans_ms,titlename):
    plt.figure()
    plt.plot(nu_t_ms,trans_ms,'g',label='ms')
    plt.plot(nu_t_us,trans_us,'b',label='us')
    plt.plot(nu_t_mw,trans_mw,'r',label='mw')
    plt.title(titlename)
    plt.xlabel('wavenumber in $cm^{-1}$')
    plt.ylabel('smooth air transmittance ')
    plt.legend()

def PlotSmoothTransmittance2(nu_t_us,trans_us,nu_t_mw,trans_mw,nu_t_ms,trans_ms,titlename):    
    plt.figure()
    plt.plot(1e7/nu_t_ms,trans_ms,'g',label='ms')
    plt.plot(1e7/nu_t_us,trans_us,'b',label='us')
    plt.plot(1e7/nu_t_mw,trans_mw,'r',label='mw')
    plt.title(titlename)
    plt.xlabel('wavelength in nm ')
    plt.ylabel('smooth Air transmission')
    #plt.xlim(500,1000)
    plt.legend()    

#####################################################################
# The program simulation start here
#   Variation of precipitable water
####################################################################

if __name__ == "__main__":
    
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12, 8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    
 
    
    #----------------------------------------------------------------------------------------------
    # retrieve atmospheric data on Temeprature, Pressure and Air density, and species densities 
    #----------------------------------------------------------------------------------------------
    df=pd.read_csv(atm_ground_file)


    print(df)
    
    d_Air_us=df['Air'][0]/(u.cm)**3  # molec/cm^3 : air density  for US standard atmosphere at ground
    d_Air_mw=df['Air'][1]/(u.cm)**3 # molec/cm^3 : air density  for Mid latitude winter atmosphere at ground
    d_Air_ms=df['Air'][2]/(u.cm)**3 # molec/cm^3 : air density  for Mid latitude summer atmosphere at ground
    
    d_O2_us=df['O2'][0]  # molec/cm^3 : O2 density  for US standard atmosphere at ground
    d_O2_mw=df['O2'][1] # molec/cm^3 : O2 density  for Mid latitude winter atmosphere at ground
    d_O2_ms=df['O2'][2] # molec/cm^3 : O2 density  for Mid latitude summer atmosphere at ground
    
    
    d_O3_us=df['O3'][0]  # molec/cm^3 : O3 density  for US standard atmosphere at ground
    d_O3_mw=df['O3'][1] # molec/cm^3 : O3 density  for Mid latitude winter atmosphere at ground
    d_O3_ms=df['O3'][2] # molec/cm^3 : O3 density  for Mid latitude summer atmosphere at ground
    
    d_H2O_us=df['PWV'][0]  # molec/cm^3 : H2O density  for US standard atmosphere at ground
    d_H2O_mw=df['PWV'][1] # molec/cm^3 : H2O density  for Mid latitude winter atmosphere at ground
    d_H2O_ms=df['PWV'][2] # molec/cm^3 : H2O density  for Mid latitude summer atmosphere at ground

    d_CO2_us=df['CO2'][0]  # molec/cm^3 : CO2 density  for US standard atmosphere at ground
    d_CO2_mw=df['CO2'][1] # molec/cm^3 : CO2 density  for Mid latitude winter atmosphere at ground
    d_CO2_ms=df['CO2'][2] # molec/cm^3 : CO2 density  for Mid latitude summer atmosphere at ground
    
    d_NO2_us=df['NO2'][0]  # molec/cm^3 : NO2 density  for US standard atmosphere at ground
    d_NO2_mw=df['NO2'][1] # molec/cm^3 : NO2 density  for Mid latitude winter atmosphere at ground
    d_NO2_ms=df['NO2'][2] # molec/cm^3 : NO2 density  for Mid latitude summer atmosphere at ground

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


    #---------------------------------  
    # Initialisation of Hapi/Hitran
    # ---------------------------------
    hitr=Hitran()
    
    ######### O2 ##################################
    
    print '*******************************************************************************'
    print '*                 Compute O2 '
    print '*******************************************************************************'
    
    nu_us_o2,coef_us_o2=hitr.getO2(Patm_us,T_us)
    nu_mw_o2,coef_mw_o2=hitr.getO2(Patm_mw,T_mw)
    nu_ms_o2,coef_ms_o2=hitr.getO2(Patm_ms,T_ms)
    if PlotFlag:
        PlotCoef(nu_us_o2,coef_us_o2,nu_mw_o2,coef_mw_o2,nu_ms_o2,coef_ms_o2,'absorption coefficient for $O_2$')
    
 
    
    
    ########## H2O #############################
    print '*******************************************************************************'
    print '*                 Compute H2O '
    print '*******************************************************************************'
    
    nu_us_h2o,coef_us_h2o=hitr.getH2O(Patm_us,T_us)
    nu_mw_h2o,coef_mw_h2o=hitr.getH2O(Patm_mw,T_mw)
    nu_ms_h2o,coef_ms_h2o=hitr.getH2O(Patm_ms,T_ms)
    if PlotFlag:
        PlotCoef(nu_us_h2o,coef_us_h2o,nu_mw_h2o,coef_mw_h2o,nu_ms_h2o,coef_ms_h2o,'absorption coefficient for $H_2O$')
    
    print '*******************************************************************************'
    print '*                 Compute CO2 '
    print '*******************************************************************************'
    
    nu_us_co2,coef_us_co2=hitr.getCO2(Patm_us,T_us)
    nu_mw_co2,coef_mw_co2=hitr.getCO2(Patm_mw,T_mw)
    nu_ms_co2,coef_ms_co2=hitr.getCO2(Patm_ms,T_ms)
    if PlotFlag:
        PlotCoef(nu_us_co2,coef_us_co2,nu_mw_co2,coef_mw_co2,nu_ms_co2,coef_ms_co2,'absorption coefficient for $CO_2$')
    
    
    print '*******************************************************************************'
    print '*                 Compute NO2 '
    print '*******************************************************************************'
    nu_us_no2,coef_us_no2=hitr.getNO2(Patm_us,T_us)
    nu_mw_no2,coef_mw_no2=hitr.getNO2(Patm_mw,T_mw)
    nu_ms_no2,coef_ms_no2=hitr.getNO2(Patm_ms,T_ms)
    if PlotFlag:
        PlotXSec(nu_us_no2,coef_us_no2,nu_mw_no2,coef_mw_no2,nu_ms_no2,coef_ms_no2,'cross-section for $NO_2$')
    
    print '*******************************************************************************'
    print '*                 Compute O3 Huggins '
    print '*******************************************************************************'
    nu_us_o3Hug,coef_us_o3Hug=hitr.getO3Huggins(Patm_us,T_us)
    nu_mw_o3Hug,coef_mw_o3Hug=hitr.getO3Huggins(Patm_mw,T_mw)
    nu_ms_o3Hug,coef_ms_o3Hug=hitr.getO3Huggins(Patm_ms,T_ms)
    #if PlotFlag:
    PlotXSec(nu_us_o3Hug,coef_us_o3Hug,nu_mw_o3Hug,coef_mw_o3Hug,nu_ms_o3Hug,coef_ms_o3Hug,'cross-section for $O_3$ Huggins band')
    
    print '*******************************************************************************'
    print '*                 Compute O3 Chappuis '
    print '*******************************************************************************'
    wl_us_o3Chap,coef_us_o3Chap=hitr.getO3Chappuis(Patm_us,T_us)
    wl_mw_o3Chap,coef_mw_o3Chap=hitr.getO3Chappuis(Patm_mw,T_mw)
    wl_ms_o3Chap,coef_ms_o3Chap=hitr.getO3Chappuis(Patm_ms,T_ms)
    if PlotFlag:
        PlotXSec2(wl_us_o3Chap,coef_us_o3Chap,wl_mw_o3Chap,coef_mw_o3Chap,wl_ms_o3Chap,coef_ms_o3Chap,'cross-section for $O_3$ Chappuis band')
    
    
    
    print '*******************************************************************************'
    print '*                 Compute Rayleigh '
    print '*******************************************************************************'
    #---------------------------
    ## Rayleigh Transmittance
    #------------------------------
    
    XDepth_us=d_Air_us*distance/N_A*M_air_dry
    XDepth_mw=d_Air_mw*distance/N_A*M_air_dry
    XDepth_ms=d_Air_ms*distance/N_A*M_air_dry
    
    
    
    ray=Rayleigh()
    wavelength=np.linspace(200.,1100.,100)*u.nm
    
    rayleigh_od_us= ray.RayOptDepthXD(wavelength,XDepth_us)
    rayleigh_od_mw= ray.RayOptDepthXD(wavelength,XDepth_mw)
    rayleigh_od_ms= ray.RayOptDepthXD(wavelength,XDepth_ms)
    
    
    rayleigh_vod=ray.RayOptDepth(wavelength, altitude=altitude)
    
    Tr_us=np.exp(-rayleigh_od_us.decompose())
    Tr_mw=np.exp(-rayleigh_od_mw.decompose())
    Tr_ms=np.exp(-rayleigh_od_ms.decompose())
    Tr_vod=np.exp(-rayleigh_vod.decompose())
    
    if PlotFlag:
        PlotRayleighTransmission(wavelength,Tr_us,Tr_mw,Tr_ms,Tr_vod)
    
    
    
    
#------------------------------------------------------------------------------------------    
    

#### --- PWV ----    
    coeff_us_l_h2o=coef_us_h2o*d_H2O_us # in cm-1
    coeff_mw_l_h2o=coef_mw_h2o*d_H2O_mw  # in cm-1
    coeff_ms_l_h2o=coef_ms_h2o*d_H2O_ms  # in cm-1
    
    nu_t_us,trans_us = transmittanceSpectrum(nu_us_h2o,coeff_us_l_h2o,Environment={'l': Distance_source_tel})
    nu_t_mw,trans_mw = transmittanceSpectrum(nu_mw_h2o,coeff_mw_l_h2o,Environment={'l': Distance_source_tel})
    nu_t_ms,trans_ms = transmittanceSpectrum(nu_ms_h2o,coeff_ms_l_h2o,Environment={'l': Distance_source_tel})
    
    if PlotFlag:
        PlotRawTransmittance(nu_t_us,trans_us,nu_t_mw,trans_mw,nu_t_ms,trans_ms,'Air transmittance for $H_2O$')
    
    
    nu_t_us_smooth_h2o,trans_us_smooth_h2o,i1,i2,slit = convolveSpectrum(nu_t_us,trans_us,SlitFunction=SLIT_RECTANGULAR,Resolution=100.)
    nu_t_mw_smooth_h2o,trans_mw_smooth_h2o,i1,i2,slit = convolveSpectrum(nu_t_mw,trans_mw,SlitFunction=SLIT_RECTANGULAR,Resolution=100.)
    nu_t_ms_smooth_h2o,trans_ms_smooth_h2o,i1,i2,slit = convolveSpectrum(nu_t_ms,trans_ms,SlitFunction=SLIT_RECTANGULAR,Resolution=100.)
    
    if PlotFlag:
        PlotSmoothTransmittance(nu_t_us_smooth_h2o,trans_us_smooth_h2o,nu_t_mw_smooth_h2o,trans_mw_smooth_h2o,nu_t_ms_smooth_h2o,trans_ms_smooth_h2o,'Smoothed Air transmittance for $H_2O$ (StarDice@OHP)')
        PlotSmoothTransmittance2(nu_t_us_smooth_h2o,trans_us_smooth_h2o,nu_t_mw_smooth_h2o,trans_mw_smooth_h2o,nu_t_ms_smooth_h2o,trans_ms_smooth_h2o,'Smoothed Air transmittance for $H_2O$ (StarDice@OHP)')
    
    
    
#### --- O2 ----    
    coeff_us_l_o2=coef_us_o2*d_O2_us # in cm-1
    coeff_mw_l_o2=coef_mw_o2*d_O2_mw  # in cm-1
    coeff_ms_l_o2=coef_ms_o2*d_O2_ms  # in cm-1
    
   
    
    nu_t_us,trans_us = transmittanceSpectrum(nu_us_o2,coeff_us_l_o2,Environment={'l': Distance_source_tel})
    nu_t_mw,trans_mw = transmittanceSpectrum(nu_mw_o2,coeff_mw_l_o2,Environment={'l': Distance_source_tel})
    nu_t_ms,trans_ms = transmittanceSpectrum(nu_ms_o2,coeff_ms_l_o2,Environment={'l': Distance_source_tel})
    
    if PlotFlag:
        PlotRawTransmittance(nu_t_us,trans_us,nu_t_mw,trans_mw,nu_t_ms,trans_ms,'Air transmittance for $O_2$')
    
    
    nu_t_us_smooth_o2,trans_us_smooth_o2,i1,i2,slit = convolveSpectrum(nu_t_us,trans_us,SlitFunction=SLIT_RECTANGULAR,Resolution=100.0)
    nu_t_mw_smooth_o2,trans_mw_smooth_o2,i1,i2,slit = convolveSpectrum(nu_t_mw,trans_mw,SlitFunction=SLIT_RECTANGULAR,Resolution=100.0)
    nu_t_ms_smooth_o2,trans_ms_smooth_o2,i1,i2,slit = convolveSpectrum(nu_t_ms,trans_ms,SlitFunction=SLIT_RECTANGULAR,Resolution=100.0)
    
    if PlotFlag:
        PlotSmoothTransmittance(nu_t_us_smooth_o2,trans_us_smooth_o2,nu_t_mw_smooth_o2,trans_mw_smooth_o2,nu_t_ms_smooth_o2,trans_ms_smooth_o2,'Smoothed Air transmittance for $O_2$ (StarDice@OHP)')
        PlotSmoothTransmittance2(nu_t_us_smooth_o2,trans_us_smooth_o2,nu_t_mw_smooth_o2,trans_mw_smooth_o2,nu_t_ms_smooth_o2,trans_ms_smooth_o2,'Smoothed Air transmittance for $O_2$ (StarDice@OHP)')    
    
 
    
#### --- O3 Huggins ----    

    coeff_us_l=coef_us_o3Hug*d_O3_us # in cm-1
    coeff_mw_l=coef_mw_o3Hug*d_O3_mw # in cm-1
    coeff_ms_l=coef_ms_o3Hug*d_O3_ms # in cm-1
    
    nu_t_us_o3hug,trans_us_o3hug = transmittanceSpectrum(nu_us_o3Hug,coeff_us_l,Environment={'l': Distance_source_tel})
    nu_t_mw_o3hug,trans_mw_o3hug = transmittanceSpectrum(nu_mw_o3Hug,coeff_mw_l,Environment={'l': Distance_source_tel})
    nu_t_ms_o3hug,trans_ms_o3hug = transmittanceSpectrum(nu_ms_o3Hug,coeff_ms_l,Environment={'l': Distance_source_tel})
    
    if PlotFlag:
        PlotSmoothTransmittance(nu_t_us_o3hug,trans_us_o3hug,nu_t_mw_o3hug,trans_mw_o3hug,nu_t_ms_o3hug,trans_ms_o3hug,'Air transmittance for $O_3$ Huggins (StarDice@OHP)')
        PlotSmoothTransmittance2(nu_t_us_o3hug,trans_us_o3hug,nu_t_mw_o3hug,trans_mw_o3hug,nu_t_ms_o3hug,trans_ms_o3hug,'Air transmittance for $O_3$ Huggins(StarDice@OHP)')    
 
    
    
    
#### --- O3 Chappuis ----    

    coeff_us_l=coef_us_o3Chap*d_O3_us # in cm-1
    coeff_mw_l=coef_mw_o3Chap*d_O3_mw # in cm-1
    coeff_ms_l=coef_ms_o3Chap*d_O3_ms # in cm-1
    
    tau_o3chap_us= coeff_us_l*Distance_source_tel
    tau_o3chap_mw= coeff_mw_l*Distance_source_tel
    tau_o3chap_ms= coeff_ms_l*Distance_source_tel
       
   
    
    Tr_o3chap_us=np.exp(-tau_o3chap_us.decompose())
    Tr_o3chap_mw=np.exp(-tau_o3chap_mw.decompose())
    Tr_o3chap_ms=np.exp(-tau_o3chap_ms.decompose())
    
    
 #### --- CO2 ----
 
    coeff_us_l_co2=coef_us_co2*d_CO2_us # in cm-1
    coeff_mw_l_co2=coef_mw_co2*d_CO2_mw  # in cm-1
    coeff_ms_l_co2=coef_ms_co2*d_CO2_ms  # in cm-1
    
   
    
    nu_t_us,trans_us = transmittanceSpectrum(nu_us_co2,coeff_us_l_co2,Environment={'l': Distance_source_tel})
    nu_t_mw,trans_mw = transmittanceSpectrum(nu_mw_co2,coeff_mw_l_co2,Environment={'l': Distance_source_tel})
    nu_t_ms,trans_ms = transmittanceSpectrum(nu_ms_co2,coeff_ms_l_co2,Environment={'l': Distance_source_tel})
    
    if PlotFlag:
        PlotRawTransmittance(nu_t_us,trans_us,nu_t_mw,trans_mw,nu_t_ms,trans_ms,'Air transmittance for $CO_2$')
    
    
    nu_t_us_smooth_co2,trans_us_smooth_co2,i1,i2,slit = convolveSpectrum(nu_t_us,trans_us,SlitFunction=SLIT_RECTANGULAR,Resolution=100.0)
    nu_t_mw_smooth_co2,trans_mw_smooth_co2,i1,i2,slit = convolveSpectrum(nu_t_mw,trans_mw,SlitFunction=SLIT_RECTANGULAR,Resolution=100.0)
    nu_t_ms_smooth_co2,trans_ms_smooth_co2,i1,i2,slit = convolveSpectrum(nu_t_ms,trans_ms,SlitFunction=SLIT_RECTANGULAR,Resolution=100.0)
    
    if PlotFlag:
        PlotSmoothTransmittance(nu_t_us_smooth_co2,trans_us_smooth_co2,nu_t_mw_smooth_co2,trans_mw_smooth_co2,nu_t_ms_smooth_co2,trans_ms_smooth_co2,'Smoothed Air transmittance for $CO_2$ (StarDice@OHP)')
        PlotSmoothTransmittance2(nu_t_us_smooth_co2,trans_us_smooth_co2,nu_t_mw_smooth_co2,trans_mw_smooth_co2,nu_t_ms_smooth_co2,trans_ms_smooth_co2,'Smoothed Air transmittance for $CO_2$ (StarDice@OHP)')      
 
    
    
    
 ####### Plot

   
   
   
    plt.figure()
    plt.plot(wavelength,Tr_us,'b-',lw=2,label='Rayleigh')
    plt.plot(1e7/nu_t_us_smooth_h2o,trans_us_smooth_h2o,'r-',lw=2,label='$H_2O$')
    plt.plot(1e7/nu_t_us_smooth_o2,trans_us_smooth_o2,'g-',lw=2,label='$O_2$')
    plt.plot(1e7/nu_t_us_o3hug,trans_us_o3hug,'m-',lw=2,label='$O_3-Huggins$')
    plt.plot(1e7/nu_t_us_smooth_co2,trans_us_smooth_co2,'k-',lw=2,label='$CO_2$')
    plt.plot(wl_us_o3Chap,Tr_o3chap_us,'-',color='grey',lw=2,label='$O_3-Chappuis$')
    plt.legend()
    plt.xlabel('$\lambda$ (nm)',fontsize=16, fontweight='bold')
    plt.ylabel('air transmittance',fontsize=16, fontweight='bold')
    plt.title('Air transmittance for US atm @ OHP',fontsize=16, fontweight='bold')
    #plt.ylim(0,1.1)
    plt.grid(True)
    plt.savefig('airrransmittanceAtOHP.png')
    
    
   
    
    rtdata1 = np.loadtxt(libradtran_file_oh)
    x1=rtdata1[:,0]
    y1=rtdata1[:,1]
    
    rtdata2 = np.loadtxt(libradtran_file_ls)
    x2=rtdata2[:,0]
    y2=rtdata2[:,1]
    
    plt.figure()
    plt.plot(x1,y1,'k:',lw=2,label='libradtran,ohp,z=1')
    plt.plot(x2,y2,':',color='grey',lw=2,label='libradtran,lsst,z=1')
    plt.plot(wavelength,Tr_us,'b-',lw=2,label='Rayleigh')
    plt.plot(1e7/nu_t_us_smooth_h2o,trans_us_smooth_h2o,'r-',lw=2,label='$H_2O$')
    plt.plot(1e7/nu_t_us_smooth_o2,trans_us_smooth_o2,'g-',lw=2,label='$O_2$')
    plt.plot(1e7/nu_t_us_o3hug,trans_us_o3hug,'m-',lw=2,label='$O_3-Huggins$')
 #   plt.plot(1e7/nu_t_us_smooth_co2,trans_us_smooth_co2,'k-',lw=2,label='$CO_2$')
#    plt.plot(wl_us_o3Chap,Tr_o3chap_us,'-',color='grey',lw=2,label='$O_3-Chappuis$')
    plt.legend(loc='best')
    plt.xlabel('$\lambda$ (nm)',fontsize=16, fontweight='bold')
    plt.ylabel('air transmittance',fontsize=16, fontweight='bold')
    plt.title('Air transmittance for US atm @ OHP',fontsize=16, fontweight='bold')
    #plt.ylim(0,1.1)
    plt.grid(True)
    plt.savefig('airrransmittanceAtOHP2.png')
    
    rtdata3 = np.loadtxt(libradtran_file_oh_q5)
    x3=rtdata3[:,0]
    y3=rtdata3[:,1]
    
    rtdata4 = np.loadtxt(libradtran_file_oh_q50)
    x4=rtdata4[:,0]
    y4=rtdata4[:,1]
    
    rtdata5 = np.loadtxt(libradtran_file_oh_q95)
    x5=rtdata5[:,0]
    y5=rtdata5[:,1]
    
    
    
    
    plt.figure()
    plt.plot(x3,y3,'-',color='orange',lw=1,label='libradtran,ohp,z=1,q=5%,50%,95%')
    plt.plot(x4,y4,'-',color='orange',lw=1,label='_nolabel_')
    plt.plot(x5,y5,'-',color='orange',lw=1,label='_nolabel_')
    plt.plot(x2,y2,':',color='grey',lw=2,label='libradtran,lsst,z=1')
    plt.plot(wavelength,Tr_us,'b-',lw=2,label='Rayleigh')
    plt.plot(1e7/nu_t_us_smooth_h2o,trans_us_smooth_h2o,'r-',lw=2,label='$H_2O$')
    plt.plot(1e7/nu_t_us_smooth_o2,trans_us_smooth_o2,'g-',lw=2,label='$O_2$')
    plt.plot(1e7/nu_t_us_o3hug,trans_us_o3hug,'m-',lw=2,label='$O_3-Huggins$')
 #   plt.plot(1e7/nu_t_us_smooth_co2,trans_us_smooth_co2,'k-',lw=2,label='$CO_2$')
#    plt.plot(wl_us_o3Chap,Tr_o3chap_us,'-',color='grey',lw=2,label='$O_3-Chappuis$')
    plt.legend(loc='best')
    plt.xlabel('$\lambda$ (nm)',fontsize=16, fontweight='bold')
    plt.ylabel('air transmittance',fontsize=16, fontweight='bold')
    plt.title('Air transmittance for US atm @ OHP',fontsize=16, fontweight='bold')
    #plt.ylim(0,1.1)
    plt.grid(True)
    plt.savefig('airrransmittanceAtOHP3.png')
    
    
    
    wavelength2=np.linspace(200.,1200.,100)*u.nm
    HAOD=HAeroOptDepth(wavelength2,L=200.*u.m,HP=1.*u.km,tau_aerosols_550=VAOD_quantiles[2])
    
    N_VAOD=VAOD_quantiles.shape[0]
    
    plt.figure()
    plt.plot(wavelength2,np.exp(-HAOD),'-',color='black',lw=2,label='Aerosols q=5%-95%')
    for i in np.arange(0,N_VAOD):
        HAOD=HAeroOptDepth(wavelength2,L=200.*u.m,HP=1.0*u.km,tau_aerosols_550=VAOD_quantiles[i])
        plt.plot(wavelength2,np.exp(-HAOD.decompose()),'-',color='black',lw=1)
    plt.plot(wavelength,Tr_us,'b-',lw=2,label='Rayleigh')
    plt.plot(1e7/nu_t_us_smooth_h2o,trans_us_smooth_h2o,'r-',lw=2,label='$H_2O$')
    plt.plot(1e7/nu_t_us_smooth_o2,trans_us_smooth_o2,'g-',lw=2,label='$O_2$')
    plt.plot(1e7/nu_t_us_o3hug,trans_us_o3hug,'m-',lw=2,label='$O_3-Huggins$')
 #   plt.plot(1e7/nu_t_us_smooth_co2,trans_us_smooth_co2,'k-',lw=2,label='$CO_2$')
#    plt.plot(wl_us_o3Chap,Tr_o3chap_us,'-',color='grey',lw=2,label='$O_3-Chappuis$')
    plt.legend(loc='best')
    plt.xlabel('$\lambda$ (nm)',fontsize=16, fontweight='bold')
    plt.ylabel('air transmittance',fontsize=16, fontweight='bold')
    plt.title('Air transmittance for US atm @ OHP',fontsize=16, fontweight='bold')
    #plt.ylim(0,1.1)
    plt.grid(True)
    plt.savefig('airrransmittanceAtOHP4.png')
    
    
    
    
    
    
    
    
    
    
    