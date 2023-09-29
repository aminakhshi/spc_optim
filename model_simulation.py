#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 06:31:24 2023

@author: amin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 20:05:33 2023

@author: amin
"""
import argparse
import os, sys
import datetime
import pickle
import random
import numpy as np
import pandas as pd
from scipy import spatial
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy import signal
from scipy.interpolate import interp1d
import common.cnoise as cn
from common.model_params import get_params
from scipy.signal import find_peaks
from lmfit import minimize, Minimizer, Parameters, conf_interval, conf_interval2d,report_ci, report_fit

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from common.visualizations import *


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
fontprops = matplotlib.font_manager.FontProperties(size=10)
cm = 1/2.54

class GhostBurst(object):
    def __init__(self, params, **optionals):
        try:
            ## Parameter values used in the two-compartmental Hodgkin-Huxley model
            self.C_m = params['C_m'].value      # membrane capacitance (uF/cm**2)
            self.kappa = params['kappa'].value      ########### membrane capacitance (uF/cm**2)
            self.gc = params['gc'].value      ########## membrane capacitance (uF/cm**2)'''
            self.VNa = params['VNa'].value      # Na+ channel Nernst potential (mV)
            self.VK = params['VK'].value        # K+ channel Nernst potential (mV)
            self.VCa = params['VCa'].value      # Ca2+ channel Nernst potential (mV)
            self.Vleak = params['Vleak'].value  # leak Nernst potential (mV)

            self.gNaS = params['gNaS'].value      # Na+ channel maximum conductances in the Soma (mS/cm**2)
            self.gDrS = params['gDrS'].value      # K+ channel maximum conductances in the Soma (mS/cm**2)
            self.gNaD = params['gNaD'].value      # Na+ channel maximum conductances in the Dendrite (mS/cm**2)
            self.gDrD = params['gDrD'].value      # K+ channel maximum conductances in the Dendrite (mS/cm**2)
            self.gleak = params['gleak'].value    # Soma to Dendrite leak maximum conductance (mS/cm**2)
            self.gSK = params['gSK'].value        # SK channel maximum conductance (mS/cm**2)
            self.gNMDA = params['gNMDA'].value    # NMDA receptor maximum conductance (mS/cm**2)

            self.kCa = params['kCa'].value        # half-maximum activation of the SK channel (uM)
            self.tau_ns = params['tau_ns'].value  # activation time constants of K+ channel in the Soma (ms)
            self.tau_hd = params['tau_hd'].value  # inactivation time constants of Na+ channels in the Dendrite (ms)
            self.tau_nd = params['tau_nd'].value  # activation time constants of K+ channel in the Dendrite (ms)
            self.tau_pd = params['tau_pd'].value  # inactivation time constants of K+ channels in the Dendrite (ms)
        
            self.V_ms = params['V_ms'].value      # steady-state conductance curve V_1/2 of Na+ channel activation in the Soma (mV)
            self.V_ns = params['V_ns'].value      # steady-state conductance curve V_1/2 of K+ channel activation in the Soma (mV)
            self.V_md = params['V_md'].value      # steady-state conductance curve V_1/2 of Na+ channel activation in the Dendrite (mV)
            self.V_hd = params['V_hd'].value      # steady-state conductance curve V_1/2 of Na+ channel inactivation in the Dendrite (mV)
            self.V_nd = params['V_nd'].value      # steady-state conductance curve V_1/2 of K+ channel activation in the Dendrite (mV)
            self.V_pd = params['V_pd'].value      # steady-state conductance curve V_1/2 of K+ channel inactivation in the Dendrite (mV)
        
            self.s_ms = params['s_ms'].value      # steady-state conductance curve constant of Na+ channel activation in the Soma 
            self.s_md = params['s_md'].value      # steady-state conductance curve constant of Na+ channel activation in the Dendrite 
            self.s_ns = params['s_ns'].value      # steady-state conductance curve constant of K+ channel activation in the Soma
            self.s_nd = params['s_nd'].value      # steady-state conductance curve constant of K+ channel activation in the Dendrite
            self.s_hd = params['s_hd'].value      # steady-state conductance curve constant of Na+ channel inactivation in the Dendrite
            self.s_pd = params['s_pd'].value      # steady-state conductance curve constant of K+ channel inactivation in the Dendrite

            ## Parameter values used in the NMDA receptors Markov model
            self.Mg_o = params['Mg_o'].value      # extracellular Mg2+ concentration (mM)
            self.R_b = params['R_b'].value        # binding rate between the closed states (1/s) 
            self.R_u = params['R_u'].value        # unbinding rate between the closed states (1/s)
            self.R_o = params['R_o'].value        # binding rate between the open and closed states (1/s)
            self.R_c = params['R_c'].value        # unbinding rate between the open and closed states (1/s)
            self.R_r = params['R_r'].value        # binding rate between the desensitized and closed states (1/s)
            self.R_d = params['R_d'].value        # unbinding rate between the desensitized and closed states (1/s)
    
            ## Parameter values used in the flux-balance Calcium model
            self.nu_PMCA = params['nu_PMCA'].value  # maximum flux rates through PMCA pumps  (uM/s)
            self.nu_Serca = params['nu_Serca'].value # maximum flux rates through SERCA pumps (uM/s)
            self.k_PMCA = params['kPMCA'].value   # half-maximum activation for calcium fluxes through PMCA (uM)
            self.k_Serca = params['kSerca'].value # half-maximum activation for calcium fluxes through SERCA (uM)
            self.nu_INleak = params['nu_INleak'].value # maximum flux rates through cell membrane (uM/s)
            self.nu_ERleak = params['nu_ERleak'].value # maximum flux rates through ER membrane (uM/s)
            self.nu_IP3 = params['nu_IP3'].value  # maximum flux rate of IP3R (uM/s)
            self.d_1 = params['d_1'].value        # dissociation constant of IP3 (uM)
            self.d_2 = params['d_2'].value        # dissociation constant of Ca2+ inhibition (uM)
            self.d_3 = params['d_3'].value        # dissociation constant of IP3 (uM)
            self.d_5 = params['d_5'].value        # dissociation constant of Ca2+ activation (uM)
            self.a2 = params['a2'].value          # binding constant of Ca2+ inhibition (1/s)
            self.IP3 = params['IP3'].value        # cytosolic concentration of IP3 in the dendrite (uM)
            self.f_c = params['f_c'].value        # fraction of free calcium concentration in the cytosolic component
            self.f_ER = params['f_ER'].value      # fraction of free calcium concentration in the ER component
            self.gamma = params['gamma'].value    # volume ratio of cytosol to ER in the dendrite
            self.alpha = params['alpha'].value    ############################

            ## setup parameters and state variables
            self.Iapp = params['Iapp'].value      # Somatic applied Current (nA) 
            self.lambda_glu = params['lambda_glu'].value    # synaptic bombardment probability (1/ms) 
            self.beta = params['beta'].value      # Synaptic input noise exponent
            self.sigma = params['sigma'].value    # synaptic input noise intensity (nA) 
            self.tau_decay = params['tau_decay'].value
            self.A_glu = params['A_glu'].value # Applied Current (nA) 
            self.lastrelease = params['lastrelease'].value
        except Exception:
            raise ValueError("parameters are not defined")

    ## steady-state conductance activation curve for Na+ channel in the Soma 
    def ms_inf(self, V):
        return (1 / (1 + np.exp(-(V + self.V_ms) / self.s_ms)))
    
    ## steady-state conductance inactivation curve for K+ channel in the Soma 
    def ns_inf(self, V):
        return (1 / (1 + np.exp(-(V + self.V_ns) / self.s_ns)))
    
    ## steady-state conductance activation curve for Na+ channel in the Dendrite 
    def md_inf(self, V):
        return (1 / (1 + np.exp(-(V + self.V_md) / self.s_md)))
    
    ## steady-state conductance inactivation curve for Na+ channel in the Dendrite 
    def hd_inf(self, V):
        return (1 / (1 + np.exp((V + self.V_hd) / self.s_hd)))
    
    ## steady-state conductance activation curve for K+ channel in the Dendrite 
    def nd_inf(self, V):
        return (1 / (1 + np.exp(-(V + self.V_nd) / self.s_nd)))
    
    ## steady-state conductance inactivation curve for K+ channel in the Dendrite 
    def pd_inf(self, V):
        return (1 / (1 + np.exp((V + self.V_pd) / self.s_pd)))

    ##################################### 
    def Q2(self, t=None):
        return (self.d_2 * ((self.IP3 + self.d_1) / (self.IP3 + self.d_3)))

    ## steady-state activation curve for IP3R 
    def minf_IP3(self, t=None):
        return (self.IP3 / (self.IP3 + self.d_1))

    ## steady-state activation curve for Calcium  
    def ninf_IP3(self, Ca):
        return (Ca / (Ca + self.d_5))
    
    ## steady-state inactivation curve for IP3R  
    def hinf_IP3(self, Ca, t=None):
        return (self.Q2(t) / (self.Q2(t) + Ca))
    
    ## magnesium block function
    def B(self, V):
        return (1 / (1 + (self.Mg_o * np.exp(-0.062 * V)) / 3.57))

    # glutamate release function 
    def glurelease(self, tau):
        glurel = (self.A_glu * np.exp(-tau/self.tau_decay)) if tau >=0 else 0
        return glurel

    def butter_highpass(data, cutoff, fs, order=5, btype='high'):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype= btype, analog=False)
        filt_data = signal.filtfilt(b, a, data)
        return filt_data
    
    def HH(self, t, y, glui, dw):
        ## State variables of the dynamical system
        Vs = y[0]
        Vd = y[1]
        ns = y[2]
        hd = y[3]
        nd = y[4]
        pd = y[5]
        Ca = y[6]
        CaER = y[7]
        hca = y[8]
        C0 = y[9]
        C1 = y[10]
        C2 = y[11]
        Ds = y[12]
        Os = y[13]
        # Vs, Vd, ns, hd, nd, pd, Ca, CaER, hca, C0, C1, C2, Ds, Os = y
        # global lastrelease
        if glui(t) == 1:
            # print(t)
            self.lastrelease = t
            rltime = t
        else:
            rltime = self.lastrelease
        glu = self.glurelease(tau = t-rltime)
        if glu < 1e-5:
            glu = 0.0

        I_NaS = self.gNaS * (self.ms_inf(Vs) ** 2) * (1 - ns) * (Vs - self.VNa)
        I_DrS = self.gDrS * (ns ** 2) * (Vs - self.VK)
        I_NaD = self.gNaD * (self.md_inf(Vd) ** 2) * hd * (Vd - self.VNa)
        I_DrD = self.gDrD * (nd ** 2) * pd * (Vd - self.VK)
        J_IP3 = self.nu_IP3 * (self.minf_IP3(t) ** 3) * (self.ninf_IP3(Ca) ** 3) * (hca ** 3) * (CaER - Ca)
        J_PMCA = self.nu_PMCA * (Ca ** 2) / (Ca ** 2 + self.k_PMCA ** 2)
        J_Serca = self.nu_Serca * (Ca ** 2) / (Ca ** 2 + self.k_Serca ** 2)
        J_leak = self.nu_ERleak * (CaER - Ca)
        I_SK = self.gSK * (Ca ** 2 / (Ca ** 2 + self.kCa ** 2)) * (Vd - self.VK)
        I_NMDA = self.gNMDA * self.B(Vd) * Os * (Vd - self.VCa)

        dvdt = [(self.Iapp - I_NaS - I_DrS - self.gleak * (Vs - self.Vleak) - (self.gc / self.kappa) * (Vs - Vd)) / self.C_m,
        (self.sigma*(dw(t)/self.dt) - I_NaD - I_DrD - self.gleak * (Vd - self.Vleak) - (self.gc / (1 - self.kappa)) * (Vd - Vs) - I_NMDA - I_SK) / self.C_m,
        (self.ns_inf(Vs) - ns) / self.tau_ns,
        (self.hd_inf(Vd) - hd) / self.tau_hd,
        (self.nd_inf(Vd) - nd) / self.tau_nd,
        (self.pd_inf(Vd) - pd) / self.tau_pd,
        self.f_c * (-self.alpha * I_NMDA + J_IP3 - J_Serca - J_PMCA + J_leak),
        self.f_ER * self.gamma * (-J_IP3 + J_Serca - J_leak),
        (self.hinf_IP3(Ca, t) - hca) * self.a2 * (self.Q2(t) + Ca),
        -(self.R_b * glu * C0) + (self.R_u * C1),
        -((self.R_b * glu + self.R_u) * C1) + (self.R_b * glu * C0 + self.R_u * C2),
        -((self.R_o + self.R_d + self.R_u) * C2) + (self.R_b * glu * C1 + self.R_c * Os + self.R_r * Ds),
        -(self.R_r * Ds) + (self.R_d * C2),
        -(self.R_c * Os) + (self.R_o * C2) 
        ]
        return dvdt

    def run(self, **options):
        if 'dt' in options:
            self.dt = options['dt']
        else:
            self.dt = 0.005
        
        if 'duration' in options:
            self.duration = options['duration']
        else:
            self.duration = 24000     # 120 seconds (each second is equivalent to 200)
        
        if 'ICs' in options:
            self.ICs = options['ICs']
            assert isinstance(self.ICs, (np.ndarray, list)), ValueError('initial conditions should be numpy array or list')
        else:
            self.ICs = np.random.random(14)
            print(f'initial conditions {self.ICs}')
        
        self.tdur = [0, self.duration]
        self.t_eval = np.arange(0, self.duration+self.dt, self.dt)
        
        rt_dist = np.random.random(len(self.t_eval))
        glui = (self.lambda_glu*self.dt) > rt_dist
        glui = glui*1

        if self.beta == 0:
            dw = np.random.normal(loc = 0, scale = np.sqrt(self.dt), size = len(self.t_eval))
        else:
            dw = cn.powerlaw_psd(self.beta, len(self.t_eval), self.dt, fmin=5)
        
        if 'filter_noise' in options:
            print('filtering the synaptic input')
            dw = self.butter_highpass(dw, cutoff=5, fs=200/self.dt)
        
        dw = interp1d(x=self.t_eval, y=dw)
        glui = interp1d(x=self.t_eval, y=glui)

        self.model = solve_ivp(self.HH, t_span=self.tdur, y0=self.ICs, t_eval=self.t_eval, args=(glui, dw, ), rtol=1e-5)
        # Add info needed by certain spiking features and efel features
        # info = {"stimulus_start": self.t_eval[0], "stimulus_end": self.t_eval[-1]}
        return self.t_eval, self.model.y[0]

#%%

def main_optimize(params, func: callable, recording_data, **kwargs):
    assert isinstance(params, Parameters), ValueError('params should be lmfit Parameters class')
    
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        mode = 'RMSE'
    print(f'{mode} error estimation')
    
    if 'hist_mode' in kwargs:
        hist_mode = kwargs['hist_mode']
    else:
        hist_mode = 'density'
    print(f'{hist_mode} mode histogram estimation')

    if 'dt' in kwargs:
        dt = kwargs['dt']
    else:
        dt = dt = 0.005
        
    if 'duration' in kwargs:
        duration = kwargs['duration']
    else:
        duration = 24000    # equal to 120 sec
    print(f'model simulation time {duration/200:.2f}')

    if 'ICs' in kwargs:
        ICs = kwargs['ICs']
        assert isinstance(ICs, (np.ndarray, list)), ValueError('initial conditions should be numpy array or list')
    else:
        ICs = [-53.2440, -56.3710, 0.0098, 0.7396, 0.0413, 0.1667, 
               0.00, 0.0, 0.000, 1.0, 0.0, 0.0, 0.0, 0.0]
    # print(f'initial conditions {ICs}')
    
    gb_model = func(params)
    t_eval, data = gb_model.run(dt=dt, duration=duration, ICs= ICs)

    if 'cut_off' in kwargs:
        cut_off = kwargs['cut_off']
        assert cut_off < duration, ValueError('cut_off threshold is larger than the duration')
    elif duration == 24000:
        cut_off = 20*200    # removing the first 20 sec
    else:
        cut_off = int(0.2*duration) # removing the first 20 percent
    print(f'removing first {cut_off/200:.2f} seconds from simulation')

    transient_time = int(cut_off/dt)
    data = data[transient_time:]
    t_eval = (t_eval[transient_time:] - cut_off)/200
    
    peaks, _ = find_peaks(data, height=20, width = 20, prominence=20)
    time_offset = t_eval[peaks]
    
    if np.abs(recording_data[-1] - time_offset[-1]) > 10:
        print(f'aligning spiketimes between the model and data')
        try:
            t_stop = min(recording_data[-1], time_offset[-1])
            ids = np.squeeze(np.where(recording_data<t_stop))
            recording_data = recording_data[ids]
            ids = np.squeeze(np.where(time_offset<t_stop))
            time_offset = time_offset[ids]
        except:
            raise ValueError(f'spiketimes dont match; data = {recording_data[-1]:.2f}, model = {time_offset[-1]:.2f}')
    
    isi_data = np.diff(recording_data)
    isi_model = np.diff(time_offset)
    if hist_mode == 'normal':
        hist_data, edge = np.histogram(isi_data, bins=bins)
        hist_model, edge = np.histogram(isi_model, bins=bins)
        hist_data = (hist_data/sum(hist_data))*100 
        hist_model = (hist_model/sum(hist_model))*100 
    elif hist_mode == 'density':
        hist_data, edge = np.histogram(isi_data, bins=bins, density=True)
        hist_model, edge = np.histogram(isi_model, bins=bins, density=True)
    else:
        hist_data, edge = np.histogram(isi_data, bins=bins)
        hist_model, edge = np.histogram(isi_model, bins=bins)
    
    if mode == 'RMSE':
        residual_err = np.sqrt(np.mean((hist_model - hist_data)**2))
    elif mode == 'residual':
        residual_err = hist_model - hist_data
    return residual_err
#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default=os.path.join(os.getcwd(), 'Recording') \
                        type=str, help='path file to read the spiketime data recording')
    parser.add_argument('--model', type=str, default="GhostBurst", choices= ['GhostBurst', 'GB_stim', 'GB_network', 'GB_minimal'],
                        help='Choices of the computational model used for simulation')
    parser.add_argument('--mode', type=str, default="RMSE", choices= ['RMSE', 'residual'], help='Choices of the error estimation mode')
    parser.add_argument('--hist_mode', type=str, default="density", choices= ['density', 'normal', 'count'], help='Choices of the histogram estimation mode')
    parser.add_argument('--duration', type=int, default=24000, help='duration of the simulation')
    parser.add_argument('--ICs', nargs='+', default=None, help='initial conditions for the model')
    parser.add_argument('--cut_off', type=int, default=None, help='cut off threshold for the simulation')
    parser.add_argument('--bins', type=int, default=100, help='number of bins for the histogram')
    parser.add_argument('--save', type=bool, default=True, help='Save the result')
    args = parser.parse_args()

    recording_data = spiketimes['control'][5]
    params = get_params()
    kw = {'mode': 'residual', 'hist_mode': 'normal', 'duration': 4000}
    optimizer_fcn = Minimizer(main_optimize, params, fcn_args=(GhostBurst, recording_data), fcn_kws=kw, nan_policy='omit')
    results = optimizer_fcn.emcee()   
    results = minimize(main_optimize, params, args=(GhostBurst, recording_data,), kws=kw, nan_policy='omit', method='leastsq') 
    report_fit(results)