#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 00:05:44 2023

@author: amin
"""
from lmfit import Parameters

def get_params():
    params = Parameters()
    ## morphological parameter values and Nernst potentials
    params.add('C_m', value=1.)
    params.add('kappa', value=0.4, min=0.01, max=1.)
    params.add('gc', value=1., min=0.5, max=1.5)
    params.add('VNa', value=40., min=30., max=50.)
    params.add('VK', value=-88.5, min=-98.5, max=-78.5)
    params.add('VCa', value=100., min=100., max=120.)
    params.add('Vleak', value=-70., min=50, max=90)
    
    ## maximum conductances
    params.add('gNaS', value=55., min=10, max=100)
    params.add('gDrS', value=20., min=0, max=40)
    params.add('gNaD', value=5, min=0., max=40.)
    params.add('gDrD', value=12.7, min=0., max=40)
    params.add('gleak', value=0.18, min=0.1, max=1.)
    params.add('gSK', value=0.4, min=0., max=40.)
    params.add('gNMDA', value=5., min=0., max=40.)
    
    ## activation/inactivation time constants
    params.add('kCa', value=0.1, min=0.01, max=1.)
    params.add('tau_ns', value=0.39, min=0.01, max=10)
    params.add('tau_hd', value=1.0, min=0.01, max=10.)
    params.add('tau_nd', value=0.9, min=0.01, max=10.)
    params.add('tau_pd', value=5., min=0.01, max=10.)

    ## steady-state conductance curve V_1/2
    # params.add('V_ms', value=40., min=30., max=60.)
    # params.add('V_ns', value=40., min=30., max=60.)
    # params.add('V_md', value=40., min=30., max=60.)
    # params.add('V_hd', value=52., min=30., max=60.)
    # params.add('V_nd', value=40., min=30., max=60.)
    # params.add('V_pd', value=65., min=40., max=80.)
    params.add('V_ms', value=40., vary=False)
    params.add('V_ns', value=40., vary=False)
    params.add('V_md', value=40., vary=False)
    params.add('V_hd', value=52., vary=False)
    params.add('V_nd', value=40., vary=False)
    params.add('V_pd', value=65., vary=False)

    
    ## steady-state conductance curve constant
    # params.add('s_ms', value=3., min=0.01, max=10.)
    # params.add('s_md', value=5., min=0.01, max=10.)
    # params.add('s_ns', value=3., min=0.01, max=10.)
    # params.add('s_nd', value=5., min=0.01, max=10.)
    # params.add('s_hd', value=5., min=0.01, max=10.)
    # params.add('s_pd', value=6., min=0.01, max=10.)
    params.add('s_ms', value=3., vary=False)
    params.add('s_ns', value=3., vary=False)
    params.add('s_md', value=5., vary=False)
    params.add('s_hd', value=5., vary=False)
    params.add('s_nd', value=5., vary=False)
    params.add('s_pd', value=6., vary=False)

    ## binding/unbinding rates in the NMDA receptor
    params.add('Mg_o', value=1., min=0.001, max=10.)
    params.add('R_b', value=4.59e1, min=1., max=1e3)
    params.add('R_u', value=12.9, min=1., max=1e3)
    params.add('R_o', value=46.5, min=1., max=1e3)
    params.add('R_c', value=73.8, min=1., max=1e3)
    params.add('R_r', value=6.8, min=1., max=1e3)
    params.add('R_d', value=8.4, min=1., max=1e3)
    
    ## maximum flux rates and parameter values in the flux-balance Ca2+ model
    params.add('nu_PMCA', value=30., min=1, max=50.)
    params.add('nu_Serca', value=22.5, min=1., max=50.)
    params.add('kPMCA', value=0.45, min=0.001, max=1.)
    params.add('kSerca', value=0.105, min=0.001, max=1.)
    params.add('nu_INleak', value=0.03, min=0.001, max=1.)
    params.add('nu_ERleak', value=0.03, min=0.001, max=1.)
    params.add('nu_IP3', value=15., min=1., max=50.)
    params.add('d_1', value=0.13, min=0.001, max=1.)
    params.add('d_2', value=1.049, min=0.01, max=5.)
    params.add('d_3', value=0.9434, min=0.01, max=5.)
    params.add('d_5', value=0.08234, min=0.001, max=1.)
    params.add('a2', value=0.2, min=0.001, max=1.)
    params.add('IP3', value=0.3, min=0.001, max=1.)
    params.add('f_c', value=0.05, min=0.001, max=1.)
    params.add('f_ER', value=0.025, min=0.001, max=1.)
    params.add('gamma', value=9., min=1., max=20.)
    params.add('alpha', value=0.5, min=0.001, max=1.)

    params.add('Iapp', value=5.67, min=0.1, max=20.)
    params.add('sigma', value=1.0, min=0.0, max=5.)
    params.add('beta', value=1.5, min=0, max=2)
    params.add('lambda_glu', value=3., min=1., max=150.)
    params.add('A_glu', value=1, min=0, max=10.)
    params.add('tau_decay', value=0.1, min=0.1, max=20)
    params.add('lastrelease', value=-5, vary=False)

    return params