#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 06:21:51 2022

@author: amin
"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors

def gen_color(cmap,n,reverse=False):
    '''Generates n distinct color from a given colormap.

    Args:
        cmap(str): The name of the colormap you want to use.
            Refer https://matplotlib.org/stable/tutorials/colors/colormaps.html to choose
            Suggestions:
            For Metallicity in Astrophysics: Use coolwarm, bwr, seismic in reverse
            For distinct objects: Use gnuplot, brg, jet,turbo.

        n(int): Number of colors you want from the cmap you entered.

        reverse(bool): False by default. Set it to True if you want the cmap result to be reversed.

    Returns: 
        colorlist(list): A list with hex values of colors.
    '''
    c_map = plt.cm.get_cmap(str(cmap)) # select the desired cmap
    arr=np.linspace(0,1,n) #create a list with numbers from 0 to 1 with n items
    colorlist=list()
    for c in arr:
        rgba=c_map(c) #select the rgba value of the cmap at point c which is a number between 0 to 1
        clr=colors.rgb2hex(rgba) #convert to hex
        colorlist.append(str(clr)) # create a list of these colors
    
    if reverse==True:
        colorlist.reverse()
    return colorlist
