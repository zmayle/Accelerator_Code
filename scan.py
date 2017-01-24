#Zach Mayle
#scan.py
import math
import matplotlib.pyplot as plt
import numpy as np
from accel_simulator import *

"""Run this script to generate a contour plot of energy spread vs. synchronous phase and r56. Prints
the ideal r56 and synchronous phase values to the console along with the least possible energy spread."""

if __name__ == '__main__':
    re=10.0
    espread=re*(.001)
    #pspread=2.0
    pspread=.6
    npart=1000
    n_bunches=50
    number_cavities=8
    wave=100.0
    cycles=3
    E_gain=40.0
    cav_gain=E_gain/number_cavities
    r56, sp, energy_spread=scan(cycles,number_cavities,wave,cav_gain,re,espread,pspread,npart,n_bunches)
    print "Ideal r56: "+str(r56)
    print "Ideal Synchronous Phase: "+str(sp)
    print "Least Possible Energy Spread: "+str(energy_spread)