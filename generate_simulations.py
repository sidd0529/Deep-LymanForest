"""
This function generates different mock catalogs.
"""
import numpy as np
#import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pn
from astropy.cosmology import WMAP9 as cosmo
import time
import sys
import argparse #For commandline input

import simulatory_function as simf



# -------------------------------- Input arguments ---------------------------------------------------------
#print("Usage: python generate_simulations.py -start start_index -end end_index")
parser = argparse.ArgumentParser(description='Makes a bunch of simulation catalogs for Lyman alpha forest lensing.')
parser.add_argument('-start', default='0', help='Start index of mock.')
parser.add_argument('-end', default='3', help='End index of mock.')

args = parser.parse_args()

start_index = int(args.start)
end_index = int(args.end)


# -------------------------------- Start computation -------------------------------------------------------
t1 = time.time()

for i in range(start_index, end_index):
    #print(i, start_index, end_index)
    simf.make_a_mock(i)
    
t2 = time.time()
print (t2-t1)