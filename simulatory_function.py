import numpy as np
#import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pn
from astropy.cosmology import WMAP9 as cosmo
import time
import sys
import gaussian_fields as gauss
import make_spectra as sp
import scipy
from scipy import stats
import os, errno



def make_a_mock (iseed_potIGM):

    ##################################################################
    ## input parameters  
    ##################################################################

    #np.random.seed(iseed_qso)          # random seed (set later now)

    iseed_qso = 12 #random seed for quasar positions - same for all realizations
    # input is iseed_potIGM, which is random seed for lya spectra and for lensing potential field

    np.random.seed(iseed_potIGM)          # set for now - will get changed for qso pos and then back

    ang_size = 0.5*np.pi/180.    # angular size of field in radians ! includes a buffer zone of 10% on all sides
    Ngrid = 16                   # 1d number of pixels in simulated field



    Nqso = 400                 # number of backlights
    sqrtNqso = Nqso**0.5
    Nfrequ = 20                # number of pixels in each spectra
    dz = 0.02                   # redshift 
    zs1 = 2.0                   # lowest redshift of forest
    Nl = 21                      # number of l-modes to a side
    Rmin = 0.1                   # minimum correlation length, flatten correlation function

    PlotJump = 3

    DFTorder = True          # order modes in standard DFT order
    fullF = False            # do only the diagonal values for Fisher matrix or the whole thing

    factor = 10.0              # factor by which the potential power is boosted for testing 
    Ndsets = 1                # number of simulated data sets
    pix_noise_factor = 1.1    # pixels are given uncorrelated noise that 
                              # is pix_noise_factor x the max of the correlation 
                              # function, C(Rmin), must be > 1
    ##################################################################


    # -------------------- Create output directories -----------------------------------------------------------
    try:
        os.makedirs('Data_nqso' + str(Nqso) + '_npix' + str(Nfrequ) + '_boost' + str(factor) + '_ngrid' + str(Ngrid) + '/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    try:
        os.makedirs('XY_nqso' + str(Nqso) + '_npix' + str(Nfrequ) + '_boost' + str(factor) + '_ngrid' + str(Ngrid) + '/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise



    DataDir = 'Data_nqso' + str(Nqso) + '_npix' + str(Nfrequ) + '_boost' + str(factor) + '_ngrid' + str(Ngrid) + '/'
    PositionDir = 'XY_nqso' + str(Nqso) + '_npix' + str(Nfrequ) + '_boost' + str(factor) + '_ngrid' + str(Ngrid) + '/'


    print ("Here is the seed:",iseed_potIGM)

    
    ##################################################################
    # some derived quantities
    ##################################################################

    zs2 = zs1 + dz*Nfrequ                   # highest redshift of forest
    Dss = cosmo.comoving_distance([zs1,zs2])
    Ds = Dss[1].value
    zlength = Dss[1].value - Dss[0].value
    ds_vec = np.arange(Dss[0].value, Dss[1].value,(Dss[1].value-Dss[0].value)/Nfrequ)


    ##################################################################
    # make the correlation function and its derivative
    ##################################################################

    corrfunc,dcorrfunc = sp.make_correlation_functions(Rmin)


    ##################################################################
    # Deflection field
    ##################################################################

    phi_field,df_power_phi = sp.make_potential_field(Ngrid,ang_size,factor = factor)

    Xqso,Yqso,X_source,Y_source = sp.make_deflections(phi_field,ang_size,Ngrid,Nqso,iseed_qso,iseed_potIGM)
    np.savetxt( PositionDir + 'Xqso_' + str(iseed_potIGM).zfill(6) + '.txt' , Xqso)
    np.savetxt( PositionDir + 'Yqso_' + str(iseed_potIGM).zfill(6) + '.txt' , Yqso)
    np.savetxt( PositionDir + 'Xsource_' + str(iseed_potIGM).zfill(6) + '.txt' , X_source)
    np.savetxt( PositionDir + 'Ysource_' + str(iseed_potIGM).zfill(6) + '.txt' , Y_source)    



    Nqso = len(Xqso)

    Cmax = corrfunc(Rmin)

    ##################################################################
    ### make simulated spectra
    ##################################################################
    print("creating fake spectra ...")
    start_time = time.time()


    for i in range(Ndsets) :

        print(( "  " + repr(i+1) +  " out of " +  repr(Ndsets) ))
        spectra,cube_im = sp.make_spectra(Xqso,Yqso,X_source,Y_source,Ngrid,Nqso
                                          ,Nfrequ,zlength,ang_size,ds_vec
                                          ,np.sqrt(pix_noise_factor*Cmax))
        


    print(("--- %s seconds ---" % (time.time() - start_time)))


    mock = [Ngrid, Nqso, Nfrequ, ang_size, iseed_potIGM, phi_field, Xqso, Yqso, spectra]

    import pickle

    filenamemaybe = DataDir + 'mock.' + str(iseed_potIGM).zfill(6)
    print (filenamemaybe)

    with open(filenamemaybe, 'wb') as fp:
        pickle.dump(mock, fp)


    return