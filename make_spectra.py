import numpy as np
#import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pn
import time
import sys


class func_interp_from_ln :
    def __init__(self, mylnfunc):
        self.lnfunc = mylnfunc
        
    def __call__(self,k):
        return np.exp(self.lnfunc(np.log(k)))

#from astropy.cosmology import WMAP9 as cosmo
import gaussian_fields as gauss


def make_correlation_functions(Rmin,plot=True):
    ##################################################################
    # 2D matter correlation function
    ##################################################################
    print("setup correlation function ...")

    cdf = pn.read_csv("correlation.csv")  
    
    #print("read in some kind of file i guess")
    
    
    cdf = cdf[cdf['z'] == 2.0]

    corrfunc =  interpolate.interp1d(cdf['r'],cdf['C'],fill_value=0,bounds_error=False)

    # flatten the correlation below a scale
    Cmax = corrfunc(Rmin)
    cdf.loc[cdf['r'] < Rmin,'C'] = Cmax
    corrfunc =  interpolate.interp1d(cdf['r'],cdf['C'],fill_value=0,bounds_error=False)

    cdf['dC'] = np.gradient(cdf['C'],cdf['r'])

    dcorrfunc =  interpolate.interp1d(cdf['r'],cdf['dC'],fill_value=0,bounds_error=False)

#    if(plot) :
#        ### plot correlation function
#        plt.subplot(111)
#        plt.title('correlation function')
#        plt.xlabel('r (Mpc)')
#        ls = np.arange(1,500)*200/499.
#        corr = abs(corrfunc(ls))
#        plt.loglog(ls,corr)
#        dcorr = abs(dcorrfunc(ls))
#        plt.loglog(ls,dcorr)
#        plt.show()
    
    #sys.exit()
    return corrfunc,dcorrfunc

def make_potential_field(Ngrid,ang_size,factor=1):
            
    ##################################################################
    # import power spectra
    ##################################################################

    df_power_phi = pn.read_csv("kappa_power.csv")
    df_power_phi['lnP_phi'] = df_power_phi['lnP_phi'] + np.log(factor)
    lnpow = interpolate.interp1d(df_power_phi['ln(l)'],df_power_phi['lnP_phi']
    ,fill_value=-100,bounds_error=False)
    P_phi =  func_interp_from_ln(lnpow)
   
    #print('power',P_phi(1.0e2))

    phi_field = gauss.gaussian_random_field2D(Pk=P_phi,size=(Ngrid,Ngrid)
    ,lengths=(ang_size,ang_size))
    
    return phi_field.real,df_power_phi

def make_deflections(phi_field,ang_size,Ngrid,Nqso,iseed_qso,iseed_potIGM) :

    print("calculate deflection field ...")
    start_time = time.time()

    #Nqso = int(Nqso*overfactor**2 + 0.5)
    alpha_field = np.gradient(phi_field,ang_size/Ngrid)

    print('var alpha = ',np.sqrt(np.var(alpha_field[1]))*180*60/np.pi,' arcmin')

    ### make interpolators for deflection
    x = np.arange(0, ang_size, ang_size/Ngrid)
    y = np.arange(0, ang_size, ang_size/Ngrid)

    alpha_interp_x = interpolate.interp2d(x,y,alpha_field[1])
    alpha_interp_y = interpolate.interp2d(x,y,alpha_field[0])

    print(("--- %s seconds ---" % (time.time() - start_time)))

    ##################################################################
    ### displace positions
    ##################################################################
    print("Displace positions ....")

    ### get random positions - these are the lensed positions
   
    np.random.seed(iseed_qso) # this is the only time this seed is used

    ### we use a buffer to eliminate- the size is picked at random and not checked 
    
    Xqso = ((np.random.rand(Nqso)*0.8)+0.1)*ang_size
    Yqso = ((np.random.rand(Nqso)*0.8)+0.1)*ang_size
    
    np.random.seed(iseed_potIGM) #back to the other seed
    
     ### interpolate deflection
    ax = []
    ay = []
    for i in range(len(Xqso)) :
        ax.append(alpha_interp_x(Xqso[i],Yqso[i]))
        ay.append(alpha_interp_y(Xqso[i],Yqso[i]))
    
    print (ax[0],ax[1])
    
    ### flatten type
    ax = [item for sublist in ax for item in sublist]
    ay = [item for sublist in ay for item in sublist]

    ### unlensed positions are '_source'
    X_source = np.add(Xqso,ax)
    Y_source = np.add(Yqso,ay)
    
    # now iterate 10 times to figure out the unlensed positions given the lensed ones
    
    for i in range(0,9):

        ax = []
        ay = []
        
        for i in range(len(Xqso)) :
            ax.append(alpha_interp_x(X_source[i],Y_source[i]))
            ay.append(alpha_interp_y(X_source[i],Y_source[i]))

        #print (ax[0],ax[1])
    
        ### flatten type
        ax = [item for sublist in ax for item in sublist]
        ay = [item for sublist in ay for item in sublist]

        X_source = np.add(Xqso,ax)
        Y_source = np.add(Yqso,ay)
    
        #### need to cull unlensed positions that are outside region
    out = []
    for i in range(0,Nqso) :
        if( (X_source[i] < 0 or X_source[i] > ang_size) 
        or (Y_source[i] < 0 or Y_source[i] > ang_size) ) :
            out.append(i)

    X_source = np.delete(X_source,out)
    Y_source = np.delete(Y_source,out)
 
    print((Nqso - len(X_source)," unlensed sources out of bounds."))
    
    Xqso = np.delete(Xqso,out)
    Yqso = np.delete(Yqso,out)

    Nqso = len(Xqso)
    
    return Xqso,Yqso,X_source,Y_source

def make_spectra(Xqso,Yqso,X_source,Y_source,Ngrid,Nqso,Nfrequ,zlength
                 ,ang_size,ds_vec,pix_sigma,plots = False) :

    start_time = time.time()
    print("   setup power spectum ...")

    df = pn.read_csv("power.csv")

    Ds = max(ds_vec)
    
    ranges = [min(X_source)*Ds,min(Y_source)*Ds,max(X_source)*Ds,max(Y_source)*Ds]
    maxrange = max(ranges[2]-ranges[0],ranges[3]-ranges[1])
    
    border = ang_size*0.1
    
    ## the cube must be much larger than the actual volume to avoid 
    ## problems with the correlation function
    
    factor = 2
    width = ang_size*Ds*factor
    Nbig = Ngrid*factor
    
    Nlength = np.max([Nfrequ*factor,10])
    depth = zlength*Nlength/Nfrequ

    
    
    lnpow = interpolate.interp1d(df['ln(k)'],df['lnP'],fill_value=-100
                                    ,bounds_error=False)
    P_delta =  func_interp_from_ln(lnpow)

    ##################################################################
    ### make absorption cube
    ##################################################################
    print("   make absorption cube ..")

    cube = gauss.gaussian_random_field3D(Pk=P_delta,size=(Nbig,Nbig,Nlength)
    ,lengths=(width,width,depth))

    cube = cube.real
    
    
    ###################################################################
    ### test correlation function
    ##################################################################
 
    if(plots) :
        l = 1.0e-3*10**( 4.5*np.arange(0,1001)/1000. )
        P = P_delta(l)
    
        #plt.plot(l,P)
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.xlabel('k (Mpc$^-1$)')
        #plt.ylabel('P(k)')
        #plt.title('matter power spectrum')
    
        #plt.ylim(1.0e-2,1.0e4)
        #plt.show()

        print("   making correlation function ..")
        corr2d,dr = gauss.correlation2D(cube[:Ngrid,:Ngrid,0])
        
        for i in range(1,Nfrequ) :
            tmp,dr = gauss.correlation2D(cube[:Ngrid,:Ngrid,i])
            corr2d += tmp

        dr *= width/Nbig
        corr2d /= Nfrequ
        #plt.imshow(corr2d)
        #plt.colorbar()
    
        #plt.show()
    
        #plt.scatter(dr,corr2d)
        #plt.show()
    
        print((np.max(corr2d),np.min(corr2d)))
 
    #sys.exit()
    
    ###################################################################
    ### cut out a more smaller cube that is aproximately Ngrid x Ngrid x Nfrequ
    ##################################################################

    Nnew = int(Nbig*maxrange/width + 10)
    cube = cube[:Nnew,:Nnew,:Nfrequ]
    
    ##################################################################
    ### interpolate to spectra
    ## *** this is not quite right ****
    ##################################################################

    ### interpolation cube
    planes = []
    #x = np.arange(0,ang_size*Ds,ang_size*Ds/Nbig)
    #y = np.arange(0,ang_size*Ds,ang_size*Ds/Nbig)
    #x = np.arange(ranges[0]-border,ranges[0] + width + border,width*padding/Nbig)
    #y = np.arange(ranges[1]-border,ranges[1] + width + border,width*padding/Nbig)
 
    
    steps = np.array(list(range(Nnew)))*width/Nbig
    border = 5*maxrange/width
    x = ranges[0] + steps - border
    y = ranges[1] + steps - border
    for i in range(Nfrequ) :
        planes.append(interpolate.interp2d(x,y,cube[:,:,i]))

    spectra = np.empty((Nqso,Nfrequ),dtype='d')
    for i in range(Nqso) :
        for j in range(Nfrequ) :
            spectra[i,j] = planes[j](X_source[i]*ds_vec[j],Y_source[i]*ds_vec[j])
            #if(spectra[i,j] > 1) :
            #    spectra[i,j] = 1

    spectra += np.random.normal(size = np.shape(spectra))*pix_sigma
    
    print(("   --- %s seconds ---" % (time.time() - start_time)))

    return spectra,cube[:,:,0].astype(float)

