import numpy.fft as fft
import numpy as np
#import matplotlib.pyplot as plt
#from enthought.mayavi import mlab

def fftIndgen(n):
    a = list(range(0, int(n/2)+1))
    b = list(range(1, int(n/2)))
    b.reverse()
    b = [-i for i in b]
    return a + b

def gaussian_random_field2D(Pk = lambda k : k**-4.0, size = (100,100),lengths=(1,1)):
    '''
    This function returns a 2 dimensional Gaussian random field with the desired power spectrum.
    :param Pk: A function returning the power spectrum as defined in continuous Fourier space.
    :param size: The number of pixels in each dimension.
    :param lengths: lengths of box in the units that the power spectrum is defined in
    :return: field in configuration space.
    '''

    vol = lengths[0]*lengths[1]
    dens = size[0]*size[1]/vol
    conv = np.array([2*np.pi/lengths[0],2*np.pi/lengths[1]])
    def Pk2(kx, ky):
        if kx == 0 and ky == 0:
            return 0.0
        return np.sqrt(dens * Pk(np.sqrt((kx*conv[0])**2 + (ky*conv[1])**2)) )

    noise = np.fft.fft2(np.random.normal(size = size))
    amplitude = np.zeros(size)
    for i, kx in enumerate(fftIndgen(size[0])):
        for j, ky in enumerate(fftIndgen(size[1])):
            amplitude[i, j] = Pk2(kx, ky)
    return np.fft.ifft2(noise * amplitude)

def gaussian_random_field3D(Pk = lambda k : k**-3.0, size = (50,50,50),lengths=(1,1,1)):
    '''
    This function returns a 3 dimensional Gaussian random field with the desired power spectrum.
    :param Pk: A function returning the power spectrum as defined in continuous Fourier space.
    :param size: The number of pixels in each dimension.
    :param lengths: lengths of box in the units that the power spectrum is defined in
    :return: field in configuration space.
    '''

    vol = lengths[0]*lengths[1]*lengths[2]
    dens = size[0]*size[1]*size[2]/vol
    conv = np.array([2*np.pi/lengths[0],2*np.pi/lengths[1],2*np.pi/lengths[2]])
    def Pk2(kx, ky, kz):
        if kx == 0 and ky == 0 and kz == 0:
            return 0.0
        #return np.sqrt(dens * Pk(np.sqrt(kx**2 + ky**2 + kz**2 ) ) )
        return np.sqrt(dens * Pk(np.sqrt((kx*conv[0])**2 + (ky*conv[1])**2 + (kz*conv[2])**2 ) ) )

    noise = np.fft.fftn(np.random.normal(size = size))
    amplitude = np.zeros(size)
    
    for i, kx in enumerate(fftIndgen(size[0])):
        for j, ky in enumerate(fftIndgen(size[1])):
            for k, kz in enumerate(fftIndgen(size[2])):
                amplitude[i, j, k] = Pk2(kx, ky, kz)
 
    return np.fft.ifftn(noise * amplitude)

def coorelationfunction3D(Pk = lambda k : k**-3, size = (50,50,50),maxlengths=(1,1,1)):
    '''
    This function returns a 3 dimensional correlation function given an input 
    power spectrum using the DFT.  The (0,0,0) mode within the box has zero 
    magnitude.
    :param Pk: A function returning the power spectrum as defined in continuous Fourier space.
    :param size: The number of pixels in each dimension.
    :param lengths: lengths of box in the units that the power spectrum is defined in
    :return: three dimesional array of the correlation function.
    '''

    lengths = 2*maxlengths
    vol = lengths[0]*lengths[1]*lengths[2]
    dens = size[0]*size[1]*size[2]/vol
    conv = np.array([2*np.pi/lengths[0],2*np.pi/lengths[1],2*np.pi/lengths[2]])

    def Pk_intern(kx, ky, kz):
        if kx == 0 and ky == 0 and kz == 0:
            return 0.0
        return dens * Pk(np.sqrt((kx*conv[0])**2 + (ky*conv[1])**2 + (kz*conv[2])**2 ) )

    amplitude = np.zeros(size,dtype=np.complex_)
    for i, kx in enumerate(fftIndgen(size[0])):
        for j, ky in enumerate(fftIndgen(size[1])):
            for k, kz in enumerate(fftIndgen(size[2])):
                amplitude[i, j, k] = Pk_intern(kx, ky, kz)

    am =  np.fft.ifftn(amplitude)
    am = am[0:size[0]/2,0:size[1]/2,0:size[2]/2]
    am = am.real
    return am


def correlation2D(map):
    '''
    Calculates the 2D correlation function of a 2D map by brute force.
    '''
    
    s = np.shape(map)
    corr2d = np.empty(s)
    dr = np.empty(s)
    
    x = np.reshape( np.repeat(np.arange(0,s[0]),s[1]) , s )
    y = np.reshape( np.tile(np.arange(0,s[1]),s[0]) , s )

    dr = np.sqrt( x*x + y*y )

    for sx in range(0,s[0]) :
        for sy in range(0,s[1]) :
            
            if(sx == 0 and sy == 0 ) :
                map2 = map
            elif(sx == 0) :
                map2 = map[:,:-sy]
            elif(sy == 0) :
                map2 = map[:-sx,:]
            else :
                map2 = map[:-sx,:-sy]
                 
            corr2d[sx,sy] = np.mean(map2*map[sx:,sy:])
        
    return corr2d,dr
    

