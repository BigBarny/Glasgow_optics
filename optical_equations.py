from mpmath import nsum, exp, inf, power,factorial, gamma, gammainc
import cmath
import numpy as np
import math
import scipy.integrate as integrate

def disp_units(value, unit):
    units = [['n',1e-9],['u',1e-6],['m',1e-3],['k',1e+3],['M',1e+6]]
    select = {}
    for x in units:
        select[x[0]] = x[1]
    return str(value/select[unit])+ ' '+unit 

def waist2pow(r,w):
    pr = 1-math.pow(math.e,-2*(r/w)**2)
    return pr

def z2waist(wv,w0,z):
    zr = math.pi*(w0**2)/wv
    return w0*math.sqrt(1+(z/zr)**2)


def pow2waist(app,pin,pout):
    w = complex(0,1)*math.sqrt(2)*app/(cmath.sqrt(cmath.log(1-(pout/pin))))
    return w.real
    

def waist(wv,zr):
    return math.sqrt((zr*wv)/math.pi)
    
def radius(w0,z,zr):
    return w0*math.sqrt(1+(z/zr)**2)


def Dispersion(N,dndT,n,k,rho,c,beta,fc,w0,w1,lam1,IS,source='surf'):
    """
    Credit to Angus? for his furthr insight into these equations

    PCI only atm!
    Determine phase and expected AC signal from initial parameters

    LIMITATIONS:
    1. No thermal expansion effects, dn/dT only
    Material properties required for calculations:

    INPUTS
    For each material:
    dndT : dn/dT at probe wavelength (coefficient of thermal expansion should be about or smaller than dn/dT to neglect elasto-optic contributions to dn/dT)
    n : index of refraction 
    k : thermal conductivity
    rho : density
    c : specific heat or D: thermal diffusivity 
    
    For setup:
    beta : Crossing angle in radians (Assuming pump is at normal incidence)
    fc : Chopping frequency
    w0 : Pump waist
    w1 : Probe waist
    lam1 : Probe wavelength 
    ??? IS : distance between crossing point and signal
    method : PCI only for now
    source : surf or bulk

    OUTPUTS
    phi : phase
    S : signal


    """
    #Normalised parameters

    zt = 1000*math.pi*(w0)**2/(2*lam1) # rayleigh range
    ep = 0.5*(w0/w1)**2 #rayleigh ratio
    zeta = IS/zt #norm distance
    tau = (w0**2)*(rho*c)*(1/(8*k)) #thermal time took out the *1e-12

    ft = (1/(2*math.pi*tau)) # thermal relaxation freq
    omg = fc/ft #norm freq

    #Photothermal Coeeficient

    if source=='bulk':
        A = math.sqrt(math.pi/8)*(w0/lam1)*(dndT*n/k)*(1/math.sin(beta))*100 # 100 offset from units?
        Dispersion = lambda tau: ((1/cmath.sqrt(1+tau + (zeta/(ep*zeta + complex(0,1))))).imag)*(cmath.exp              (-complex(0,1)*omg*tau)).real
        Dispersion_im = lambda tau: ((1/cmath.sqrt(1+tau + (zeta/(ep*zeta + complex(0,1))))).imag)*                     (cmath.exp(-complex(0,1)*omg*tau)).imag

        D,err = integrate.quad(Dispersion,0,N*(math.pi/omg))
        D_im,err_im = integrate.quad(Dispersion_im,0,N*(math.pi/omg))

        D_abs = abs(complex(D,D_im))
        D_arg = np.angle(complex(D,D_im),deg=True)
    else:
            # i.e source == 'surface'
        A = math.sqrt(math.pi/8)*(w0/lam1)*(dndT/k)*100 # 100 offset from units?
        Dispersion = lambda tau: ((1/(1+tau + (zeta/(ep*zeta + complex(0,1))))).imag)*(cmath.exp                        (-complex(0,1)*omg*tau)).real
        Dispersion_im = lambda tau: ((1/(1+tau + (zeta/(ep*zeta + complex(0,1))))).imag)*                               (cmath.exp(-complex(0,1)*omg*tau)).imag

        D,err = integrate.quad(Dispersion,0,N*(math.pi/omg))
        D_im,err_im = integrate.quad(Dispersion_im,0,N*(math.pi/omg))

        D_abs = abs(complex(D,D_im))
        D_arg = np.angle(complex(D,D_im),deg=True)
        
        #print(D)


        """
        #Journal - PCI Technique for Thermal Absorption Measurements
        C = 1
        #L is optical thickness of optic?
        L = 1e-3
        Wp = 1e-3
        dphi = 60
        #dI/I is the normalised intensity change i.e            distrotion
        dist = 2*dphi/(z/zt + zt/z) 
        absorption = c*dndT*(lam1/L)*((w0**2)/Wp)*dist*fc
        #print(absorption)
        """
        #print(A)
    return [A*D_abs,D_arg]

def Aperture_offset(a,w,d_range):
    """Calculate power drop from 1D Aperture offsets
    INPUTS
    a : aperture width
    w : beam waist
    d_range : np.array of offsets

    OUTPUTS
    Scaled power 
    """
    Poff = np.asarray([[d,float(nsum(lambda x: power(2,x)*power(d,2*x)*gammainc(x+1,0,2*(a/w)**2)/(power(w,2*x)*factorial(x)**2), [0, inf]))*np.exp(-2*(d/w)**2)] for d in d_range])
    return Poff