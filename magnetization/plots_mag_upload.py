import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
#gen_curve_fitting is available in co2_levels repository 
from gen_curve_fitting import sup_fit,fn_c,fnder


def fn_sum(params,xax,ams):
    supmod = 0
    a = params['a'];xc=params['xc'];yc=params['yc']
    for i in range(ams+1):
        p=params['p'+str(i)];m=params['m'+str(i)]
        supmod+=p*(fn_c(xax-xc[i],a,m)+yc[i])
    return supmod
def fn_cparts(x,ap,m):
    xdep = (-27*x*m/2/ap+np.sqrt(729*(x*m/ap)**2+108/ap**3)/2)
    return np.array([-xdep**(1./3)/3,xdep**(-1./3)/ap])

def fn_sumparts(params,xax,ams):
    supmod = 0
    a = params['a'];xc=params['xc'];yc=params['yc']
    for i in range(ams+1):
        p=params['p'+str(i)];m=params['m'+str(i)]
        supmod+=p*(fn_cparts(xax-xc[i],a,m)+yc[i])
    return supmod

def fndersum(params,x,ams):
    der=0
    a = params['a'];xc=params['xc'];yc=params['yc']
    for i in range(ams+1):
        p=params['p'+str(i)];m=params['m'+str(i)]
        y=fn_c(x-xc[i],a,m)
        der+=p*m/(1+3*a*y**2)
    return der

def fnddersum(params,x,ams):
    der=0
    a = params['a'];xc=params['xc'];yc=params['yc']
    for i in range(ams+1):
        p=params['p'+str(i)];m=params['m'+str(i)]
        y=fn_c(x-xc[i],a,m)
        der+=-6*p*a*y*m**2/(1+3*a*y**2)**3
    return der
def fndddersum(params,x,ams):
    der=0
    a = params['a'];xc=params['xc'];yc=params['yc']
    for i in range(ams+1):
        p=params['p'+str(i)];m=params['m'+str(i)]
        y=fn_c(x-xc[i],a,m)
        der+=6*p*a*m**3*(15*a*y**2-1)/(1+3*a*y**2)**5
    return der
def fnddddersum(params,x,ams):
    der=0;
    a = params['a'];xc=params['xc'];yc=params['yc']
    for i in range(ams+1):
        p=params['p'+str(i)];m=params['m'+str(i)]
        y=fn_c(x-xc[i],a,m)
        der+=-360*p*m**4*a**2*y*(6*a*y**2-1)/(1+3*a*y**2)**7
    return der
    
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.45
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize']='small'

##dpts = pd.read_table('~/Downloads/mag/NewFile',sep=',',header=None)
##x,y=dpts[dpts.columns[0]].to_numpy(),dpts[dpts.columns[1]].to_numpy()
##_,params = sup_fit(x,y,ams)
##with open('irodov_ams_6.pickle', 'rb') as handle:
##    params = pickle.load(handle)



xder=np.linspace(x[0],x[-1],10000)
ams = len(params['xc'])-1

plt.plot(x,y,'o',xder,fn_sum(params,xder,ams)+params['offset'])
plt.show()

#Newton-Raphson
der=fndersum(params,xder,ams)
xr=xder[np.where(der==np.max(der))[0][0]]
for i in range(10):
    xr -= fnddersum(params,xr,ams)/fndddersum(params,xr,ams)
x0=xr
y0=fn_sum(params,xr,ams)+params['offset']
m=fndersum(params,x0,ams)

der2=fnddersum(params,xder,ams)
xr1=xder[np.where(der2==np.max(der2))[0][0]]
xr2=xder[np.where(der2==np.min(der2))[0][0]]
for i in range(10):
    xr1 -= fndddersum(params,xr1,ams)/fnddddersum(params,xr1,ams)
    xr2 -= fndddersum(params,xr2,ams)/fnddddersum(params,xr2,ams)
y1=fn_sum(params,xr1,ams)+params['offset']
y2=fn_sum(params,xr2,ams)+params['offset']
a1=(-y1+y0+m*xr1-m*x0)/(y1-y0)**3
a2=(-y2+y0+m*xr2-m*x0)/(y2-y0)**3
print('a_interval:',a1,a2)
print('m0:',m,'x0:',x0,'y0:',y0)
