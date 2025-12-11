import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from gen_curve_fitting import sup_fit,fn_c,fnder
from scipy.optimize import fsolve
from scipy.integrate import quad

def fn_sum(params,xax,ams):
    supmod = 0
    a = params['a']
    for i in range(ams+1):
        supmod+=params['p'+str(i)]*(fn_c(xax-params['xc'][i],a,
                params['m'+str(i)])+params['yc'][i])
    return supmod
def fndersum(params,x,ams):
    der=0
    a = params['a'];xc=params['xc'];yc=params['yc']
    for i in range(ams+1):
        p=params['p'+str(i)];m=params['m'+str(i)]
        y=fn_c(x-xc[i],a,m)
        der+=p*m/(1+3*a*y**2)
    return der

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.45
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize']='small'

##dpts = pd.read_table('~/Downloads/mag/NewDir2/B(H).Ibrayeva.LNGT72.tab')
##dpts = pd.read_csv('~/Downloads/mag/NewDir2/B(H).Altair_Flux.Example.csv')
dpts = pd.read_csv('~/Downloads/mag/MnZn ferrite.csv',sep=';')

x,y=dpts[dpts.columns[0]],dpts[dpts.columns[1]]
x1=x[0:np.where(x==np.min(x))[0][0]+1].to_numpy()
y1=y[0:np.where(x==np.min(x))[0][0]+1].to_numpy()
x2=x[np.where(x==np.min(x))[0][0]:].to_numpy()
y2=y[np.where(x==np.min(x))[0][0]:].to_numpy()

x=x1;y=y1
ams=1
with open('mnzn_mag.pickle', 'rb') as handle:
    params,params2 = pickle.load(handle)
##yeval,params = sup_fit(x,y,ams)
xc=params['xc'];yc=params['yc']

xder1=np.linspace(x1[0],x1[-1],10000)
xder2=np.linspace(x2[0],x2[-1],10000)
##yeval2,params2 = sup_fit(x2,y2,ams)
yeval = fn_sum(params,x1,ams)+params['offset']
yeval2 = fn_sum(params2,x2,ams)+params2['offset']

#Newton-Raphson
xr1=x1[np.where(yeval==np.max(yeval))[0][0]]
xr2=x1[np.where(yeval==np.min(yeval))[0][0]]
for i in range(10):
    xr1 -= (fn_sum(params,xr1,ams)+params['offset']-fn_sum(
        params2,xr1,ams)-params2['offset'])/(fndersum(params,xr1,ams)-
            fndersum(params2,xr1,ams))
    xr2 -= (fn_sum(params,xr2,ams)+params['offset']-fn_sum(
        params2,xr2,ams)-params2['offset'])/(fndersum(params,xr2,ams)-
            fndersum(params2,xr2,ams))
ymin=fn_sum(params,xr1,ams)+params['offset']
ymax=fn_sum(params,xr2,ams)+params['offset']
print(xr1,xr2,ymin,ymax)

area1 = lambda x:fn_sum(params,x,ams)+params['offset']
area2 = lambda x:fn_sum(params2,x,ams)+params2['offset']
print(quad(area1,xr2,xr1)[0]-quad(area2,xr2,xr1)[0])

fig,ax = plt.subplots(1,1)
ax.plot(x1,y1,'xk',x2,y2,'ok',alpha=0.7,fillstyle='none');
ax.plot(x1,yeval,'k');ax.plot(x2,yeval2,'k',lw=3,alpha=0.5)
##ax.plot(xc,yc,'dk',xmax,ymax,'dr')
ax.legend(['Data\n (opposing)','Data','Fit 1','Fit 2'])
plt.show()
