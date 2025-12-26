import numpy as np
m=2.7393439993652122/365
a=4.975564707676572e-05
def fnelem(x,a,m):
    n=1
    t1 = (2*n+1)**(1/n+2)
    t2 = t1*x*m/2/a
    t3 = -t2+np.sqrt(t2**2+t1/(a**(1/n+2)))        
    t4 = t3**(1./(2*n+1))
    return 1/t4/(a**(1/n))-t4/((2*n+1)**(1/n))

def fnbig(x,a,m):
    xdep=(-27*x*m/2/a+np.sqrt(729*(x*m/a)**2+108/a**3)/2)
    xappr=xdep**(1./3)
    for i in  np.where(xdep<1e-6):
        xsmall=4/(27*x[i]**2*m**2*a)
        xappr[i]=(1/2)*2**(2/3)*xsmall**(1/3)-(1/24)*2**(2/3)*xsmall**(4/3)+(5/288
            )*2**(2/3)*xsmall**(7/3)-(13/1296)*2**(2/3)*xsmall**(10/3)+(209/31104
            )*2**(2/3)*xsmall**(13/3)
        xappr[i]=xappr[i]*np.sign(27*x[i]*m/2/a)*np.abs(27*x[i]*m/2/a)**(1./3)
    return -xappr/3+1/a/xappr

def fn(x):
##    return fnelem(x,a,m)
    return fnbig(x,a,m)
def fninv(y):
    return (a*y**3+y)/m
def fn10(x):
##    return fnelem(x,10*a,m)
    return fnbig(x,10*a,m)
def fninv10(y):
    return (10*a*y**3+y)/m
def fn100(x):
##    return fnelem(x,100*a,m)
    return fnbig(x,100*a,m)
def fninv100(y):
    return (100*a*y**3+y)/m

def fnpow(x):
    return np.sign(m*x)*np.abs(m*x)**(1./3)
def fninvpow(x):
    return (m*x)**3

def fnxfit(x):
    return 2.7393/365*x
def fninvxfit(x):
    return 365*x/2.7393
def fnyfit(x):
    return (x+a*x**3)/m
def fninvyfit(x):
    return fnelem(x,a,m)


def finset(traw,ppm,ax,axlims,insetsize):
    xe1, xe2, ye1, ye2 = axlims
    axins = ax.inset_axes(insetsize,xlim=(xe1, xe2), ylim=(ye1, ye2))
    axins.ticklabel_format(style='sci', axis='x', scilimits=(0,0),
                         useMathText=True)
    axins.plot((traw-traw[0]+1),ppm,'o-k',markersize=2,
         fillstyle=None,alpha=0.4)
    axins.xaxis
    axins.annotate("1948",xy=(342598, 310.3), xytext=(342598,315),
             arrowprops=dict(arrowstyle="->"))
    ax.indicate_inset_zoom(axins, edgecolor="black")
    return ax

