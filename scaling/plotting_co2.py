import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from plots_co2data_visfns import *

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.45
plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlesize']='small'

dpts = pd.read_csv('data_2025-10-20.csv',skip_blank_lines=True,
                   skiprows=1)
conv=lambda x: dt.datetime.strptime(x,"%Y-%m-%d").date().toordinal()
traw=dpts['CO2 Date'].apply(conv).to_numpy()
t=traw-traw[142]
ppm=dpts['CO2 PPM'].to_numpy()
fig1,(ax0,ax1,ax2)=plt.subplots(1,3,sharey=True,dpi=50,figsize=(13,4))
ax0.set_xscale("symlog",base=10);ax0.plot(t,ppm,'o-k',markersize=3,
        fillstyle=None,alpha=0.4);
ax0.set_xticks(ax0.get_xticks()[1:-1:2]);
ax0.add_artist(AnchoredText('Symmetric logarithm',loc='upper left'),)
ax1.set_xscale("asinh",base=10);ax1.plot(t,ppm,'o-k',markersize=3,
         fillstyle=None,alpha=0.4);
ax1.add_artist(AnchoredText(r'$\sinh^{-1}(x)$',loc='upper left'),)
ax1.set_xticks(ax0.get_xticks());
ax2.set_xscale("function",functions=(fnpow,fninvpow));ax2.plot(t,ppm,'o-k',
       markersize=3,fillstyle=None,alpha=0.4);
##fig1,(ax0,ax1,ax2)=plt.subplots(1,3,sharey=True,dpi=50,figsize=(13,4))
##ax0.set_xscale("function",functions=(fn,fninv));ax0.plot(t,ppm,'o-k',
##        markersize=3,fillstyle=None,alpha=0.4);
##ax0.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
##ax1.set_xscale("function",functions=(fn10,fninv10));ax1.plot(t,ppm,'o-k',
##    markersize=3,fillstyle=None,alpha=0.4);
##ax1.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
##ax2.set_xscale("function",functions=(fn,fninv));ax2.plot(t,ppm,'o-k',
##    markersize=3,fillstyle=None,alpha=0.4);
##ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.show()

