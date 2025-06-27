#! /usr/bin/python3
import numpy as np
import math
#import matplotlib as mpl
import matplotlib.pyplot as plt
#from scipy.interpolate import CubicSpline
#from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#from scipy import interpolate
#from scipy.interpolate import RegularGridInterpolator
import sys

fname=sys.argv[1] #data filename
fout=sys.argv[2] #output filename
s=int(sys.argv[3]) #data width
Pkform=sys.argv[4]

axtitle='P(k)='+Pkform
titlecolor='midnightblue'

scales=np.unique(np.genfromtxt(fname, usecols=(0),unpack=True))

axa_range=[min(scales),max(scales)]
axb_range=[min(scales),max(scales)]
[naxa,naxb]=[len(scales),len(scales)]

background_color='white' #'#001b23'
ax_color='black' #axes and text color
grid_color='black'
grid=[naxa,naxb] #data grid of nnR,nbins



#contours
#contours_vals=[-2,-1,0,1,2] 




def Colouring(ax):
    ax.set_facecolor(background_color) #graph color (inside graph)
    ax.grid(linestyle='--',color=grid_color,alpha=.5)
    ax.tick_params(axis="y",colors=ax_color) #axis tickslabels colors
    ax.tick_params(axis="x",colors=ax_color)
    ax.spines['left'].set_color(ax_color) #axis ticks colors
    ax.spines['right'].set_color(ax_color)
    ax.spines['bottom'].set_color(ax_color)
    ax.spines['top'].set_color(ax_color)
    return



def Set_ax(ax,title):
    Colouring(ax)

    ax.set_xlabel('x[pix]',color=ax_color,fontsize=14)
    ax.set_ylabel('y[pix]',color=ax_color,fontsize=14)
    ax.set_title(title,color=titlecolor,fontsize=15,alpha=1)
    return



def Set_cb(cf,ax,cbtitle):
    cb=fig.colorbar(cf,ax=ax)
    cb.remove()
    cb=fig.colorbar(cf,ax=ax)
    cb.set_label(cbtitle,color=ax_color,fontsize=13)
    cb.ax.tick_params(color=ax_color,labelsize=13) #colorbar ticks colors
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'),color=ax_color) #colorbar ticklabels colors
    cb.outline.set_edgecolor(ax_color) #colorbar edgecolor
    return cb





#collecting data and plotting--------------------------------------------------
val_flat=np.genfromtxt(fname,usecols=2,unpack=True)
val=val_flat.reshape([s,s])


fig, ax = plt.subplots(figsize=(6,5)) #figsize=(50,20)
fig.patch.set_facecolor(background_color) #background color (outside graph)

ax.set_box_aspect(1) #square size

Set_ax(ax,axtitle)
cf=plt.imshow(np.transpose(val[:,::-1]), cmap='cubehelix',interpolation = 'none',aspect=1)#,norm=mpl.colors.LogNorm(vmin=np.amin(val),vmax=np.amax(val)))
cb=Set_cb(cf,ax,'value')
#contours = plt.contour(np.arange(0,s-1,1),np.arange(0,s-1,1), np.transpose(val[:,::-1]),contours_vals, linestyles='--',colors='cyan')
#plt.clabel(contours, inline=True, fontsize=11)


plt.savefig(fout,bbox_inches='tight')