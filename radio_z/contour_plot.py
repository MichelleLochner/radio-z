#  Code to plot a contour from an MCMC chain
#  Author: Michelle Lochner (2013)

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


#Finds the 95% and 68% confidence intervals, given a 2d histogram of the likelihood
def findconfidence(H):
    H2 = H.ravel()
    H2 = np.sort(H2)


    # Loop through this flattened array until we find the value in the bin which contains 95% of the points
    tot = sum(H2)
    tot95=0
    tot68=0

    # Changed this to 68% and 30% C.I
    for i in range(len(H2)):
        tot95 += H2[i]
        if tot95 >= 0.05*tot:
            N95 = H2[i]
            #print i
            break

    for i in range(len(H2)):
        tot68 += H2[i]
        if tot68>=0.32*tot:
            N68 = H2[i]
            break   
    return max(H2),N95,N68

#Given a chain, labels and a list of which parameters to plot, plots the contours
# Arguments:
# chain=an array of the chain (not using weights, i.e. each row counts only once)
# p= a list of integers: the two parameters you want to plot (refers to two columns in the chain)
#kwargs: labels= the labels of the parameters (list of strings)
#               col=a tuple of the two colours for the contour plot
#               line=boolean whether or not to just do a line contour plot
def contour(chain,p,**kwargs):
    binsize=50
    if ('weights' in kwargs) & (len(kwargs['weights'])!=0):
        H, xedges, yedges = np.histogram2d(chain[:,p[0]],chain[:,p[1]], weights=kwargs['weights'], bins=(binsize,binsize))
    else:
        H, xedges, yedges = np.histogram2d(chain[:,p[0]],chain[:,p[1]], bins=(binsize,binsize))
    
    x=[]
    y=[]
    z=[]
    for i in range(len(xedges[:-1])):
        for j in range(len(yedges[:-1])):
            x.append(xedges[:-1][i])
            y.append(yedges[:-1][j])
            z.append(H[i, j])

    if 'smooth' in kwargs:
        SMOOTH=True
        smth=kwargs['smooth']
        if smth==0:
            SMOOTH=False
    else:
        SMOOTH=True
        smth=10e5
    if SMOOTH:
        sz=50
        spl = interpolate.bisplrep(x, y, z,  s=smth)
        X = np.linspace(min(xedges[:-1]), max(xedges[:-1]), sz)
        Y = np.linspace(min(yedges[:-1]), max(yedges[:-1]), sz)
        Z = interpolate.bisplev(X, Y, spl)
    else:
        X=xedges[:-1]
        Y=yedges[:-1]
        Z=H
    
    #I think this is the weird thing I have to do to make the contours work properly
    X1=np.zeros([len(X), len(X)])
    Y1=np.zeros([len(X), len(X)])
    for i in range(len(X)):
        X1[ :, i]=X
        Y1[i, :]=Y
    X=X1
    Y=Y1
    
    N100,N95,N68 = findconfidence(Z)

    if 'col' in kwargs:
        col=kwargs['col']
    else:
        col =('#a3c0f6','#0057f6') #A pretty blue
        


    if 'line' in kwargs and kwargs['line']==True:
        plt.contour(X, Y,Z,levels=[N95,N68,N100],colors=col, linewidth=100)
    else:
        plt.contourf(X, Y,Z,levels=[N95,N68,N100],colors=col)
    if 'labels' in kwargs:
        labels=kwargs['labels']
        plt.xlabel(labels[0],fontsize=22)
        plt.ylabel(labels[1],fontsize=22)
    #plt.show()

def triangle_plot(chain,params=[],labels=[],true_vals=[],best_params=[],smooth=5e3,weights=[],rot=0):
    """
        Plots the triangle plot for a sampled chain.
        chain = Input chain_
        params = List of indices of parameters, otherwise every column of chain is used
        labels = Labels for parameters
        true_vales = If provided, plots the true values on the histograms and contours
        best_params = List of lists for each parameter (mean, minus uncertainty, plus uncertainty) plotted on histograms
        smooth = Smoothing scale for the contours. Contour will raise warning is this is too small. Set to 0 for no smoothing.
        weights = If the chain needs reweighting before histogramming
        rot = Rotation of labels for plots
    """
    fntsz=18
    if len(params)==0:
        #If a list of parameter indices is not explicitly given, assume we plot all columns of chain except the last
        # (assumed to be likelihood)
        params=range(len(chain[0,:-1]))
    if len(labels)==0:
        labels=['%d' %i for i in range(len(params))]


    for i in range(len(params)):
        plt.subplot(len(params),len(params),i*(len(params)+1)+1)
        #Plot the histograms
        if len(weights)!=0:
            plt.hist(chain[:,params[i]],25,weights=weights,facecolor='#a3c0f6')
        else:
            plt.hist(chain[:,params[i]],25,facecolor='#a3c0f6')
        if len(true_vals)!=0:
            plt.plot([true_vals[i],true_vals[i]],plt.gca().get_ylim(),'k',lw=2.5)
        if len(best_params)!=0:
            plt.plot([best_params[i][0],best_params[i][0]],plt.gca().get_ylim(),'r',lw=2.5)
            plt.plot([best_params[i][0]+best_params[i][2],best_params[i][0]+best_params[i][2]],plt.gca().get_ylim(),'r--',lw=2.5)
            plt.plot([best_params[i][0]-best_params[i][1],best_params[i][0]-best_params[i][1]],plt.gca().get_ylim(),'r--',lw=2.5)
        plt.ticklabel_format(style='sci',scilimits=(-3,5))
        plt.xticks(rotation=rot)

        #Plot the contours
        for j in range(0,i):
            plt.subplot(len(params),len(params),i*(len(params))+j+1)
            contour(chain,[params[j],params[i]],smooth=smooth,weights=weights)
            if len(true_vals)!=0:
                plt.plot([true_vals[j]],[true_vals[i]],'*k',markersize=10)
            plt.ticklabel_format(style='sci',scilimits=(-3,5))
            plt.xticks(rotation=rot)
        plt.ticklabel_format()
        plt.tight_layout()


    for i in range(len(params)):
        ax=plt.subplot(len(params),len(params),len(params)*(len(params)-1)+i+1)
        ax.set_xlabel(labels[i])
        ax=plt.subplot(len(params),len(params),i*len(params)+1)
        ax.set_ylabel(labels[i])
        plt.tight_layout()

##Testing all functionality
#c=np.loadtxt('chain_2d_gaussian.txt')
#contour(c,[0,1], labels=['1', '2'],line=False)
#plt.show()
