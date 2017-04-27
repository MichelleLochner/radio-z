import numpy as np
import pandas as pd
from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib import rc
import glob
import pdb

import sys
sys.path.append('/Users/harrison/Dropbox/code_mcr/radio-z/radio_z/')
from hiprofile import *

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

cpal = ['#185aa9','#008c48','#ee2e2f','#f47d23','#662c91','#a21d21','#b43894','#010202']

plt.close('all') # tidy up any unshown plots

plot_dir = './plots/'
root_dir = './data/examples_band2/'

fig, axlist = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4.5, 3.75))
axlist = axlist.reshape([4])

lineid_list = glob.glob(root_dir+'*')
#lineid_list = lineid_list[-5:-1]
lineid_list = np.random.choice(lineid_list, size=4, replace=False)

for i,lineid in enumerate(lineid_list):

  linestore = pd.HDFStore(lineid)
  lineparms = linestore['parameters']
  linedata = linestore['data']
  lineev = linestore['evidence']

  z_true = linestore['summary']['True']['z']
  v0 = linestore['chain']['v0']
  z_pdf = -v0/(v0+3e5)
  if z_pdf.var() < 0.005:
    axlist[i].axvline(0, linestyle='--', color='m', zorder=1)
    axlist[i].hist((z_pdf-z_true)*1.e3, color='k', zorder=10, histtype='step', normed=True, range=(-4, 4), bins=50, label='$\ln(B) = $'+('%.2f' % lineev['Bayes factor']))#, color=cpal[i])
    #axlist[i].legend(loc='upper left', frameon=False, fontsize='x-small')
    axlist[i].text(0.5,3,'$\ln(B) = $'+('%.2f' % lineev['Bayes factor']), fontsize='small')

#plt.axvline(0, linestyle='--', color=cpal[3])
fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[:]], visible=False)
plt.setp([fig.axes[2].get_xticklabels()], visible=True)
plt.setp([fig.axes[3].get_xticklabels()], visible=True)
plt.setp([fig.axes[0].get_yticklabels()], visible=True)
plt.setp([fig.axes[2].get_yticklabels()], visible=True)
plt.xlim([-4, 4])
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.ylabel('$\mathrm{Probability}\, P(z - z_{\\rm true})$')
plt.xlabel('$10^{3}\\times\mathrm{Redshift \, Error} \, \Delta z$')
#plt.ylabel('$\mathrm{Probability}\, P(z - z_{\\rm true})$')
#plt.xlabel('$\mathrm{Redshift \, Error} \, \Delta z$')
plt.savefig(plot_dir+'multiline_zpdf.png', bbox_inches='tight', dpi=300)