'''
Plot posterior P(z) for all lines in a folder

run script from radio-z/plotting_scripts
data comes from root_dir
plots go into plot_dir
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc
import glob
import pdb

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

cpal = ['#185aa9','#008c48','#ee2e2f','#f47d23','#662c91','#a21d21','#b43894','#010202']

plt.close('all') # tidy up any unshown plots

plot_dir = '../plots/'
root_dir = '../data/examples_band2/'

lineid_list = glob.glob(root_dir+'*')
#lineid_list = lineid_list[-5:-1]
#lineid_list = np.random.choice(lineid_list, size=4, replace=False)

for i,lineid in enumerate(lineid_list):

  fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(4.5, 3.75))

  linestore = pd.HDFStore(lineid)
  lineparms = linestore['parameters']
  linedata = linestore['data']
  lineev = linestore['evidence']

  z_true = linestore['summary']['True']['z']
  v0 = linestore['chain']['v0']
  z_pdf = -v0/(v0+3e5)
  ax.axvline(0, linestyle='--', color='c', zorder=1)
  ax.hist((z_pdf-z_true)*1.e3, color='m', zorder=10, histtype='step', normed=True, range=(-4, 4), bins=50,
               label='$\ln(B) = $'+('%.2f' % lineev['Bayes factor'])+'\n$\mathrm{'+lineid.split('/')[-1]+'}$')#, color=cpal)
  ax.legend(loc='upper left', frameon=False, fontsize='x-small')

  plt.xlim([-4, 4])
  plt.ylabel('$\mathrm{Probability}\, P(z - z_{\\rm true})$')
  plt.xlabel('$10^{3}\\times\mathrm{Redshift \, Error} \, \Delta z$')
  plt.savefig(plot_dir+'line_{0}_zpdf.png'.format(lineid.split('/')[-1]), bbox_inches='tight', dpi=300)
  linestore.close()