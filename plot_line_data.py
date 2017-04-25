# source activate mpl34

import numpy as np
import pandas as pd
from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib import rc
import pdb

import sys
sys.path.append('/Users/harrison/Dropbox/code_mcr/radio-z/radio_z/')
from hiprofile import *

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

cpal = ['#185aa9','#008c48','#ee2e2f','#f47d23','#662c91','#a21d21','#b43894','#010202']

plt.close('all') # tidy up any unshown plots

def makePlot(line_id, root_dir, plot_dir ='./', drawwidth=300):

  linestore = pd.HDFStore(root_dir+line_id+'.hdf5')
  lineparms = linestore['parameters']
  linedata = linestore['data']

  v = linedata['v'].values
  v0 = lineparms['v0'].values
  psi = linedata['psi'].values
  v_channel_width = abs(v[1]-v[0]) 
  theoryline = LineProfile(v0,
                       lineparms['w_obs_20'].values[0], lineparms['w_obs_50'].values[0], lineparms['w_obs_peak'].values[0],
                       lineparms['psi_obs_max'].values[0], lineparms['psi_obs_0'].values[0])

  v0_idx = np.where(abs((v-v0))==np.min(abs(v-v0)))[0][0]

  n_drawchannels = int(drawwidth/v_channel_width)
  v_plot = v[v0_idx - n_drawchannels:v0_idx + n_drawchannels]
  psi_plot = psi[v0_idx - n_drawchannels:v0_idx + n_drawchannels]
  v_good = v_plot

  psi_plot = psi[v0_idx - n_drawchannels:v0_idx + n_drawchannels]
  psi_good = theoryline.get_line_profile(v_good)
  plt.close('all')
  plt.figure(1, figsize=(4.5, 3.75))
  plt.plot(v_plot*1.e-3, psi_plot*1.e3, c='k', label='$\mathrm{SKA-like}$')
  plt.plot(v_good*1.e-3, psi_good*1.e3, c='c', label='$\mathrm{Model}$', lw=2)
  #plt.text(-200, 0.9*psi_plot.max()*1.e3, ('$\mathrm{SNR} = %.2f$' % line.snr))
  plt.xlabel('$\\times10^{3} \, v \, [\mathrm{kms}^{-1}]$')
  plt.ylabel('$\Psi \, [\mathrm{mJy}]$')
  plt.xlim([v_plot[0]*1.e-3, v_plot[-1]*1.e-3])
  plt.legend(frameon=False, fontsize='small')
  plt.savefig(plot_dir+'line_{0}.png'.format(line_id), bbox_inches='tight', dpi=160)

  psi_good_large = theoryline.get_line_profile(v)
  plt.close('all')
  plt.figure(2, figsize=(4.5, 3.75))
  plt.plot(v*1.e-3, psi*1.e3, c='k', label='$\mathrm{SKA-like}$')
  plt.plot(v*1.e-3, psi_good_large*1.e3, c='c', label='$\mathrm{Model}$', lw=2)
  #plt.text(-200, 0.9*psi_plot.max()*1.e3, ('$\mathrm{SNR} = %.2f$' % line.snr))
  plt.xlabel('$\\times10^{3} \, v \, [\mathrm{kms}^{-1}]$')
  plt.ylabel('$\Psi \, [\mathrm{mJy}]$')
  plt.xlim([v[0]*1.e-3, v[-1]*1.e-3])
  plt.legend(frameon=False, fontsize='small')
  plt.savefig(plot_dir+'line_{0}_zoomout.png'.format(line_id), bbox_inches='tight', dpi=160)

  plt.figure(3, figsize=(4.5, 3.75))
  z_true = linestore['summary']['True']['z']
  v0 = linestore['chain']['v0']
  z_pdf = -v0/(v0+3e5)
  plt.hist((z_pdf-z_true)*1.e5, histtype='step', normed=True, color=cpal[-1])
  plt.axvline(0, linestyle='--', color=cpal[3])
  plt.ylabel('$\mathrm{Probability}\, P(z - z_{\\rm true})$')
  plt.xlabel('$10^{5}\\times\mathrm{Redshift \, Error} \, \Delta z$')
  plt.savefig(plot_dir+'line_{0}_zpdf.png'.format(line_id), bbox_inches='tight', dpi=160)

if __name__=='__main__':

  dat2 = Table.read('./output_snr_1_band_2/redshift_estimates.dat', format='ascii', delimiter=',', header_start=0)
  snr7 = dat2[(dat2['snr'] > 6)*(dat2['snr'] <8)]
  snr7 = snr7[(snr7['z_true'] > 0.0)*(snr7['z_true'] < 5)]

  #makePlot(snr7['id'][0], './output_snr_1_band_2/')
  makePlot('ID12600790', './output_snr_1_band_2/')