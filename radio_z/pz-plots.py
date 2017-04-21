import numpy as np
import pandas as pd
from astropy.table import Table
from matplotlib import pyplot as plt
from matplotlib import rc
import pdb

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

cpal = ['#185aa9','#008c48','#ee2e2f','#f47d23','#662c91','#a21d21','#b43894','#010202']

plt.close('all') # tidy up any unshown plots

band_list = ['1', '2']
B_arr = np.linspace(-1,10,128)

area_input = 16.
area_output = 5000.

# from input catalogue
Ngal_inband1 = 524600
Ngal_inband2 = 129557

ngal_cont_band1 = 2.25
ngal_cont_band2 = 0.95

bcut = 0.0
snrcut = 5

for band in band_list:
  plt.close('all')
  #raw_dat = Table.read('./output_snr_1_band_{0}/redshift_estimates.dat'.format(band), format='ascii', delimiter=',', header_start=0)
  #raw_dat = Table.read('./output_snr_1_band_{0}/updated_redshift_estimates_band{0}.dat'.format(band), format='ascii', delimiter=',', header_start=0)
  raw_dat = Table.read('./output_snr_1_band_{0}/updated_redshift_estimates_band{0}_sigmas.dat'.format(band), format='ascii', delimiter=',', header_start=0)

  if band=='1':
    Ngal = Ngal_inband1
    ngal_cont = ngal_cont_band1
  elif band=='2':
    Ngal = Ngal_inband2
    ngal_cont = ngal_cont_band2

  dat = raw_dat[raw_dat['B'] > bcut]
  snr_dat = raw_dat[raw_dat['snr_band{0}_santos'.format(band)] > snrcut]
  snr_dat_10 = raw_dat[raw_dat['snr_band{0}_santos'.format(band)] > 2.*snrcut]
  f_snr = float(snr_dat['snr_band{0}_santos'.format(band)].shape[0])/Ngal
  n5k_snr = ngal_cont*60.*60.*area_output*f_snr

  fname = 'Band {0}'.format(band)
  z_true = dat['z_true']
  z_map = dat['z_est']
  z_l3 = dat['z_err_lower3']
  z_u3 = dat['z_err_upper3']

  eta_h = abs(z_true - z_map) / (1.e0 + z_true) > 0.15
  eta_h = float(np.sum(eta_h)) / eta_h.shape[0]
  
  eta_b = abs(np.log((1.e0 + z_map)/(1.e0 + z_true)))
  eta_b = float(np.sum(eta_b)) / eta_b.shape[0]
  
  eta_sigma = (z_true > (z_map-z_l3))*(z_true < (z_map+z_u3))
  eta_sigma = float(np.sum(eta_sigma)) / eta_sigma.shape[0]

  f_tot = float(z_true.shape[0]) / Ngal
  n5k_b = ngal_cont*60.*60.*area_output*f_tot

  x = np.linspace(z_true.min(), z_true.max(), 10)
  tx = z_true.min() + 0.6*(z_true.max() - z_true.min())
  ty = z_true.min() + 0.1*(z_true.max() - z_true.min())

  plt.figure(1, figsize=(5, 3.75))
  plt.hexbin(z_true, z_map, cmap='gnuplot2_r', bins='log', gridsize=50)
  cbar = plt.colorbar(ticks=[0,1,2,3,4])
  cbar.set_label('$N_{\\rm gal}$')
  cbar.ax.set_yticklabels(['$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{3}$', '$10^{4}$'])
  plt.xlabel('$z_{\\rm true}$')
  plt.ylabel('$z_{\\rm est}$')
  plt.xlim([z_true.min(), z_true.max()])
  plt.ylim([z_map.min(), z_map.max()])
  plt.text(tx, ty, '$\eta_{\mathrm{H}} = \, $'+('%.3f' % eta_h)+'\n$\eta_{\mathrm{B}} = \, $'+('%.3f' % eta_b)+'\n$\eta_{3\sigma} = \, $'+('%.3f' % (1-eta_sigma))+'\n$f_{\\rm tot} = \, $'+('%.2f' % (f_tot*100))+'$\%$'+'\n$N_{5\\rm{k}} = \, $'+('%.2e' % n5k_b), color='black')
  plt.title('$\mathrm{Band \, '+band+'}$')
  plt.savefig(fname+'.png', bbox_inches='tight', dpi=300)

  plt.figure(2, figsize=(5, 3.75))
  plt.hexbin(snr_dat['z_true'], snr_dat['z_true'], cmap='gnuplot2_r', bins='log', gridsize=50)
  cbar = plt.colorbar(ticks=[0,1,2,3,4])
  cbar.set_label('$N_{\\rm gal}$')
  cbar.ax.set_yticklabels(['$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{3}$', '$10^{4}$'])
  plt.xlabel('$z_{\\rm true}$')
  plt.ylabel('$z_{\\rm est}$')
  plt.xlim([z_true.min(), z_true.max()])
  plt.ylim([z_map.min(), z_map.max()])
  plt.text(tx, ty, '$\mathrm{SNR}_{\\rm vel} > \,$'+('%d' % snrcut)+'\n$f_{\\rm tot} = \, $'+('%.2f' % (f_snr*100))+'$\%$'+'\n$N_{5\mathrm{k}} = \, $'+('%.2e' % n5k_snr), color='black')
  plt.title('$\mathrm{Band \, '+band+'}$')
  plt.savefig(fname+'_snrcut.png', bbox_inches='tight', dpi=300)

  plt.figure(3, figsize=(5, 3.75))
  plt.yscale('log')
  plt.hist(snr_dat['z_true'], histtype='step', color='k', label='$\mathrm{SNR}_{\\rm vel} > \,$'+('%d' % snrcut))
  plt.hist(snr_dat_10['z_true'], linestyle='-.', histtype='step', color='k', label='$\mathrm{SNR}_{\\rm vel} > \,$'+('%d' % (2*snrcut)))
  plt.hist(z_map, histtype='step', color='c', label='$\mathrm{ln}(B) > \,$'+('%.1f' % bcut)+'$\, \mathrm{estimated}$')
  plt.hist(z_true, linestyle='--', histtype='step', color='m', label='$\mathrm{ln}(B) > \,$'+('%.1f' % bcut)+'$\, \mathrm{true}$')
  if band == '2':
    plt.ylim([4.e1, 8.e3])
  plt.xlabel('$\mathrm{Redshift} \, \, z$')
  plt.ylabel('$N_{\\rm gal}$')
  plt.legend(frameon=False, loc='upper right')
  plt.title('$\mathrm{Band \, '+band+'}$')
  plt.savefig(fname+'_zhist.png', bbox_inches='tight', dpi=300)

  plt.figure(4, figsize=(5, 3.75))
  deltaz = z_map-z_true
  deltazerr = np.column_stack([dat['z_err_lower'], dat['z_err_upper']])
  plt.errorbar(z_true, deltaz, yerr=deltazerr.T, fmt='o', alpha=0.6)
  plt.xlabel('$z_{\\rm true}$')
  plt.ylabel('$\Delta z$')
  plt.title('$\mathrm{Band \, '+band+'}$')
  plt.savefig(fname+'_zerr.png', bbox_inches='tight', dpi=300)

  plt.figure(5, figsize=(5, 3.75))
  deltaz = z_map-z_true
  plt.yscale('log')
  plt.hist(deltaz, histtype='step')
  plt.xlabel('$\Delta z$')
  plt.title('$\mathrm{Band \, '+band+'}$')
  plt.savefig(fname+'_zerr_hist.png', bbox_inches='tight', dpi=300)

  plt.figure(6, figsize=(5, 3.75))
  plt.scatter(np.exp(raw_dat['snr_band{0}_santos'.format(band)]), raw_dat['B'], c=raw_dat['z_true'], cmap='plasma_r', linewidths=0, alpha=0.6)
  plt.axhline(np.exp(0), linestyle='--')
  plt.axvline(5, linestyle='--')
  plt.yscale('log')
  plt.xscale('log')
  plt.ylim([1.e-4, 1.e7])
  plt.xlim([2.e0, 1.e6])
  cbar = plt.colorbar()
  cbar.set_label('$z_{\\rm true}$')
  plt.xlabel('$\mathrm{SNR_{vel}}$')
  plt.ylabel('$B$')
  plt.title('$\mathrm{Band \, '+band+'}$')
  plt.savefig(fname+'_snr_B.png', bbox_inches='tight', dpi=300)

  plt.figure(7, figsize=(5, 3.75))
  plt.scatter(np.exp(raw_dat['snr_band{0}_std'.format(band)]), raw_dat['B'], c=raw_dat['z_true'], cmap='plasma_r', linewidths=0, alpha=0.6)
  plt.axhline(np.exp(0), linestyle='--')
  plt.axvline(5, linestyle='--')
  plt.yscale('log')
  plt.xscale('log')
  plt.ylim([1.e-4, 1.e7])
  plt.xlim([2.e0, 1.e6])
  cbar = plt.colorbar()
  cbar.set_label('$z_{\\rm true}$')
  plt.xlabel('$\mathrm{SNR_{peak}}$')
  plt.ylabel('$B$')
  plt.title('$\mathrm{Band \, '+band+'}$')
  plt.savefig(fname+'_snr_std_B.png', bbox_inches='tight', dpi=300)

  plt.figure(8, figsize=(5, 3.75))
  plt.scatter(np.exp(raw_dat['snr_band{0}_santos'.format(band)]), np.exp(raw_dat['snr_band{0}_std'.format(band)]), c=raw_dat['z_true'], cmap='plasma_r', linewidths=0, alpha=0.6)
  plt.axhline(5, linestyle='--')
  plt.axvline(5, linestyle='--')
  plt.yscale('log')
  plt.xscale('log')
  plt.xlim([1.e0, 1.e6])
  plt.ylim([1.e0, 1.e6])
  cbar = plt.colorbar()
  cbar.set_label('$z_{\\rm true}$')
  plt.xlabel('$\mathrm{SNR_{vel}}$')
  plt.ylabel('$\mathrm{SNR_{peak}}$')
  plt.title('$\mathrm{Band \, '+band+'}$')
  plt.savefig(fname+'_snr_std_santos.png', bbox_inches='tight', dpi=300)

  eta_h_arr = np.zeros_like(B_arr)
  eta_b_arr = np.zeros_like(B_arr)
  eta_sigma_arr = np.zeros_like(B_arr)
  f_tot_arr = np.zeros_like(B_arr)

  for i,B in enumerate(B_arr):
    dat = raw_dat[raw_dat['B'] > B]
    z_true = dat['z_true']
    z_map = dat['z_est']
    z_l3 = dat['z_err_lower3']
    z_u3 = dat['z_err_upper3']

    eta_h = abs(z_true - z_map) / (1.e0 + z_true) > 0.15
    eta_h_arr[i] = float(np.sum(eta_h)) / eta_h.shape[0]

    eta_b = abs(np.log((1.e0 + z_map)/(1.e0 + z_true)))
    eta_b_arr[i] = float(np.sum(eta_b)) / eta_b.shape[0]

    eta_sigma = (z_true > (z_map-z_l3))*(z_true < (z_map+z_u3))
    eta_sigma_arr[i] = float(np.sum(eta_sigma)) / eta_sigma.shape[0]
    
    f_tot_arr[i] = float(z_true.shape[0]) / Ngal

  fig2, ax1 = plt.subplots(figsize=(4.5, 3.75))
  ax1.plot(B_arr, 1-eta_h_arr, '-', color='c', lw=2, label='$\eta_{\\rm H}$')
  ax1.plot(B_arr, 1-eta_b_arr, '--', color='m', lw=2, label='$\eta_{\\rm B}$')
  ax1.plot(B_arr, eta_sigma_arr, '-.', color='y', lw=2, label='$\eta_{\sigma}$')
  ax1.set_ylim([(eta_sigma_arr).min(), 1.e0])
  ax1.set_ylabel('$1 - \mathrm{Outlier \, Fraction} \, \, \eta$')
  ax1.set_xlabel('$\mathrm{Bayes \, Factor \, Cut} \, \, \mathrm{ln}(B)$')
  ax1.legend(frameon=False, loc='upper right')
  ax2 = ax1.twinx()
  ax2.plot(B_arr, f_tot_arr*100, 'k-', lw=2)
  ax2.set_ylabel('$\mathrm{Detected \, Fraction} \, f_{\\rm tot} \, \%$')
  ax2.set_xlim([B_arr.min(), B_arr.max()])
  ax2.set_xlabel('$\mathrm{Bayes \, Factor \, Cut} \, \mathrm{ln}(B)$')
  ax2.set_title('$\mathrm{Band \, '+band+'}$')
  plt.savefig(fname+'_meas.png', bbox_inches='tight', dpi=300)


