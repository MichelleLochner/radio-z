from __future__ import division, print_function
import pymultinest
import numpy as np
import pandas as pd
from radio_z import hiprofile, contour_plot
from collections import OrderedDict
import os
import time
import sys
import glob
from multiprocessing import Pool
from functools import partial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tables.exceptions import HDF5ExtError  # Needed to catch errors when loading hdf5 files


def _fit_object(filename, output_dir='output', save_to_hdf=True, delete_files=False, n_live_points=500, log_file=''):
    """
    Given a key, fits a single spectral line from a catalogue. External function to the FitCatalogue class to get
    around the pickling issues in the multiprocessing library.

    Parameters
    ----------
    key : str
        ID of object
    cat : pandas.DataFrame or dict
        Contains the catalogue of all the objects
    output_dir : str, optional
        Output directory. Chains will automatically be named using the ID.
    n_live_points : int, optional
        Number of live points for multinest
    convert_to_binary : bool, optional
        If true, converts the multinest output files to binary numpy files to save space.
    """
    id = filename.split(os.sep)[-1].split('.')[0]
    print('Fitting object', id)
    fd = FitData(filename=filename)
    t1 = time.time()
    fd.fit(chain_name=output_dir + '/' + id + '-', save_to_hdf=save_to_hdf, delete_files=delete_files,
           n_live_points=n_live_points)
    # time.sleep(np.random.randn()*2+5)
    tm = time.time() - t1

    if len(log_file)!= 0:
        prms = pd.read_hdf(filename, 'parameters')
        fl = open(log_file, 'a')
        fl.write('%s\t%2.2f\t%2.2f\t%3.2f\n' %(id, prms.snr_band1_santos, prms.snr_band2_santos, tm/60.))
        fl.close()


class FitData:
    """
    Encapsulated class for fitting some HI profile data
    """
    def __init__(self, read_from_hdf=True, filename='chain.hdf5', v=[], psi=[], sigma=[], bounds=[]):
        """
        Class for using Multinest for inference on a single galaxy. Either read the data from an object HDF5 file (in
        the 'data' table) or provide the data directly in the arguments. Can also save the output chain directly to
        the same HDF5 file.

        Parameters
        ----------
        read_from_hdf : boolean, optional
            If true, read the data directly from an individual object's HDF5 file
        filename : str, optional
            The HDF5 file to read the data from and/or write the output chain to
        v : array, optional
            Velocities (use if read_from_hdf = False)
        psi : array, optional
            Flux (use if read_from_hdf = False)
        sigma : array or float, optional
            Uncertainties in the flux (use if read_from_hdf = False)
        bounds : OrderedDict, optional
            Uniform prior bounds on the parameters
        """
        if not os.path.exists(filename):
            print(filename, "does not exist")
            raise IOError

        self.filename = filename

        self.complib = 'bzip2'  # What compression library should be used when storing hdf5 files

        if read_from_hdf:
            try:
                hstore = pd.HDFStore(self.filename)
                # We'll assume the data is stored in a child table in the hdf5 file
                data = hstore['data']
                self.v, self.psi, self.sigma = data.as_matrix().T

            except HDF5ExtError:
                if len(v) == 0:
                    print('Error: File provided is not an HDF5 file or is corrupt. Please provide v, psi and sigma '
                          'instead.')
                    sys.exit(0)
                else:
                    print('Warning: File provided is not an HDF5 file or is corrupt')

                self.v = v
                self.psi = psi
                self.sigma = sigma
        else:
            self.v = v
            self.psi = psi
            self.sigma = sigma

        if len(bounds) == 0:
            vmax = self.v.max()
            if vmax > 0:
                vmax = 0 # Redshift can't be less than zero
            self.bounds = OrderedDict([
                    ('v0', [self.v.min(), vmax]),
                    ('w_obs_20', [-1, 7.5]),
                    ('w_obs_50', [-1, 7.5]),
                    ('w_obs_peak', [-1, 7.5]),
                    ('psi_obs_max', [-11, -2]),
                    ('psi_obs_0', [-11, -2])
                ])
        else:
            self.bounds = bounds

        self.log_params = np.arange(1, 6)

        self.ndim = len(self.bounds)
        self.likecalls = 0

    def apply_bounds(self, params):
        """
        Additional prior to ensure constraints are met in the sampling.
        """
        return (params[1] > params[2]) and (params[2] > params[3]) and (params[4] > params[5]) and (params[3]>0)

    def loglike(self, cube, ndim, nparams):
        """
        Log likelihood for multinest

        Parameters
        ----------
        cube : array-like
            Input parameter cube
        ndim : int
            Number of dimensions
        nparams : int
            Number of parameters actually varying

        Returns
        -------
        float
            Log likelihood
        """
        self.likecalls += 1
        params = []


        # This is the only obvious way to convert a ctypes pointer to a numpy array
        for i in range(nparams):
            if i in self.log_params:
                p = np.exp(cube[i])
            else:
                p = cube[i]
            params.append(p)
        # Now we check to make sure all conditions in eq.8-12 of Obreschkow are met
        if not self.apply_bounds(params):
            return -1e110  # Default multinest "zero" value is -1e100. We must return less than that.

        lp = hiprofile.LineProfile(*params)
        psi_fit = lp.get_line_profile(self.v, noise=0)
        return -0.5*np.sum(((psi_fit-self.psi)/self.sigma)**2)

    def prior(self, cube, ndim, nparams):
        """
        Prior for multinest

        Parameters
        ----------
        cube : array-like
            Input parameter cube
        ndim : int
            Number of dimensions
        nparams : int
            Number of parameters actually varying

        Returns
        -------
        array
            Transformed parameter cube

        """
        keys = list(self.bounds.keys())
        for i in range(ndim):
            lower = self.bounds[keys[i]][0]
            upper = self.bounds[keys[i]][1]
            cube[i] = cube[i]*(upper-lower)+lower
        return cube



    def fit(self, chain_name='hi_run', save_to_hdf=True, delete_files=False, n_live_points=500, multimodal=True):
        """
        Actually run multinest to fit model to the data

        Parameters
        ----------
        n_live_points : int, optional
            Number of live points to use
        chain_name : str, optional
            Root for all the chains (including directory)
            Note: This path can't be too long because multinest has a hardcoded character limit (100 characters)
        save_to_hdf : boolean, optional
            Whether or not to store the chain (only the equal weighted posterior) and the evidence in the object hdf5
            file (provided at initialisation)
        delete_files : boolean, optional
            Whether or not to delete the base chain files (will not exectue if not saved to hdf5 first)
        multimodal : boolean, optional
            Whether or not to run multinest in multimodal mode. If true, can occasionally fix modes too early so it's
            worth changing for difficult problems.
        """
        t1 = time.time()
        pymultinest.run(self.loglike, self.prior, self.ndim, importance_nested_sampling = True, init_MPI = False,
                        resume = False, verbose = False, sampling_efficiency = 'model', evidence_tolerance = 0.5,
                        n_live_points = n_live_points, outputfiles_basename = chain_name, multimodal = multimodal)

        if save_to_hdf:
            # These are the files we can convert
            x = np.loadtxt(chain_name+'post_equal_weights.dat')
            df = pd.DataFrame(data=x, columns=list(self.bounds.keys())+['loglike'])
            df.to_hdf(self.filename, 'chain', complib=self.complib)

            ev, ev_sig, ev_is = self.read_evidence(chain_name)
            bayes_fact, bayes_sig = self.compute_evidence_ratio(chain_name)
            df_ev = pd.DataFrame(data=np.array([[ev, ev_sig, ev_is, bayes_fact]]), columns=['ln(evidence)',
                                                                                           'uncertainty',
                                                                                            'IS ln(evidence)',
                                                                                     'Bayes factor'])
            df_ev.to_hdf(self.filename, 'evidence', complib=self.complib)

            if delete_files:
                fls = glob.glob(chain_name+'*')
                print('Deleting files')
                for f in fls:
                    os.system('rm '+f)

        print('Time taken', (time.time()-t1)/60, 'minutes')

    def compute_null_evidence(self):
        """
        Computes the Bayesian evidence for the "null hypothesis" (i.e. y=0)

        Returns
        -------
        float
            Bayesian evidence
        """
        return -0.5*np.sum((self.psi/self.sigma)**2)

    def read_evidence(self, chain_name):
        """
        Reads in the ln(evidence) and uncertainty for a run multinest chain.

        Parameters
        ----------
        chain_name : str
            The name of an already run chain where the evidence is stored

        Returns
        -------
        float
            ln(evidence)
        float
            Uncertainty in ln(evidence)
        """
        lns = open(chain_name+'stats.dat').readlines()
        line = lns[0].split(':')[1].split()
        ev = float(line[0])
        ev_sig = float(line[-1])
        line = lns[1].split(':')[1].split()  # Get the importance sampled evidence
        ev_is = float(line[0])
        return ev, ev_sig, ev_is

    def compute_evidence_ratio(self, chain_name):
        """
        Computes the Bayesian evidence ratio of the fitted model (M2) to the "null hypothesis" (M1)

        Parameters
        ----------
        chain_name : str
            The name of an already run chain where the evidence is stored

        Returns
        -------
        float
            ln(E2/E1)
        float
            Uncertainty in ln(E2/E1)
        """

        E2, E2_sig, E_is = self.read_evidence(chain_name)

        E1 = self.compute_null_evidence()
        return E2-E1, E2_sig


class FitCatalogue:
    """
    Fit an entire catalogue of data
    """
    def __init__(self, filepath='./', log_name='log', append_to_log=False, subset=[]):
        """
        Class to fit a catalogue of data, in parallel if requested. Assumes data are stored as individual HDF5 files
        in a single directory.
        Parameters
        ----------
        filepath : str, optional
            Catalogue of data where each object is a different HDF5 file
        """
        self.filepath = filepath
        self.subset = subset
        if len(log_name) != 0:
            self.log_file = os.path.join(filepath, log_name+'.txt')
            if not append_to_log:
                fl = open(self.log_file, 'w')
                fl.write('%s\tSNR1\tSNR2\tTime\n' %('#ID'.ljust(11)))
                fl.close()
        else:
            self.log_file = ''

    def fit_all(self, nprocesses=1, output_dir='output', save_to_hdf=True, delete_files=False, n_live_points=500):
        """
        Fits all the spectral lines in a catalogue.

        Parameters
        ----------
        nprocesses : int, optional
            Number of processors to be used (note: parallelisation only available with shared memory)
        output_dir : str, optional
            Output directory. Chains will automatically be named using the ID.
        n_live_points : int, optional
            Number of live points for multinest
        convert_to_binary : bool, optional
            If true, converts the multinest output files to binary numpy files to save space.
        subset : list, optional
            Give a list of keys to run on a subset of the data
        """

        if len(self.subset) == 0:
            files = glob.glob(os.path.join(self.filepath, 'ID*hdf5'))
        else:
            files = self.subset


        if nprocesses > 1:
            new_func = partial(_fit_object, output_dir=output_dir, save_to_hdf=save_to_hdf, delete_files=delete_files,
                               n_live_points=n_live_points, log_file=self.log_file)
            p = Pool(nprocesses)
            p.map(new_func, files)

        else:
            for f in files:
                _fit_object(f, output_dir=output_dir, save_to_hdf=save_to_hdf, delete_files=delete_files,
                            n_live_points=n_live_points, log_file=self.log_file)


class ChainAnalyser:
    """
    Class with convenience functions to analyse multinest output.
    """
    def __init__(self, filename, log_params=[4,5]):
        """
        Multinest chain analysis class.

        Parameters
        ----------
        filename : str, optional
            The HDF5 file to read the chain and evidence from
        log_params : list, optional
            Which parameters were varied in log space and so should be exponentiated
        """
        self.filename = filename
        self.log_params = log_params

        try:
            self.chain = pd.read_hdf(filename, 'chain').as_matrix()
            self.evidence = pd.read_hdf(filename, 'evidence')
        except KeyError:
            print('Chain not found in file', filename)
            raise KeyError

        self.param_names = ['v0', 'w_obs_20', 'w_obs_50', 'w_obs_peak', 'psi_obs_max', 'psi_obs_0', 'z']

    def convert_z(self, v):
        c = 3e5
        return -(v/(c+v))

    def p_of_z(self, delta_z=0, z_min=0, z_max=0, nbins=50, v0_ind=0, save_to_file=True):
        """
        Function to return the marginalised probability density function of redshift for a given object.

        Parameters
        ----------
        delta_z : float, optional
            Approximate desired width of bin
        v0_ind : int, optional
            The column of the chain containing the v0 values
        save_to_file : bool, optional
            Whether or not to store the output back in the original hdf5 file

        Returns
        -------
        bins : array
            The mid points of the z bins
        pdf : array
            The values of the pdf at the corresponding z value

        """
        c = 3e5 # Speed of light in km/s

        z = self.convert_z(self.chain[:, v0_ind])

        if z_max==0:
            if delta_z == 0:
                nbins = 25
            else:
                nbins = (int)((z.max() - z.min())/delta_z)
            bins = np.linspace(z.min(), z.max(), nbins)
        else:
            bins = np.linspace(z_min, z_max, nbins)
        pdf, bins = np.histogram(z, bins=bins)
        pdf = pdf/np.sum(pdf) # Normalise

        # We want to return the mid points of the bins
        new_bins = (bins[1:] + bins[:-1])/2

        if save_to_file:
            df = pd.DataFrame(data=np.column_stack((new_bins, pdf)), columns=['z', 'p(z)'])
            df.to_hdf(self.filename, 'p(z)')

        return new_bins, pdf

    def plot_p_of_z(self, delta_z=0, v0_ind=0, true_val=0, colour='#0057f6', smooth=False, rot=0):
        """
        Plots P(z)
        Parameters
        ----------
        delta_z : float, optional
            Approximate desired width of bin
        v0_ind : int, optional
            The column of the chain containing the v0 values
        smooth : bool, optional
            Whether or not to smooth the resulting curve
        """

        bins, pdf = self.p_of_z(delta_z=delta_z, v0_ind=v0_ind)
        if smooth:
            f = interp1d(bins, pdf, kind='cubic')
            newbins = np.linspace(bins.min(), bins.max(), 100)
            newpdf = f(newbins)
            plt.plot(newbins, newpdf, color=colour, lw=1.5)
        else:
            plt.plot(bins, pdf, color=colour, lw=1.5)

        plt.xticks(rotation=rot)
        plt.xlabel('z')
        plt.ylabel('P(z)')


        if true_val != 0:
            plt.plot([true_val, true_val], plt.gca().get_ylim(), lw=1.5, color='k')
        plt.tight_layout()



    def get_errors(self, x, max_post, perc):
        """
        Returns the error estimates from a chain

        Parameters
        ----------
        x : numpy array
            Column of a chain
        max_post : float
            Maximum posterior point in x
        perc : float
            Which percentile to compute (68 for 1 sigma, 95 for 2 sigma, 99 for 3 sigma)

        Returns
        -------
        sig1, sig2
            The values of x corresponding the input percentile

        """
        xnew = np.sort(x)
        x1 = xnew[xnew < max_post]
        x2 = xnew[xnew >= max_post]

        if len(x1) == 0:
            sig1 = max_post
        else:
            sig1 = np.percentile(x1, 100-perc)
        if len(x2) == 0:
            sig2 = max_post
        else:
            sig2 = np.percentile(x2, perc)

        return sig1, sig2

    def parameter_estimates(self, true_params=[], save_to_file=True):
        """
        Returns the best fit estimate of the parameters and their uncertainties.

        Parameters
        ----------
        true_params : list-like, optional
            If the true parameters are supplied, add them to the output dataframe for ease of comparison
        save_to_file : bool, optional
            Whether or not to store the output back in the original hdf5 file

        Returns
        -------
        pd.DataFrame
            The parameter estimates (mean, median and maximum posterior) as well as the 16th and 84th percentiles
            (corresponding to upper and lower 1 sigma estimates for a Gaussian)

        """

        z = self.convert_z(self.chain[:, 0])
        logpost = self.chain[:, -1]
        chain = np.column_stack((self.chain[:, :-1], z))

        parameters = pd.DataFrame(columns = ['Mean', 'Median', 'MAP', 'lower_1sigma', 'upper_1sigma','lower_2sigma',
                                             'upper_2sigma','lower_3sigma', 'upper_3sigma'], index=self.param_names)

        parameters['Mean'] = np.mean(chain, axis=0)
        parameters['Median'] = np.median(chain, axis=0)
        maps = chain[np.argmax(logpost), :]
        parameters['MAP'] = maps

        lower1 = np.zeros(len(chain[0,:]))
        upper1 = np.zeros(len(chain[0,:]))
        lower2 = np.zeros(len(chain[0, :]))
        upper2 = np.zeros(len(chain[0, :]))
        lower3 = np.zeros(len(chain[0, :]))
        upper3 = np.zeros(len(chain[0, :]))

        for i in range(len(self.param_names)):
            lower1[i], upper1[i] = self.get_errors(chain[:,i], maps[i], 68)
            lower2[i], upper2[i] = self.get_errors(chain[:, i], maps[i], 95)
            lower3[i], upper3[i] = self.get_errors(chain[:, i], maps[i], 99.7)

        parameters['lower_1sigma'] = lower1
        parameters['upper_1sigma'] = upper1
        parameters['lower_2sigma'] = lower2
        parameters['upper_2sigma'] = upper2
        parameters['lower_3sigma'] = lower3
        parameters['upper_3sigma'] = upper3

        parameters.iloc[self.log_params, :] = np.exp(parameters.iloc[self.log_params, :])

        if len(true_params) != 0:
            true_z = self.convert_z(true_params[0])
            true_params = np.append(true_params, true_z)
            parameters['True'] = true_params

        if save_to_file:
            parameters.to_hdf(self.filename, 'summary')

        return parameters

    def triangle_plot(self, params=[], labels=[], true_vals=[], best_params=[], smooth=5e3, rot=0):
        """
        Plots the triangle plot for a sampled chain.

        Parameters
        ----------
        params : list-like, optional
            List of indices of parameters, otherwise every column of chain is used
        labels : list-like, optional
            Labels for parameters
        true_vals : list-like, optional
            If provided, plots the true values on the histograms and contours
        best_params : list-like, optional
            List of lists for each parameter (mean, minus uncertainty, plus uncertainty) plotted on histograms
        smooth : float, optional
            Smoothing scale for the contours. Contour will raise warning is this is too small. Set to 0 for no smoothing.
        rot : float, optional
            Rotation angle for the x axis tick labels (they often clash and need to be rotated)

        Returns
        -------

        """
        contour_plot.triangle_plot(self.chain.copy(), params=params, labels=labels, true_vals=true_vals,
                                   best_params=best_params, smooth=smooth, rot=rot)






















