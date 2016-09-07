from __future__ import division, print_function
import pymultinest
import numpy as np
import pandas as pd
from radio_z import hiprofile
from collections import OrderedDict
import os, time
from multiprocessing import Pool
from functools import partial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def _fit_object(idx, cat, output_dir='output', n_live_points=500, convert_to_binary=False, store_in_hdf=True):
    """
    Given a idx, fits a single spectral line from a catalogue. External function to the FitCatalogue class to get
    around the pickling issues in the multiprocessing library.

    Parameters
    ----------
    idx : int
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
    print('Fitting object', idx)
    dat = cat[cat.index==idx]
    fd = FitData(dat['v'].as_matrix(), dat['psi'].as_matrix(), dat['psi_err'].as_matrix())
    fd.idx, n_live_points=n_live_points, chain_name=output_dir + '/' + idx + '-',
           convert_to_binary=convert_to_binary)
    if store_in_hdf:
        if 'results' not in cat.columns:
            cat['results'] = ''
        cat['results'][idx] = fd.results_df


class FitData:
    """
    Encapsulated class for fitting some HI profile data
    """
    def __init__(self, v, psi, sigma, bounds=[]):
        """
        Provide the data in the arguments

        Parameters
        ----------
        v : array
            Velocities
        psi : array
            Flux
        sigma : array or float
            Uncertainties in the flux
        bounds : OrderedDict, optional
            Uniform prior bounds on the parameters
        """
        self.v = v
        self.psi = psi
        self.sigma = sigma
        if len(bounds) == 0:
            # self.bounds = OrderedDict([
            #         ('v0', [v.min(),v.max()]),
            #         ('w_obs_20', [7, 20]),
            #         ('w_obs_50', [7, 20]),
            #         ('w_obs_peak', [7,14]),
            #         ('psi_obs_max', [-11, -4]),
            #         ('psi_obs_0', [-11, -4])
            #         ])
            # self.bounds = OrderedDict([
            #     ('v0', [v.min(), v.max()]),
            #     ('w_obs_20', [0, 1500]),
            #     ('w_obs_50', [0, 1500]),
            #     ('w_obs_peak', [0, 1500]),
            #     ('psi_obs_max', [0, 0.1]),
            #     ('psi_obs_0', [0, 0.1])
            # ])
            self.bounds = OrderedDict([
                    ('v0', [v.min(), v.max()]),
                    ('w_obs_20', [0, 1500]),
                    ('w_obs_50', [0, 1500]),
                    ('w_obs_peak', [0, 1500]),
                    ('psi_obs_max', [-11, -2]),
                    ('psi_obs_0', [-11, -2])
                ])
        else:
            self.bounds = bounds
        self.ndim = len(self.bounds)
        self.likecalls = 0

    def apply_bounds(self, params):
        """
        Additional prior to ensure constraints are met in the sampling.
        """
        return (params[1] > params[2]) and (params[2] > params[3]) and (params[4] > params[5])

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
        log_params = [4, 5]

        # This is the only obvious way to convert a ctypes pointer to a numpy array
        for i in range(nparams):
            if i in log_params:
                p = np.exp(cube[i])
            else:
                p = cube[i]
            params.append(p)
        # Now we check to make sure all conditions in eq.8-12 of Obreschkow are met
        if not self.apply_bounds(params):
            return -1e110  # Default multinest "zero" value is -1e100. We must return less than that.

        lp = hiprofile.LineProfile(*params)
        psi_fit = lp.get_line_profile(self.v, noise=0)
        # Multiply by dN/dz prior
        return np.sum(np.log(1/(np.sqrt(2*np.pi*self.sigma**2))))-0.5*np.sum(((psi_fit-self.psi)/self.sigma)**2)

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

    # def loglike_flat(self, cube, ndim, nparams):

    def fit(self, idx, n_live_points=500, chain_name='hi_run', store_in_hdf=True, convert_to_binary=False):
        """
        Actually run multinest to fit model to the data

        Parameters
        ----------
        n_live_points : int, optional
            Number of live points to use
        chain_name : str, optional
            Root for all the chains (including directory)
            Note: This path can't be too long because multinest has a hardcoded character limit (100 characters)
        convert_to_binary : bool, optional
            If true, will convert the large chain files to numpy binary files
        """
        t1 = time.time()
        pymultinest.run(self.loglike, self.prior, self.ndim, importance_nested_sampling = True, init_MPI = False,
                        resume = False, verbose = False, sampling_efficiency = 'parameter',
                        n_live_points = n_live_points, outputfiles_basename = chain_name, multimodal=True)

        if convert_to_binary:
            # These are the files we can convert
            ext = ['ev.dat', 'phys_live.points', 'live.points', '.txt', 'post_equal_weights.dat']
            for e in ext:
                infile = os.path.join(chain_name+e)
                outfile = infile+'.npy'
                x = np.loadtxt(infile)
                np.save(outfile, x)
                os.system('rm %s' % infile)

        if store_in_hdf:
            # These are the files we can convert
            ext = ['ev.dat', 'phys_live.points', 'live.points', '.txt', 'post_equal_weights.dat']
            self.results_df = pd.DataFrame(index=idx)
            for e in ext:
                infile = os.path.join(chain_name+e)
                x = np.loadtxt(infile)
                self.results_df[e] = ''
                self.results_df[e][idx] = x

        print('Time taken', (time.time()-t1)/60, 'minutes')


class FitCatalogue:
    """
    Fit an entire catalogue of data
    """
    def __init__(self, cat):
        """
        Class to fit a catalogue of data, in parallel if requested
        Parameters
        ----------
        cat : pandas.DataFrame
            Catalogue of data in an HDF5 format where each object is a new row
        """
        self.cat = cat

    def fit_all(self, nprocesses=1, output_dir='output', n_live_points=500, store_in_hdf=True, convert_to_binary=False, subset=[]):
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

        if len(subset) == 0:
            idx = self.cat.index
        else:
            idx = subset

        if nprocesses > 1:
            new_func = partial(_fit_object, idx, cat=dict(self.cat), output_dir=output_dir, n_live_points=n_live_points,
                               convert_to_binary=convert_to_binary, store_in_hdf=store_in_hdf)
            p = Pool(nprocesses)
            p.map(new_func, idx)

        else:
            for i in idx:
                _fit_object(i, self.cat, output_dir = output_dir, n_live_points = n_live_points,
                            convert_to_binary = convert_to_binary, store_in_hdf=store_in_hdf)


class ChainAnalyser:
    """
    Class with convenience functions to analyse multinest output.
    """
    def __init__(self, chain_name, idx, data_source, log_params=[4,5]):
        """
        Multinest chain analysis class.

        Parameters
        ----------
        chain_name : str
            The full root of all the chains (e.g. '/my/multinest/chain-') such that <chain_name>.stats and other
             output files exist
        idx : int
            The catalogue index of the chain to be analysed
        data_source : str
            The location of the stored data we want to analyse.
            If 'binary' stored in a numpy binary .npy
            If 'frame' stored in a pandas DataFrame
        log_params : list, optional
            Which parameters were varied in log space and so should be exponentiated
        """
        self.chain_name = chain_name
        self.log_params = log_params
        if data_source=='binary':
            post = self.chain_name + 'post_equal_weights.dat'
            if os.path.exists(post + '.npy'):
                self.chain = np.load(post + '.npy')
            else:
                self.chain = np.loadtxt(post)
        elif data_source=='frame':
            store = pd.HDFStore(chain_name)
            cat = store['table']
            chain_series = cat['results'][cat.index==idx]['post_equal_weights.dat']
            self.chain = chain_series[idx].values

        if len(self.log_params) > 0:
            self.chain[:,self.log_params] = np.exp(self.chain[:,self.log_params])

        self.param_names = ['v0', 'w_obs_20', 'w_obs_50', 'w_obs_peak', 'psi_obs_max', 'psi_obs_0', 'z']

    def convert_z(self, v):
        c = 3e5
        return -(v/(v+c))

    def p_of_z(self, delta_z=0, v0_ind=0):
        """
        Function to return the marginalised probability density function of redshift for a given object.

        Parameters
        ----------
        delta_z : float, optional
            Approximate desired width of bin
        v0_ind : int, optional
            The column of the chain containing the v0 values

        Returns
        -------
        bins : array
            The mid points of the z bins
        pdf : array
            The values of the pdf at the corresponding z value

        """
        c = 3e5 #Speed of light in km/s

        z = self.convert_z(self.chain[:, v0_ind])

        if delta_z == 0:
            nbins = 25
        else:
            nbins = (int)((z.max() - z.min())/delta_z)
        pdf, bins = np.histogram(z, bins=nbins, density=True)

        # We want to return the mid points of the bins
        new_bins = (bins[1:] + bins[:-1])/2

        return new_bins, pdf

    def plot_p_of_z(self, delta_z=0, v0_ind=0, smooth=False):
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
            plt.plot(newbins, newpdf, color='#0057f6', lw=1.5)
        else:
            plt.plot(bins, pdf, color='k')

        plt.xlabel('z')
        plt.ylabel('P(z)')
        plt.tight_layout()

    def parameter_estimates(self, true_params=[]):

        z = self.convert_z(self.chain[:, 0])
        logpost = self.chain[:, -1]
        chain = np.column_stack((self.chain[:, :-1], z))

        parameters = pd.DataFrame(columns = ['Mean', 'Median', 'MAP', '16p', '84p'], index=self.param_names)

        parameters['Mean'] = np.mean(chain, axis=0)
        parameters['Median'] = np.median(chain, axis=0)
        parameters['MAP'] = chain[np.argmax(logpost), :]
        parameters['16p'] = np.percentile(chain, 16, axis=0)
        parameters['84p'] = np.percentile(chain, 84, axis=0)

        if len(true_params) != 0:
            true_z = self.convert_z(true_params[0])
            true_params = np.append(true_params, true_z)
            parameters['True'] = true_params

        return parameters

    def triangle_plot(self, params=[], labels=[], true_vals=[], best_params=[], smooth=5e3, rot=0):
        """
            Plots the triangle plot for a sampled chain.
            chain = Input chain
            params = List of indices of parameters, otherwise every column of chain is used
            labels = Labels for parameters
            true_vales = If provided, plots the true values on the histograms and contours
            best_params = List of lists for each parameter (mean, minus uncertainty, plus uncertainty) plotted on histograms
            smooth = Smoothing scale for the contours. Contour will raise warning is this is too small. Set to 0 for no smoothing.
        """
        contour_plot.triangle_plot(self.chain.copy(), params=params, labels=labels, true_vals=true_vals,
                                   best_params=best_params, smooth=smooth, rot=rot)




















