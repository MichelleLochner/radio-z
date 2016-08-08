from __future__ import division, print_function
import pymultinest
import numpy as np
import pandas as pd
from radio_z import hiprofile
from collections import OrderedDict
import os, time
from multiprocessing import Pool
from functools import partial

def _fit_object(key, cat, output_dir='output', n_live_points=500, convert_to_binary=True):
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
    print('Fitting object', key[1:])
    dat = cat[key]
    fd = FitData(dat['v'].as_matrix(), dat['psi'].as_matrix(), dat['psi_err'].as_matrix())
    fd.fit(n_live_points=n_live_points, chain_name=output_dir + '/' + key[1:] + '-',
           convert_to_binary=convert_to_binary)

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
            self.bounds = OrderedDict([
                ('v0', [v.min(), v.max()]),
                ('w_obs_20', [0, 1500]),
                ('w_obs_50', [0, 1500]),
                ('w_obs_peak', [0, 1500]),
                ('psi_obs_max', [0, 0.1]),
                ('psi_obs_0', [0, 0.1])
            ])
        else:
            self.bounds = bounds
        self.ndim = len(self.bounds)

    def apply_bounds(self, params):
        """
        Additional prior to ensure constraints are met in the sampling.
        """
        return (params[1]>params[2]) and (params[2]>params[3]) and (params[4]>params[5])

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
        params = []
        #log_params = range(1,6)  # These parameters need to be exponentiated
        log_params = []

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

    def fit(self, n_live_points=500, chain_name='hi_run', convert_to_binary=False):
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
        pymultinest.run(self.loglike, self.prior, self.ndim, importance_nested_sampling = False, init_MPI = False,
            resume = False, verbose = False, sampling_efficiency = 'parameter', n_live_points = n_live_points,
            outputfiles_basename = chain_name, multimodal=False)

        if convert_to_binary:
            # These are the files we can convert
            ext = ['ev.dat', 'phys_live.points', 'live.points', '.txt', 'post_equal_weights.dat']

            for e in ext:
                infile = os.path.join(chain_name+e)
                outfile = infile+'.npy'
                x = np.loadtxt(infile)
                np.save(outfile, x)
                os.system('rm %s' % infile)

        print('Time taken',(time.time()-t1)/60,'minutes')
class FitCatalogue:
    """
    Fit an entire catalogue of data
    """
    def __init__(self, cat):
        """
        Class to fit a catalogue of data, in parallel if requested
        Parameters
        ----------
        cat : pandas.HDFStore
            Catalogue of data in an HDF5 format where each object is a new table
        """
        self.cat = cat


    def fit_all(self, nprocesses=1, output_dir='output', n_live_points=500, convert_to_binary=True):
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
        """

        if nprocesses>1:
            ids = self.cat.keys()
            new_func = partial(_fit_object, cat=dict(self.cat), output_dir=output_dir, n_live_points=n_live_points,
                               convert_to_binary=convert_to_binary)
            p = Pool(nprocesses)
            p.map(new_func, ids)

        else:
            ids = self.cat.keys()

            for i in ids[:1]:
                _fit_object(i, self.cat, output_dir = output_dir, n_live_points = n_live_points,
                convert_to_binary = convert_to_binary)


class ChainAnalyser:
    """
    Class with convenience functions to analyse multinest output.
    """
    def __init__(self, chain_name):
        """
        Multinest chain analysis class.

        Parameters
        ----------
        chain_name : str
            The full root of all the chains (e.g. '/my/multinest/chain-') such that <chain_name>.stats and other
             output files exist
        """
        self.chain_name = chain_name

    def convert_z(self, v):
        c = 3e5
        return -(v/(v+c))

    def p_of_z(self, delta_z=0.1, v0_ind=0):
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

        post = self.chain_name+'post_equal_weights.dat'
        if os.path.exists(post+'.npy'):
            chain = np.load(post+'.npy')
        else:
            chain = np.loadtxt(post)

        z = self.convert_z(chain[:, v0_ind])

        nbins = (int)((z.max() - z.min())/delta_z)

        pdf, bins = np.histogram(z, bins=nbins, density=True)

        # We want to return the mid points of the bins
        new_bins = (bins[1:] + bins[:-1])/2

        return new_bins, pdf

    def parameter_estimates(self):

        #log_params = range(1,6)
        post = self.chain_name + 'post_equal_weights.dat'
        if os.path.exists(post + '.npy'):
            chain = np.load(post + '.npy')
        else:
            chain = np.loadtxt(post)

        #chain[log_params] = np.exp(chain[log_params])

        param_names = ['v0', 'w_obs_20', 'w_obs_50', 'w_obs_peak', 'psi_obs_max', 'psi_obs_0', 'z']

        z = self.convert_z(chain[:,0])
        logpost = chain[:,-1]
        chain = np.column_stack((chain[:,:-1],z))

        parameters = pd.DataFrame(columns = ['Mean', 'Median', 'MAP', '16p', '84p'], index=param_names)

        parameters['Mean'] = np.mean(chain, axis=0)
        parameters['Median'] = np.median(chain, axis=0)
        parameters['MAP'] = chain[np.argmax(logpost),:]
        parameters['16p'] = np.percentile(chain, 16, axis=0)
        parameters['84p'] = np.percentile(chain, 84, axis=0)

        return parameters





















