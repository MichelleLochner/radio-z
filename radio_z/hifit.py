from __future__ import division, print_function
import pandas as pd
import pymultinest
import numpy as np
from radio_z import hiprofile
from collections import OrderedDict
import os


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
            self.bounds = OrderedDict([
                    ('v0', [v.min(),v.max()]),
                    ('w_obs_20', [7, 20]),
                    ('w_obs_50', [7, 20]),
                    ('w_obs_peak', [7,14]),
                    ('psi_obs_max', [-11, -4]),
                    ('psi_obs_0', [-11, -4])
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
        log_params = range(1,6)  # These parameters need to be exponentiated

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

        lp = hiprofile.lineProfile(*params)
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
        keys = self.bounds.keys()
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
