from __future__ import division, print_function
import pandas as pd
from tables.exceptions import HDF5ExtError  # Needed to catch errors when loading hdf5 files
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from radio_z import hiprofile
import os


class SaxCatalogue:
    """
    Reads in a Sax catalogue
    """

    def __init__(self, filename, nu_rest=1.42e3):
        """
        Reads in a Sax catalogue

        Parameters
        ----------
        filename : str
            Either an ascii, csv file or an hdf5 file
        nu_rest : float, optional
            Rest frame frequency of the line in question (e.g. neutral hydrogen) in MHz
        """
        self.filename = filename
        self.nu_rest = nu_rest

    def convert_parameters(self, df):
        """
        Given a pandas dataframe with SAX parameters, computes the parameters needed to create an HI profile and
        stores them in the same dataframe.

        Parameters
        ----------
        df : pandas.Dataframe
            A catalogue of SAX objects as a dataframe
        """
        df['id'] = 'ID' + df['id'].astype(int).astype(str) # For HDF5 naming conventions
        df['v0'] = - 3e5*df['zapparent']/(1 + df['zapparent'])
        df['w_obs_20'] = df['hiwidth20']
        df['w_obs_50'] = df['hiwidth50']
        df['w_obs_peak'] = df['hiwidthpeak']
        df['psi_obs_max'] = df['hiintflux']*df['hilumpeak']
        df['psi_obs_0'] = df['hiintflux']*df['hilumcenter']

    def get_data(self, output_file = ''):
        """
        Actually reads in the S3 parameters

        Parameters
        ----------
        output_file : str, optional
            Saves to another file

        Returns
        -------
        df : pandas.DataFrame
            Sax catalogue in the form of a pandas dataframe

        """
        try:
            hstore = pd.HDFStore(self.filename)
            # We'll assume the data is stored in a child table in the hdf5 file
            key = hstore.keys()[-1]
            df = hstore[key]

        except HDF5ExtError:
            # In which case this is a sql.result ascii file
            df = pd.read_csv(self.filename)

        if len(output_file) != 0:
            df.to_hdf(output_file, key='table')

        self.convert_parameters(df)

        return df

    def get_params(self, df, ind):
        """
        Gets the true parameters for a dataframe as an array (useful for plotting etc.)
        Parameters
        ----------
        df : pandas.DataFrame
            Catalogue dataframe
        ind :   str
            ID of object to return

        Returns
        -------
        array
            Returns just the true parameters
        """
        cols = ['v0', 'w_obs_20', 'w_obs_50', 'w_obs_peak', 'psi_obs_max', 'psi_obs_0']
        return df[df['id'] == ind][cols].as_matrix()[0]


class Survey:
    """
    Collects useful, survey-specific methods and attributes
    """
    def __init__(self, survey_name):
        """
        Initialise survey attributes

        Parameters
        ----------
        survey_name : str
            The name of the survey (e.g. 'ska1_band1')
        """

        if survey_name == 'ska1_band1':
            self.name = survey_name
            self.nu_min = 350
            self.nu_max = 1050
            self.delta_nu = 0.01
            self.s_rms = 315.e-6
            self.z_min = 0.35
            self.z_max = 3.06
            self.band = 1

        elif survey_name == 'ska1_band2':
            self.name = survey_name
            self.nu_min = 950
            self.nu_max = 1760
            self.delta_nu = 0.01
            self.s_rms = 187.e-6
            self.z_min = 0
            self.z_max = 0.5
            self.band = 2

        else:
            print('Survey name not recognised')

        self.kb = 1.38e-16/1.e4/1.e-23  # Boltzmann constant

    def aeff_on_tsys(self, nu, normed_at_1ghz=False, makeplot=False, band=0):
        """
        Function to calculate SKA *Band 1* A_eff / T_sys.
        Functionisation of python code from Phil Bull, which in turn was based on
        the SuperMongo code 'sensaswbx.mon' from Robert Braun, who comments:
        "Its all quite crude, but it reproduces the SEFD numbers provided by the
        DSH consortium in their PDR documentation."

        Use make_plot = True for testing, compare with figure 1 in http://astronomers.skatelescope.org/wp-content/uploads/2015/11/SKA1-Observing-Bands-V4.pdf

        Parameters
        ----------
        nu : float
            Input frequency [MHz]
        normed_at_1ghz : bool, optional
            If True, return normalised to the value at 1GHz
            (for re-scaling S_rms^ref as defined by Santos et al)
            Default = False
        makeplot : bool, optional
            If True, make and show a plot with the A_eff / T_sys
            Default = False
        band : int, optional
            If 0, use interpolation function for all five bands.
            If 1 or 2 use interpolation function for that band.
        """

        # Frequency coverage
        frq = np.linspace(0.35, 20., 4000)
        lfrq = np.log10(frq)

        # SKA1-MID parameters
        sdeg = 1. # FOV in sq. deg.? (PJB: Not sure...)
        nant = 1. # No. of antennae (1 == A_eff/T_sys per antenna)
        stad = 15. # Dish diameter [m]

        # Define T_recv piecewise, in each band (bands 1,5 are defined in two pieces)
        trcvb1a = 17. + 3.*(frq-0.35)/(1.05-0.35)
        trcvb1b = 17. + 3.*(frq-0.35)/(1.05-0.35)
        trcvb2 = 8.2 + 0.7*(frq-0.95)/(1.76-0.95)
        trcvb3 = 10.6 + 1.5*(frq-1.65)/(3.05-1.65)
        trcvb4 = 14.3 + 2.4*(frq-2.8)/(5.18-2.8)
        trcvb5 = 16.7 + 6.1*(frq-4.6)/(13.8-4.6)
        trcvb5b = 17. + 6.*(frq-4.6)/(24.-4.6)
        tsky = 20. * (0.408/frq)**2.75 + 2.73 \
           + 288. * ( 0.005 + 0.1314 * np.exp((lfrq-np.log10(22.23))*8.) ) # T_sky
        tspl = 4.0 # T_spillover?
        tgnd = 300. # T_ground?

        # Aperture efficiency as a fn. of frequency
        etaa0 = 0.92
        etaa = etaa0 - 70.*((3.e8/(frq*1.e9))/stad)**2.
        etaa = etaa - 0.36*(np.abs(frq-1.6)/(24.-1.6))**0.6

        # Band boundaries (GHz)
        frb1alo = 0.35
        frb1ahi = 0.58
        frb1blo = 0.58
        frb1bhi = 1.05
        frb2lo = 0.95
        frb2hi = 1.76
        frb3lo = 1.65
        frb3hi = 3.05
        frb4lo = 2.8
        frb4hi = 4.6
        frb5lo = 4.6
        frbd5hi = 13.8
        frb5hi = 26.

        # Initialise SEFD, FOV, T_recv arrays
        sefd = 1e6 * np.ones(lfrq.shape)
        fov = np.ones(lfrq.shape)
        trcv = np.ones(lfrq.shape)

        # Calculate piecewise A_eff / T_sys curve
        # (N.B. Ordering seems to have been chosen to take the largest value of A/T when
        # two bands overlap)
        bands = [
        (frb1ahi, frb1alo, trcvb1a),
        (frb1bhi, frb1blo, trcvb1b),
        (frb5hi, frb5lo, trcvb5b),
        (frb4hi, frb4lo, trcvb4),
        (frb3hi, frb3lo, trcvb3),
        (frb2hi, frb2lo, trcvb2),
        ]
        trcv_bands = []
        for fhi, flo, _trcv in bands:
            idx = np.where(np.logical_and(frq < fhi, frq >= flo))
            trcv[idx] = _trcv[idx] # Overall T_recv

            # Get per-band T_recv curve
            trcv_band = np.nan * np.ones(trcv.shape) # Should make A/T -> 0 out of band
            trcv_band[idx] = _trcv[idx]
            trcv_bands.append(trcv_band)
        trcv_bands = np.array(trcv_bands) # Array containing T_rcv separately, for each band

        # Calculate T_sys, A_eff, SEFD across all bands
        tsys = trcv + tsky + tspl
        aeff = nant * etaa * np.pi * stad**2./4.
        sefd = 2.*self.kb*tsys/aeff

        # Calculate FOV, A/T, survey speed
        fovd = 2340. * ((3.e8/(frq*1.e9))/stad)**2.
        aont = 2.*self.kb/sefd/sdeg
        surv = aont**2*fovd

        # Do the same for the separate bands
        aot_bands = []
        for i in range(trcv_bands.shape[0]):
            _tsys = trcv_bands[i] + tsky + tspl
            _aeff = nant * etaa * np.pi * stad**2./4.
            aot_bands.append( _aeff / _tsys )
        aot_bands = np.array(aot_bands) # per-band A_eff / T_sys

        # Construct interpolation function for A/T (takes freq. argument in MHz)
        interp_aont = interp1d(frq*1e3, aont, kind='linear',
                                               bounds_error=False)

        if makeplot:
            print(interp_aont(1000.))

            # Plot results
            ff = np.logspace(np.log10(350.), np.log10(20e3), 1000)
            plt.subplot(111)

            # Combined curve
            plt.plot(ff, interp_aont(ff), 'k-', lw=1.5)

            # Per-band
            for i in range(trcv_bands.shape[0]):
                plt.plot(frq*1e3, aot_bands[i], 'y--', lw=1.5)

            plt.xlabel("Freq. [MHz]", fontsize=18)
            plt.ylabel(r"$A_{\rm eff} / T_{\rm sys}$", fontsize=18)

            plt.xscale('log')
            plt.xlim((300., 30e3))
            plt.tight_layout()
            plt.show()

        if band == 0:
            interp_aont_return = interp_aont
        elif band == 1:
            aot_b1_full = np.nansum(np.vstack([aot_bands[0], aot_bands[1]]), axis=0)
            interp_aont_return = interp1d(frq*1e3,
                                                        aot_b1_full,
                                                        kind='linear',
                                                        bounds_error=False)
        elif band == 2:
            interp_aont_return = interp1d(frq*1e3,
                                                        aot_bands[5],
                                                        kind='linear',
                                                        bounds_error=False)

        if normed_at_1ghz:
            return interp_aont_return(nu) / interp_aont_return(1000.)
        else:
            return interp_aont_return(nu)

    def v2nu(self, v, nu_rest=1.42e3):
        """
        Convert velocity measurements back to observed frequency measurements.

        Parameters
        ----------
        v : array
            Velocities (in km/s)
        nu_rest : float, optional
            Rest frame frequency

        Returns
        -------
        array
            Frequency array

        """
        return (v/3.e5 + 1)*nu_rest

    def nu2v(self, nu, nu_rest=1.42e3):
        """
        Convert frequencies to velocity around rest frame reference frequency

        Parameters
        ----------
        nu : array
            Frequencies
        nu_rest : float, optional
            Rest frame frequency

        Returns
        -------
        array
            Velocity array in km/s
        """

        return 3.e5*(nu/nu_rest - 1.e0)

    def get_noise(self, v, nu_rest=1.42e3):
        """
        Returns the noise as a function of v for a given survey.

        Parameters
        ----------
        v : array
            Velocities
        nu_rest : float, optional
            Observing frequency

        Returns
        -------
        sigma : array
            Noise on the flux as a function of v
        """

        nu = self.v2nu(v, nu_rest = nu_rest)
        aot = self.aeff_on_tsys(nu, normed_at_1ghz = True, band=self.band)

        return self.s_rms * aot
        #return [self.s_rms]*len(v)

    def inband(self, df):
        """
        Given a catalogue as a dataframe objects, as a column that tests if the object will be observable in the
        chosen band.

        Parameters
        ----------
        df : pandas.DataFrame
            SAX catalogue as a pandas object
        """

        df['inband'] = (df['zapparent']>=self.z_min) & (df['zapparent']<=self.z_max)


class DataFromCatalogue:
    """
    Generate fake data from a catalogue
    """
    def __init__(self):
        """
        Tools to generate mock data.
        """

    def create_data(self, params, survey, noise=True):
        """
        Generate fake data from a dataframe of parameters and for a given survey.

        Parameters
        ----------
        params : list-like
            Contains the profile parameters (v0, w_obs_20, w_obs_50, w_obs_peak, psi_obs_max, psi_obs_0)
        survey : saxdata.Survey
            Which survey to do this for
        noise : bool, optional
            Whether or not to add noise

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the velocities, psi values and noise
        """
        nchan = int(survey.nu_max - survey.nu_min)/survey.delta_nu
        nu_range = np.linspace(survey.nu_min, survey.nu_max, nchan)
        v_range = survey.nu2v(nu_range)

        lp = hiprofile.LineProfile(*params)

        if noise:
            sigma = survey.get_noise(v_range)
        else:
            sigma = 0
        psi = lp.get_line_profile(v_range, sigma)

        return pd.DataFrame(data=np.column_stack([v_range, psi, sigma]), columns=['v', 'psi', 'psi_err'])

    def create_from_cat(self, df, survey, output_hdf5_file='catalogue_data.hdf5'):
        """
        Generates all data from a set of objects in a catalogue.

        Parameters
        ----------
        df : pandas.DataFrame
            The catalogue with parameters for each object
        survey : saxdata.Survey
            Which survey to make this for
        output_hdf5_file : str, optional
            We have to store the output inside an HDF5 file

        Returns
        -------
        pandas.HDFstore
            Each object is stored in a different HDF5 dataset, stored in one big file
        """
        
        df['data'] = ''

        ids = df.id

        cols = ['v0', 'w_obs_20', 'w_obs_50', 'w_obs_peak', 'psi_obs_max', 'psi_obs_0']  # The parameters we need
        for i in ids:
            params = df[df.id == i][cols]
            params = params.as_matrix()[0].tolist()

            data = self.create_data(params, survey, noise=True)

            cat_df['data'][i] = data


        # We delete any existing file to avoid issues with leaving files open etc.
        if os.path.exists(output_hdf5_file):
            os.system('rm -r ' + output_hdf5_file)

        hstore = pd.HDFStore(output_hdf5_file, 'a')

        ids = df.id

        cols = ['v0', 'w_obs_20', 'w_obs_50', 'w_obs_peak', 'psi_obs_max', 'psi_obs_0']  # The parameters we need
        for i in ids:
            params = df[df.id == i][cols]
            params = params.as_matrix()[0].tolist()

            data = self.create_data(params, survey, noise=True)

            hstore[i] = data

        return hstore

    def plot_profile(self, df, plot_model=False, model_params=[], plot_fit=False, fit_params=[], zoom=True,
                     fontsize=14, data_colour='#c2c2d6', model_colour='k', fit_colour='r', rotation=0):
        """
        Plots a single data set

        Parameters
        ----------
        df : pandas.DataFrame or array
            Contains the data in columns 'v', 'psi', 'psi_err'
        plot_model : bool, optional
            Whether or not to overplot the underlying model
        model_params : list-like, optional
            If the model is to be plotted, list of parameters
        plot_fit : bool, optional
            Whether or not to overplot a best fit model
        fit_params : list-like, optional
            If the best fit is to be plotted, list of parameters
        zoom : bool, optional
            Whether to zoom in on the region where the line actually is or to plot the whole thing
        fontsize : float, optional
            Font size of labels
        data_colour : str, optional
            Colour of the plotted data
        model_colour : str, optional
            Colour of the plotted model
        fit_colour : str, optional
            Colour of the plotted fit
        rotation : float, optional
            Angle by which to rotate x labels
        """

        if isinstance(df, pd.DataFrame):
            v = df['v'].as_matrix()
            psi = df['psi'].as_matrix()
        else:
            v = df[:,0]
            psi = df[:,1]

        plt.plot(v, psi, color=data_colour)

        if plot_model:
            model_params = list(model_params)
            lp = hiprofile.LineProfile(*model_params)
            psi_model = lp.get_line_profile(v, noise=0)
            plt.plot(v, psi_model, color=model_colour, lw=1.5)

        if plot_fit:
            fit_params = list(fit_params)
            lp = hiprofile.LineProfile(*fit_params)
            psi_fit = lp.get_line_profile(v, noise=0)
            plt.plot(v, psi_fit, color=fit_colour, lw=1.5)

        if zoom and len(model_params) != 0:  # We need to know where the true line is if we want to zoom in
            delta = 5*model_params[1]
            if not model_params[0]-delta<v.min():
                plt.xlim([model_params[0]-delta,model_params[0]+delta])

        plt.xlabel('Velocity (km/s)',fontsize=fontsize)
        plt.ylabel('Normalised flux density (Jy s/km)',fontsize=fontsize)
        plt.xticks(rotation=rotation)

        plt.tight_layout()
































