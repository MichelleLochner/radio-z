from __future__ import division, print_function
import numpy as np


class LineProfile:
    """
    Class describing a symmetric HI 21cm spectral line profile as parameterised in
    Obreschkow et al (2009) arXiv:0908.0983.
    """
    def __init__(self,
                 v0,
                 w_obs_20, w_obs_50, w_obs_peak,
                 psi_obs_max, psi_obs_0):
        """
        HI line profile

        Parameters
        ----------
        v0 : number
          Observed velocity [km/s]
        w_obs_peak : float
          Line width between the two horns of the HI-line profile [km/s]
        w_obs_50 : float
          Line width at 50% of peak luminosity density [km/s]
        w_obs_20 : float
          Line width at 20% of peak luminosity density [km/s]
        psi_obs_max : float
          Normalized peak flux density [Jy]
        psi_obs_0 : float
          Normalized flux density at the line center [Jy]
        """
        self.v0 = v0
        self.w_obs_20 = w_obs_20
        self.w_obs_50 = w_obs_50
        self.w_obs_peak = w_obs_peak
        self.psi_obs_max = psi_obs_max
        self.psi_obs_0 = psi_obs_0

    def _k1(self):
        """
        Functional form parameter for HI line profile.
        As defined in Obreschkow et al (2009) arXiv:0908.0983 Eq. A2
    
        Returns
        -------
        k1 : number
        """
        numerator = np.log(self.w_obs_50 - self.w_obs_peak) - np.log(2.)
        denominator = np.log(self.w_obs_50 - self.w_obs_peak) - np.log(self.w_obs_20 - self.w_obs_peak)

        retvar = -0.693*pow(2.322, numerator/denominator)
        if retvar == 0.0:
            retvar = -1.e-11
        return retvar

    def _k2(self):
        """
        Functional form parameter for HI line profile.
        As defined in Obreschkow et al (2009) arXiv:0908.0983 Eq. A3
    
        Returns
        -------
        k2 : number
        """
        retvar = 0.842/(np.log(self.w_obs_20 - self.w_obs_peak) - np.log(self.w_obs_50 - self.w_obs_peak))

        return retvar

    def _k3(self):
        """
        Functional form parameter for HI line profile.
        As defined in Obreschkow et al (2009) arXiv:0908.0983 Eq. A4
    
        Returns
        -------
        k3 : number
        """
        retvar = self.w_obs_peak/2.

        return retvar

    def _k4(self):
        """
        Functional form parameter for HI line profile.
        As defined in Obreschkow et al (2009) arXiv:0908.0983 Eq. A5
    
        Returns
        -------
        k4 : number
        """
        if self.psi_obs_max == self.psi_obs_0:
            retvar = 0.
        elif self.psi_obs_max > 0:
            retvar = 0.25 * (((self.w_obs_peak**2.)*(self.psi_obs_max**2.))
                             /(self.psi_obs_max**2. - self.psi_obs_0**2.))
        else:
            # Something went wrong
            retvar = -1

        return retvar

    def _k5(self):
        """
        Functional form parameter for HI line profile.
        As defined in Obreschkow et al (2009) arXiv:0908.0983 Eq. A6
    
        Returns
        -------
        k5 : number
        """
        retvar = self.psi_obs_0*np.sqrt(self._k4())
        return retvar

    def get_line_profile(self, v, noise=0):
        """
        Produces a parameterised HI spectral line profile as specified in
        Obreschkow et al (2009) arXiv:0908.0983 Eq A1 plotted over a velocity
        interval v.

        Parameters
        ----------
        v : '~np.ndarray'
            Velocity range over which to plot the line profile [km/s]
        noise : float or array, optional
            If not zero, generates noise from a normal distribution defined by either a float (sigma) or an array
            (same size as v).

        Returns
        -------
        psi : '~np.ndarray'
            Array containing the line profile.
        """
        v = v.copy()
        v -= self.v0
        psi = np.zeros_like(v)

        fact = 20
        v0 = abs(v) < fact*self.w_obs_20
        v1 = abs(v) >= self.w_obs_peak / 2.
        v2 = (abs(v) < self.w_obs_peak / 2.) * (self.psi_obs_max > self.psi_obs_0)
        v3 = (abs(v) < self.w_obs_peak / 2.) * (self.psi_obs_max == self.psi_obs_0)

        psi[v1*v0] = self.psi_obs_max * np.exp(self._k1() * pow(abs(v[v0*v1]) - self._k3(), self._k2()))
        psi[v2*v0] = self._k5() * pow(self._k4() - v[v0*v2] ** 2., -0.5)
        psi[v3*v0] = self.psi_obs_0

        norm = psi.max() / self.psi_obs_max
        psi = psi * norm

        if hasattr(noise, "__len__") or noise != 0:
            noise = np.random.randn(len(psi)) * noise
            psi = psi + noise

        return psi
