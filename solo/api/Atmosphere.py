from __future__ import division
from collections import namedtuple
import os.path
import numpy as np


ATTRS = ["p", "rho", "o3", "h2o", "alpha", "beta", "w0", "g"]

# Define the default values for optional atmospheric input arguments.
DEFAULT_P = 1013.
DEFAULT_W0 = 0.90
DEFAULT_G = 0.85

# Load the array of molecular absorption coefficients in read-only mode.
DIRFOLD = os.path.dirname(os.path.abspath(__file__))
ABSCOEF_PATH = os.path.join(os.path.dirname(DIRFOLD), "dat", "abscoef.dat")
ABSCOEF = np.loadtxt(ABSCOEF_PATH, usecols=(0, 1, 2, 3, 4)).T
ABSCOEF.flags.writeable = False


class Atmosphere(namedtuple("Atmosphere", ATTRS)):
    """Class to define the atmospheric properties.

    Every instance allows the access to the following properties:

        nscen : int
            number of scenarios

        p : array-like, shape (nscen,)
            atmospheric surface pressure in hPa

        rho : array-like, shape (nscen,)
            surface albedo

        o3 : array-like, shape (nscen,)
            vertical ozone content in DU

        h2o : array-like, shape (nscen,)
            total amount of water vapour in cm-pr

        alpha : array-like, shape (nscen,)
            Angstrom alpha parameter

        beta : array-like, shape (nscen,)
            Angstrom beta parameter

        w0 : array-like, shape (nscen,)
            single scattering albedo

        g : array-like, shape (nscen,)
            aerosol asymmetry parameter
    """

    def __new__(cls, p, rho, o3, h2o, alpha, beta, w0=None, g=None):
        """Return a new instance of Atmosphere.

        Receive:

            p : array-like, shape (nscen,)
                atmospheric surface pressure in hPa
            rho : array-like, shape (nscen,)
                surface albedo
            o3 : array-like, shape (nscen,)
                vertical ozone content in DU
            h2o : array-like, shape (nscen,)
                total amount of water vapour in cm
            alpha : array-like, shape (nscen,)
                Angstrom alpha parameter
            beta : array-like, shape (nscen,)
                Angstrom beta parameter
            w0 : array-like, shape (nscen,)
                single scattering albedo, default given by DEFAULT_W0
            g : array-like, shape (nscen,)
                aerosol asymmetry parameter, default given by DEFAULT_G

        Return:

            atm : Atmosphere
                instance of Atmosphere based on the input parameters

        Raise:

            AttributeError
                if input arguments have inconsistent or wrong shapes
            ValueError
                if the input arguments are out of range
        """

        # Ensure that the input arguments have consistent shapes and sizes.
        items = [p, rho, o3, h2o, alpha, beta]
        items = items + [x for x in [w0, g] if x is not None]
        set_shapes = set(np.shape(x) for x in items)
        if len(set_shapes) > 1:
            raise AttributeError("size mismatch among input arguments")
        set_shapes = list(set_shapes)[0]
        if len(set_shapes) > 1:
            raise AttributeError("input arguments must be 0- or 1-dimensional")

        # Ensure that the input arguments are within range and set the default
        # values for 'w0' and 'g' if they were not defined.
        p = np.atleast_1d(p)
        if np.any(p < 0):
            raise ValueError("pressure out of range")
        rho = np.atleast_1d(rho)
        if np.any(rho < 0) or np.any(rho > 1):
            raise ValueError("albedo out of range")
        o3 = np.atleast_1d(o3)
        if np.any(o3 < 0):
            raise ValueError("ozone out of range")
        h2o = np.atleast_1d(h2o)
        if np.any(h2o < 0):
            raise ValueError("water vapour out of range")
        alpha = np.atleast_1d(alpha)
        if np.any(alpha < 0):
            raise ValueError("Angstrom alpha out of range")
        beta = np.atleast_1d(beta)
        if np.any(beta < 0):
            raise ValueError("Angstrom beta out of range")
        if w0 is None:
            w0 = DEFAULT_W0 * np.ones(shape=set_shapes, dtype=float)
        elif np.any(w0 < 0) or np.any(w0 > 1):
            raise ValueError("single scattering albedo out of range")
        w0 = np.atleast_1d(w0)
        if g is None:
            g = DEFAULT_G * np.ones(shape=set_shapes, dtype=float)
        elif np.any(np.abs(g) > 1):
            raise ValueError("asymmetry parameter out of range")
        g = np.atleast_1d(g)

        # Return the new instance.
        args = [cls, p, rho, o3, h2o, alpha, beta, w0, g]
        atm = super(Atmosphere, cls).__new__(*args)
        return atm

    @property
    def nscen(self):
        """Return the number of scenarios stored within the instance."""

        return np.shape(self.p)[0]

    @property
    def abscoef(self):
        """Return the molecular absorption coefficients.

        The first row stores the wavelengths in nm (from 300 to 2600 nm),
        while the other rows provide the absorption coefficients for the
        following constituents:

            1 : water vapour,
            2 : water vapour,
            3 : ozone,
            4 : molecular oxygen.

        Return:

            abscoef : array-like, shape (1 + ncoef, nwvln)
                absorption coefficients for different types of molecules
                as a function of the wavelength in nm, which is stored
                in the first row
        """

        return ABSCOEF

    def tau_rayleigh(self, wvln_um, return_albedo=False):
        """Return the Rayleigh optical depth for the given wavelengths.

        The optical depth is computed by using Bates's formula:

            tau_ray(wvln) = 1 / (117.2594 * (wvln / um)**4
                                 - 1.3215 * (wvln / um)**2 + 0.000320
                                 - 0.000076 * (wvln / um)**(-2)),

        which is obtained for a location with atmospheric pressure equal
        to 1 atm. For other pressures, the formula must be multiplied by
        the factor (pressure / 1013 hPa).

        Receive:

            wvln_um : array-like, shape (nwvln,)
                wavelengths in microns
            return_albedo : bool, optional
                if True, return also the Rayleigh contribution to the
                atmospheric albedo (default False)

        Return:

            tau : array-like, shape (nscen, nwvln)
                Rayleigh optical depth for every scenario and wavelength
            salb : array-like, shape (nscen, nwvln), optional
                Rayleigh contribution to the atmospheric albedo

        Raise:

            ValueError
                if the input 'wvln_um' does not have a proper shape
            TypeError
                if 'return_albedo' is not a boolean flag
        """

        # Ensure the shape and type of the input arguments.
        if np.ndim(wvln_um) > 1:
            raise ValueError("'wvln_um' must be 0- or 1-dimensional")
        if not isinstance(return_albedo, bool):
            raise TypeError("'return_albedo' must be a bool")

        # Define the coefficients used in Bates' formula.
        c = [117.2594, -1.3215, 0.000320, -0.000076]

        # Broadcast arrays before the computation of 'tau'.
        wvln_um = np.atleast_1d(wvln_um)
        pressure = self.p[:, None]

        # Compute the optical thickness using Bates' formula, which must be
        # corrected with the real pressure because the original formula is
        # only valid for an atmospheric pressure of 1 atm.
        wvln_um2 = wvln_um**2
        wvln_um4 = wvln_um2**2
        div = c[0] * wvln_um4 + c[1] * wvln_um2 + c[2] + c[3] / wvln_um4
        tau = (pressure / DEFAULT_P) / div

        # If requested, calc Rayleigh contribution to the atmospheric albedo.
        if return_albedo:
            salb = tau * (1. - np.exp(-2. * tau)) / (2. + tau)
            salb = (salb,)
        else:
            salb = ()

        out = (tau,) + salb
        return out if len(out) > 1 else out[0]

    def tau_aerosols(self, wvln_um, return_albedo=False):
        """Return the aerosol optical depth for the given wavelengths.

        The optical depth is computed by using the Angstrom's formula:

            tau_aer(wvln) = beta * (wvln / um)**(-alpha).

        Receive:

            wvln_um : array-like, shape (nwvln,)
                wavelengths in microns
            return_albedo : bool, optional
                if True, return also the aerosol contribution to the
                atmospheric albedo (default False)

        Return:

            tau : array-like, shape (nscen, nwvln)
                aerosol optical depth for every scenario and wavelength
            salb : array-like, shape (nscen, nwvln), optional
                aerosol contribution to the atmospheric albedo

        Raise:

            ValueError
                if the input 'wvln_um' does not have a proper shape
            TypeError
                if 'return_albedo' is not a boolean flag
        """

        # Ensure the shape and type of the input arguments.
        if np.ndim(wvln_um) > 1:
            raise ValueError("'wvln_um' must be 0- or 1-dimensional")
        if not isinstance(return_albedo, bool):
            raise TypeError("'return_albedo' must be a bool")

        # Broadcast arrays before the computation of 'tau'.
        wvln_um = np.atleast_1d(wvln_um)
        alpha = self.alpha[:, None]
        beta = self.beta[:, None]

        # Compute the optical thickness using Angstrom's formula.
        tau = beta / wvln_um**alpha

        # If requested, calc aerosol contribution to the atmospheric albedo.
        if return_albedo:
            g = ((1 - self.g) * self.w0)[:, None]
            salb = g * tau / (2. + g * tau) * (1. + np.exp(-g * tau))
            salb = (salb,)
        else:
            salb = ()

        out = (tau,) + salb
        return out if len(out) > 1 else out[0]

    def trn_rayleigh(self, wvln_um, mu0, return_albedo=False):
        """Return the Rayleigh transmittances.

        The direct transmittance is just computed as:

            tdir_ray(wvln) = np.exp(-tau_ray(wvln) / mu0),

        while Sobolev's formula is used for the global transmittance:

            tglb_ray(wvln) = ((2/3 + mu0) + (4/3 - mu0) * tdir_ray))
                             / (2/3 + tau_ray),

        and the diffuse transmittance is the difference between the
        global and the direct transmittances:

            tdif_ray(wvln) = tglb_ray - tdir_ray.

        Receive:

            wvln_um : array-like, shape (nwvln,)
                wavelengths in microns
            mu0 : array-like, shape (nscen,)
                cosine of solar zenith angles
            return_albedo : bool, optional
                if True, return also the Rayleigh contribution to the
                atmospheric albedo (default False)

        Return:

            tglb : array-like, shape (nscen, nwvln)
                Rayleigh global transmittance for every scenario
                and wavelength
            tdir : array-like, shape (nscen, nwvln)
                Rayleigh direct transmittance for every scenario
                and wavelength
            tdif : array-like, shape (nscen, nwvln)
                Rayleigh diffuse transmittance for every scenario
                and wavelength
            salb : array-like, shape (nscen, nwvln), optional
                Rayleigh contribution to the atmospheric albedo

        Raise:

            ValueError
                if the input 'wvln_um' or 'mu0' have invalid shape
            IndexError
                if the shape of 'mu0' does not match to the number of
                scenarios in the Atmosphere instance
            TypeError
                if 'return_albedo' is not a boolean flag
        """

        # Ensure the shape of 'mu0'. The other arguments are checked when
        # calling the method 'tau_rayleigh'.
        if np.ndim(mu0) > 1:
            raise ValueError("'mu0' must be 0- or 1-dimensional")
        mu0 = np.atleast_1d(mu0)[:, None]
        if self.nscen != 1 and mu0.shape[0] not in [1, self.nscen]:
            msg = "mismatch in shapes of 'mu0' and the Atmosphere instance"
            raise IndexError(msg)

        # Compute the optical thickness and the atmospheric albedo.
        args = [wvln_um, return_albedo]
        out = self.tau_rayleigh(*args)
        tau, salb = (out[0], (out[1],)) if return_albedo else (out, ())

        # Compute the Rayleigh direct transmittance.
        tdir = np.exp(-tau / mu0)

        # Compute the global and diffuse transmittances.
        c = [2. / 3., 4. / 3.]
        tglb = ((c[0] + mu0) + (c[0] - mu0) * tdir) / (c[1] + tau)
        tdif = tglb - tdir

        out = (tglb, tdir, tdif) + salb
        return out

    def trn_aerosols(self, wvln_um, mu0, return_albedo=False, coupling=False):
        """Return the aerosol transmittances.

        The direct transmittance is just computed as:

            tdir_aer(wvln) = np.exp(-tau_aer(wvln) / mu0),

        while Ambartsumian's formulation is used for the global
        transmittance assuming that w0 is not 1:

            tglb_aer(wvln) = ((1 - r0**2) * np.exp(-K * tau_aer / mu0))
                             / (1 - r0**2 * np.exp(-K * tau_aer / mu0)),

        where:

            K = np.sqrt((1 - w0) * (1 - w0 * g)),
            r0 = (K - 1 + w0) / (K + 1 - w0),

        and the diffuse transmittance is the difference between the
        global and the direct transmittances:

            tdif_aer(wvln) = tglb_aer - tdir_aer.

        Receive:

            wvln_um : array-like, shape (nwvln,)
                wavelengths in microns
            mu0 : array-like, shape (nscen,)
                cosine of solar zenith angles
            return_albedo : bool, optional
                if True, return also the aerosol contribution to the
                atmospheric albedo (default False)
            coupling : bool, optional
                if True, include Rayleigh-aerosol coupling effect;
                this parameter is intended only for internal use
                (default False)

        Return:

            tglb : array-like, shape (nscen, nwvln)
                aerosol global transmittance for every scenario
                and wavelength
            tdir : array-like, shape (nscen, nwvln)
                aerosol direct transmittance for every scenario
                and wavelength
            tdif : array-like, shape (nscen, nwvln)
                aerosol diffuse transmittance for every scenario
                and wavelength
            salb : array-like, shape (nscen, nwvln), optional
                aerosol contribution to the atmospheric albedo

        Raise:

            ValueError
                if the input 'wvln_um' or 'mu0' have invalid shape
            IndexError
                if the shape of 'mu0' does not match to the number of
                scenarios in the Atmosphere instance
            TypeError
                if 'return_albedo' or 'coupling' are not boolean flags
        """

        # Ensure the type of 'coupling' and the shape of 'mu0'. The other
        # arguments are already checked when calling the method 'tau_aerosols'.
        if not isinstance(coupling, bool):
            raise TypeError("'coupling' must be a bool")
        if np.ndim(mu0) > 1:
            raise ValueError("'mu0' must be 0- or 1-dimensional")
        mu0 = np.atleast_1d(mu0)[:, None]
        if self.nscen != 1 and mu0.shape[0] not in [1, self.nscen]:
            msg = "mismatch in shapes of 'mu0' and the Atmosphere instance"
            raise IndexError(msg)

        # Compute the optical thickness and the atmospheric albedo.
        args = [wvln_um, return_albedo]
        out = self.tau_aerosols(*args)
        tau, salb = (out[0], out[1]) if return_albedo else (out, ())
        g, w0 = [x[:, None] for x in (self.g, self.w0)]

        # If requested, Rayleigh contribution is coupled to the aerosols.
        if coupling:
            out = self.tau_rayleigh(*args)
            tau_ray, sray = (out[0], out[1]) if return_albedo else (out, ())
            tau_aer, saer = tau, salb
            # Update the optical depth and the atmospheric albedo.
            tau, salb = tau_ray + tau_aer, sray + saer
            # Update the effective asymmetry parameter and single-scattering
            # albedo due to Rayleigh-aerosol coupling.
            g = (tau_aer * g) / tau
            w0 = (tau_ray + w0 * tau_aer) / tau
        salb = (salb,)

        # Compute intermediate parameters.
        ak = np.sqrt((1. - w0) * (1. - w0 * g))
        r0 = ((ak - 1. + w0) / (ak + 1. - w0))

        # Compute direct, global and diffuse transmittances.
        tdir = np.exp(-tau / mu0)
        tdir_k = tdir**ak
        tglb = (1. - r0**2) * tdir_k / (1. - (r0 * tdir_k)**2)
        tdif = tglb - tdir

        out = (tglb, tdir, tdif) + salb
        return out

    def trn_mixture(self, wvln_um, mu0, return_albedo=False, coupling=False):
        """Return the transmittances for the Rayleigh-aerosols mixture.

        The methos allows to consider these transmittance just as a
        combination of independent processes (Rayleigh and aerosols),
        but also as a process with coupling effects.

        Receive:

            wvln_um : array-like, shape (nwvln,)
                wavelengths in microns
            mu0 : array-like, shape (nscen,)
                cosine of solar zenith angles
            return_albedo : bool, optional
                if True, return also the atmospheric albedo
                (default False)
            coupling : bool, optional
                if True, include Rayleigh-aerosol coupling effect
                (default False)

        Return:

            tglb : array-like, shape (nscen, nwvln)
                mixture global transmittance for every scenario
                and wavelength
            tdir : array-like, shape (nscen, nwvln)
                mixture direct transmittance for every scenario
                and wavelength
            tdif : array-like, shape (nscen, nwvln)
                mixture diffuse transmittance for every scenario
                and wavelength
            salb : array-like, shape (nscen, nwvln), optional
                atmospheric albedo

        Raise:

            ValueError
                if the input 'wvln_um' or 'mu0' have invalid shape
            IndexError
                if the shape of 'mu0' does not match to the number of
                scenarios in the Atmosphere instance
            TypeError
                if 'return_albedo' or 'coupling' are not boolean flags
        """

        # Ensure the type of 'coupling'. The other arguments are already
        # checked when calling the methods 'trn_rayleigh' and 'trn_aerosols'.
        if not isinstance(coupling, bool):
            raise TypeError("'coupling' must be a bool")

        if coupling:
            # Call aerosol transmittance method with coupling set to True.
            args = [wvln_um, mu0, return_albedo, True]
            out = self.trn_aerosols(*args)
            tglb, tdir, tdif = out[:3]
            salb = (out[3],) if return_albedo else ()
        else:
            args = [wvln_um, mu0, return_albedo]
            # Compute Rayleigh transmittances.
            out = self.trn_rayleigh(*args)
            tglb_ray, tdir_ray, tdif_ray = out[:3]
            sray = out[3] if return_albedo else ()
            # Compute aerosol transmittances.
            out = self.trn_aerosols(*args)
            tglb_aer, tdir_aer, tdif_aer = out[:3]
            saer = out[3] if return_albedo else ()
            # Compute mix transmittances without Rayleigh-aerosol coupling.
            tglb = tglb_ray * tglb_aer
            tdir = tdir_ray * tdir_aer
            tdif = tglb - tdir
            salb = (sray + saer,) if return_albedo else ()

        out = (tglb, tdir, tdif) + salb
        return out

    def trn_water(self, wvln, mu0):
        """Return the transmittance due to water vapour absorption.

        The transmittance is computed by using the formula:

            trn(wvln) = np.exp(-(kabs_h2o(wvln) * path_h2o / mu0)**a),

        where 'kabs_o2' denotes the water vapour absorption coefficients
        in cm-1, 'path_h2o' is the water vapour absorption path given in
        cm, 'mu0' is the cosine of the solar zenith angle and 'a' is an
        empirical exponent (which depends on the wavelength for the
        water vapour).

        Receive:

            wvln : array-like, shape (nwvln,)
                wavelengths in nanometers
            mu0 : array-like, shape (nscen,)
                cosines of the solar zenith angle

        Return:

            trn : array-like, shape (nscen, nwvln)
                water vapour transmittance for every scenario
                and wavelength

        Raise:

            ValueError
                if 'wvln' or 'mu0' have invalid shapes
            IndexError
                if the shape of 'mu0' does not match to the number of
                scenarios in the Atmosphere instance
        """

        # Ensure the shape and type of the input arguments.
        if np.ndim(wvln) > 1:
            raise ValueError("'wvln' must be 0- or 1-dimensional")
        wvln = np.atleast_1d(wvln)
        if np.ndim(mu0) > 1:
            raise ValueError("'mu0' must be 0- or 1-dimensional")
        mu0 = np.atleast_1d(mu0)[:, None]
        if self.nscen != 1 and mu0.shape[0] not in [1, self.nscen]:
            msg = "mismatch in shapes of 'mu0' and the Atmosphere instance"
            raise IndexError(msg)

        # Compute the absorption coefficients and exponents for water vapour
        # at the given input wavelengths by using linear interpolation.
        water_coef = np.interp(wvln, *self.abscoef[[0, 1]])
        water_exp = np.interp(wvln, *self.abscoef[[0, 2]])
        water_path = self.h2o[:, None]

        trn = np.where(np.isclose(water_exp, 0.0), 1.0,
                       np.exp(-(water_coef * water_path / mu0)**water_exp))
        return trn

    def trn_ozone(self, wvln, mu0):
        """Return the transmittance due to ozone absorption.

        The transmittance is computed by using the formula:

            trn_o3(wvln) = np.exp(-kabs_o3(wvln) * path_o3 / mu0),

        where 'kabs_o3' denotes the ozone absorption coefficients
        in cm-1, 'path_o3' is the ozone absorption path given in cm
        and 'mu0' is the cosine of the solar zenith angle.

        Receive:

            wvln : array-like, shape (nwvln,)
                wavelengths in nanometers
            mu0 : array-like, shape (nscen,)
                cosines of the solar zenith angle

        Return:

            trn : array-like, shape (nscen, nwvln)
                ozone transmittance for every scenario and wavelength

        Raise:

            ValueError
                if the input 'wvln' does not have a proper shape
            IndexError
                if the shape of 'mu0' does not match to the number of
                scenarios in the Atmosphere instance
        """

        # Ensure shape of the input arguments.
        if np.ndim(wvln) > 1:
            raise ValueError("'wvln' must be 0- or 1-dimensional")
        wvln = np.atleast_1d(wvln)
        if np.ndim(mu0) > 1:
            raise ValueError("'mu0' must be 0- or 1-dimensional")
        mu0 = np.atleast_1d(mu0)[:, None]
        if self.nscen != 1 and mu0.shape[0] not in [1, self.nscen]:
            msg = "mismatch in shapes of 'mu0' and the Atmosphere instance"
            raise IndexError(msg)

        # Compute the absorption cross sections for ozone at the given input
        # wavelengths by using linear interpolation, and convert them to
        # absorption coefficients in cm-1 by using Loschmidt's number.
        ozone_xsec = np.interp(wvln, *self.abscoef[[0, 3]])
        ozone_coef = (2.687E19 * ozone_xsec)

        # Convert from ozone amount in DU to ozone absorption path in cm.
        ozone_path = (1E-3 * self.o3)[:, None]

        trn = np.exp(-ozone_coef * ozone_path / mu0)
        return trn

    def trn_oxygen(self, wvln, mu0):
        """Return the transmittance due to molecular oxygen absorption.

        The transmittance is computed by using the formula:

            trn(wvln) = np.exp(-(kabs_o2(wvln) * path_o2 / mu0)**a),

        where 'kabs_o2' denotes the oxygen absorption coefficients
        in cm-1, 'path_o2' is the oxygen absorption path given in cm,
        'mu0' is the cosine of the solar zenith angle and 'a' is an
        empirical exponent (equal to 0.5641 for the molecular oxygen).

        Receive:

            wvln : array-like, shape (nwvln,)
                wavelengths in nanometers
            mu0 : array-like, shape (nscen,)
                cosines of the solar zenith angle

        Return:

            trn : array-like, shape (nscen, nwvln)
                oxygen transmittance for every scenario and wavelength

        Raise:

            ValueError
                if 'wvln' or 'mu0' have invalid shapes
        """

        # Ensure shape of the input arguments.
        if np.ndim(wvln) > 1:
            raise ValueError("'wvln' must be 0- or 1-dimensional")
        wvln = np.atleast_1d(wvln)
        if np.ndim(mu0) > 1:
            raise ValueError("'mu0' must be 0- or 1-dimensional")
        mu0 = np.atleast_1d(mu0)[:, None]
        if self.nscen != 1 and mu0.shape[0] not in [1, self.nscen]:
            msg = "mismatch in shapes of 'mu0' and the Atmosphere instance"
            raise IndexError(msg)

        # Compute the absorption coefficients for oxygen at the given input
        # wavelengths by using linear interpolation.
        oxygen_coef = np.interp(wvln, *self.abscoef[[0, 4]])

        # Declare the oxygen path and the oxygen exponent as constants.
        oxygen_path = np.array([0.209 * 173200])[:, None]
        oxygen_exp = 0.5641

        trn = np.exp(-(oxygen_coef * oxygen_path / mu0)**oxygen_exp)
        return trn

    @staticmethod
    def from_file(path):
        """Create Atmosphere instance from file.

        Receive:

            path : str
                location of input file

        Return:

            atm : Atmosphere
                instance of Atmosphere based on the input file

        Raise:

            ValueError
                if the input file does not have a valid format
        """

        data = np.atleast_2d(np.loadtxt(path))

        # Case if the file contains one input scenario in 4-row format.
        if data.shape in [(3, 2), (4, 2)]:
            args = data.ravel()
        # Case if the file contains n input scenarios in 1-row format.
        elif data.shape[1] in [6, 8]:
            args = data.T
        else:
            raise ValueError("invalid file format")

        return Atmosphere(*args)

