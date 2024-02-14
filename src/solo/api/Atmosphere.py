# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-2019, 2023 Víctor Molina García
#
# This file is part of solo.
#
# solo is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# solo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with solo; if not, see <https://www.gnu.org/licenses/>.
#
"""Atmosphere class encapsulation."""

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

    Attributes
    ----------

    p : array-like
        atmospheric surface pressure in hPa

    rho : array-like
        surface albedo

    o3 : array-like
        vertical ozone content in DU

    h2o : array-like
        total amount of water vapour in cm-pr

    alpha : array-like
        Angstrom alpha parameter

    beta : array-like
        Angstrom beta parameter

    w0 : array-like, optional
        single scattering albedo; if not given, it defaults to 0.90

    g : array-like, optional
        aerosol asymmetry parameter; if not given, it defaults to 0.85
    """

    def __new__(cls, p, rho, o3, h2o,  # pylint: disable=too-many-arguments
                alpha, beta, w0=None, g=None):
        """Return a new :class:`Atmosphere` instance."""

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
        # values for `w0` and `g` if they were not defined.
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
            w0 = np.full(shape=set_shapes, fill_value=DEFAULT_W0, dtype=float)
        elif np.any(w0 < 0) or np.any(w0 > 1):
            raise ValueError("single scattering albedo out of range")
        w0 = np.atleast_1d(w0)
        if g is None:
            g = np.full(shape=set_shapes, fill_value=DEFAULT_G, dtype=float)
        elif np.any(np.abs(g) > 1):
            raise ValueError("asymmetry parameter out of range")
        g = np.atleast_1d(g)

        # Return the new instance.
        args = [cls, p, rho, o3, h2o, alpha, beta, w0, g]
        atm = super(Atmosphere, cls).__new__(*args)
        return atm

    @property
    def nscen(self):
        """Number of scenarios stored by the instance."""

        return np.shape(self.p)[0]

    @property
    def abscoef(self):
        """Molecular absorption coefficients.

        The returned array of molecular absorption coefficients has
        shape ``(1 + ncoef, nwvln)``.

        The first row stores the wavelengths in nm (from 300 to 2600 nm),
        while the other rows provide the absorption coefficients for the
        following constituents:

            1. water vapour,
            2. water vapour,
            3. ozone,
            4. molecular oxygen.
        """

        return ABSCOEF

    def tau_rayleigh(self, wvln_um, return_albedo=False):
        r"""Return the Rayleigh optical depth for the given wavelengths.

        The optical depth is computed by using the Bates' formula:

        .. math::
            \tau_\text{ray}(\lambda) =
                \dfrac{1}
                      {117.2594 \times {\lambda_u}^4
                       - 1.3215 \times {\lambda_u}^2
                       + 0.000320 - 0.000076 \times {\lambda_u}^{-2}},

        where :math:`\lambda_u = \lambda / \mu\text{m}` is equal to
        the wavelength :math:`\lambda` in microns without units, and
        which is obtained for a reference location with atmospheric
        pressure :math:`p_0 = 1\text{ atm}`. For other pressures
        :math:`p` the formula is multiplied internally by the
        factor :math:`(p / p_0)`.

        Parameters
        ----------

        wvln_um : array-like
            wavelengths in microns, with shape ``(nwvln,)``

        return_albedo : bool, optional
            if True, return also the Rayleigh contribution to the
            atmospheric albedo

        Returns
        -------

        tau : array-like
            Rayleigh optical depth, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        salb : array-like, optional
            Rayleigh contribution to the atmospheric albedo, with shape
            ``(nscen, nwvln)``, for every scenario and wavelength

        Raises
        ------

        ValueError
            if the input ``wvln_um`` does not have a proper shape

        TypeError
            if ``return_albedo`` is not a boolean flag
        """

        # Ensure the shape and type of the input arguments.
        if np.ndim(wvln_um) > 1:
            raise ValueError("'wvln_um' must be 0- or 1-dimensional")
        if not isinstance(return_albedo, bool):
            raise TypeError("'return_albedo' must be a bool")

        # Define the coefficients used in Bates' formula.
        c = [117.2594, -1.3215, 0.000320, -0.000076]

        # Broadcast arrays before the computation of `tau`.
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
        r"""Return the aerosol optical depth for the given wavelengths.

        The optical depth is computed by using the Angstrom's formula:

        .. math::
            \tau_\text{aer}(\lambda) = \beta \times {\lambda_u}^{-\alpha},

        where :math:`\lambda_u = \lambda / \mu\text{m}` is equal to
        the wavelength :math:`\lambda` in microns without units.

        Parameters
        ----------

        wvln_um : array-like
            wavelengths in microns, with shape ``(nwvln,)``

        return_albedo : bool, optional
            if True, return also the aerosol contribution to the
            atmospheric albedo

        Returns
        -------

        tau : array-like
            aerosol optical depth, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        salb : array-like, optional
            aerosol contribution to the atmospheric albedo, with shape
            ``(nscen, nwvln)``, for every scenario and wavelength

        Raises
        ------

        ValueError
            if the input ``wvln_um`` does not have a proper shape

        TypeError
            if ``return_albedo`` is not a boolean flag
        """

        # Ensure the shape and type of the input arguments.
        if np.ndim(wvln_um) > 1:
            raise ValueError("'wvln_um' must be 0- or 1-dimensional")
        if not isinstance(return_albedo, bool):
            raise TypeError("'return_albedo' must be a bool")

        # Broadcast arrays before the computation of `tau`.
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
        r"""Return the Rayleigh transmittances.

        The direct transmittance is just computed as:

        .. math::
            T_\text{ray}^\text{dir}(\lambda) =
                \exp(-\tau_\text{ray}(\lambda) / \mu_0),

        while Sobolev's formula is used for the global transmittance:

        .. math::
            T_\text{ray}^\text{glb}(\lambda) =
                \dfrac{(\frac{2}{3} + \mu_0)
                       + (\frac{2}{3} - \mu_0)
                       \times T_\text{ray}^\text{dir}(\lambda)}
                      {\frac{4}{3} + \tau_\text{ray}(\lambda)},

        and the diffuse transmittance is the difference between the
        global and the direct transmittances:

        .. math::
            T_\text{ray}^\text{dif}(\lambda) =
                T_\text{ray}^\text{glb}(\lambda)
                - T_\text{ray}^\text{dir}(\lambda).

        Parameters
        ----------

        wvln_um : array-like
            wavelengths in microns, with shape ``(nwvln,)``

        mu0 : array-like
            cosine of solar zenith angles, with shape ``(nscen,)``

        return_albedo : bool, optional
            if True, return also the Rayleigh contribution to the
            atmospheric albedo

        Returns
        -------

        tglb : array-like
            Rayleigh global transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        tdir : array-like
            Rayleigh direct transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        tdif : array-like
            Rayleigh diffuse transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        salb : array-like, optional
            Rayleigh contribution to the atmospheric albedo, with shape
            ``(nscen, nwvln)``, for every scenario and wavelength

        Raises
        ------

        ValueError
            if the input ``wvln_um`` or ``mu0`` have invalid shape

        IndexError
            if the shape of ``mu0`` does not match to the number of
            scenarios in the :class:`Atmosphere` instance

        TypeError
            if ``return_albedo`` is not a boolean flag
        """

        # Ensure the shape of `mu0`. The other arguments are checked when
        # calling the method `tau_rayleigh`.
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
        tdir = np.exp(-1.0 * tau / mu0)

        # Compute the global and diffuse transmittances.
        c = [2. / 3., 4. / 3.]
        tglb = ((c[0] + mu0) + (c[0] - mu0) * tdir) / (c[1] + tau)
        tdif = tglb - tdir

        out = (tglb, tdir, tdif) + salb
        return out

    def trn_aerosols(self, wvln_um, mu0, return_albedo=False, coupling=False):
        r"""Return the aerosol transmittances.

        The direct transmittance is just computed as:

        .. math::
            T_\text{aer}^\text{dir}(\lambda) =
                \exp(-\tau_\text{aer}(\lambda) / \mu_0),

        while Ambartsumian's formulation is used for the global
        transmittance assuming that the single-scattering albedo
        :math:`\omega_0` is not 1:

        .. math::
            T_\text{aer}^\text{glb}(\lambda) =
            \dfrac{(1 - r_0^2) \times \exp(-K \times \tau_\text{aer}(\lambda) / \mu_0)}
                  {1 - r_0^2 \times \exp(-K \times \tau_\text{aer}(\lambda) / \mu_0)},

        where:

        .. math::
            K = \sqrt{(1 - \omega_0) \times (1 - \omega_0 \times g)},
        .. math::
            r_0 = \dfrac{K - 1 + \omega_0}{K + 1 - \omega_0},

        and the diffuse transmittance is the difference between the
        global and the direct transmittances:

        .. math::
             T_\text{aer}^\text{dif}(\lambda) =
                T_\text{aer}^\text{glb}(\lambda)
                - T_\text{aer}^\text{dir}(\lambda).

        Parameters
        ----------

        wvln_um : array-like
            wavelengths in microns, with shape ``(nwvln,)``

        mu0 : array-like
            cosine of solar zenith angles, with shape ``(nscen,)``

        return_albedo : bool, optional
            if True, return also the aerosol contribution to the
            atmospheric albedo

        coupling : bool, optional
            if True, include Rayleigh-aerosol coupling effect;
            this parameter is intended only for internal use

        Returns
        -------

        tglb : array-like
            aerosol global transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        tdir : array-like
            aerosol direct transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        tdif : array-like
            aerosol diffuse transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        salb : array-like, optional
            aerosol contribution to the atmospheric albedo, with shape
            ``(nscen, nwvln)``, for every scenario and wavelength

        Raises
        ------

        ValueError
            if the input ``wvln_um`` or ``mu0`` have invalid shape

        IndexError
            if the shape of ``mu0`` does not match to the number of
            scenarios in the :class:`Atmosphere` instance

        TypeError
            if ``return_albedo`` or ``coupling`` are not boolean flags
        """

        # Ensure the type of `coupling` and the shape of `mu0`. The other
        # arguments are already checked when calling the method `tau_aerosols`.
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
        r0 = (ak - 1. + w0) / (ak + 1. - w0)

        # Compute direct, global and diffuse transmittances.
        tdir = np.exp(-1.0 * tau / mu0)
        tdir_k = tdir**ak
        tglb = (1. - r0**2) * tdir_k / (1. - (r0 * tdir_k)**2)
        tdif = tglb - tdir

        out = (tglb, tdir, tdif) + salb
        return out

    def trn_mixture(self, wvln_um, mu0, return_albedo=False, coupling=False):
        """Return the transmittances for the Rayleigh-aerosols mixture.

        The method allows to consider these transmittances just as a
        combination of independent processes (Rayleigh and aerosols),
        but also as a process with coupling effects.

        Parameters
        ----------

        wvln_um : array-like
            wavelengths in microns, with shape ``(nwvln,)``

        mu0 : array-like
            cosine of solar zenith angles, with shape ``(nscen,)``

        return_albedo : bool, optional
            if True, return also the atmospheric albedo

        coupling : bool, optional
            if True, include Rayleigh-aerosol coupling effect

        Returns
        -------

        tglb : array-like
            mixture global transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        tdir : array-like
            mixture direct transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        tdif : array-like
            mixture diffuse transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        salb : array-like, optional
            atmospheric albedo, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        Raises
        ------

        ValueError
            if the input ``wvln_um`` or ``mu0`` have invalid shape

        IndexError
            if the shape of ``mu0`` does not match to the number of
            scenarios in the Atmosphere instance

        TypeError
            if ``return_albedo`` or ``coupling`` are not boolean flags
        """

        # Ensure the type of `coupling`. The other arguments are already
        # checked when calling the methods `trn_rayleigh` and `trn_aerosols`.
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
            tglb_ray, tdir_ray, _tdif_ray = out[:3]
            sray = out[3] if return_albedo else ()
            # Compute aerosol transmittances.
            out = self.trn_aerosols(*args)
            tglb_aer, tdir_aer, _tdif_aer = out[:3]
            saer = out[3] if return_albedo else ()
            # Compute mix transmittances without Rayleigh-aerosol coupling.
            tglb = tglb_ray * tglb_aer
            tdir = tdir_ray * tdir_aer
            tdif = tglb - tdir
            salb = (sray + saer,) if return_albedo else ()

        out = (tglb, tdir, tdif) + salb
        return out

    def trn_water(self, wvln, mu0):
        r"""Return the transmittance due to water vapour absorption.

        The transmittance is computed by using the formula:

        .. math::
            T_\text{H2O}(\lambda) =
                \exp(-(k_\text{abs}^\text{H2O}(\lambda)
                       \times L_\text{abs}^\text{H2O} / \mu_0)^a),

        where :math:`k_\text{abs}^\text{H2O}` denotes the water vapour
        absorption coefficients in cm-1, :math:`L_\text{abs}^\text{H2O}`
        is the water vapour absorption path given in cm, :math:`\mu_0`
        is the cosine of the solar zenith angle and :math:`a` is an
        empirical exponent (which depends on the wavelength for the
        water vapour).

        Parameters
        ----------

        wvln : array-like
            wavelengths in nanometers, with shape ``(nwvln,)``

        mu0 : array-like
            cosines of the solar zenith angle, with shape ``(nscen,)``

        Returns
        -------

        trn : array-like
            water vapour transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        Raises
        ------

        ValueError
            if ``wvln`` or ``mu0`` have invalid shapes

        IndexError
            if the shape of ``mu0`` does not match to the number of
            scenarios in the :class:`Atmosphere` instance
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
        r"""Return the transmittance due to ozone absorption.

        The transmittance is computed by using the formula:

        .. math::
            T_\text{O3}(\lambda) =
                \exp(-k_\text{abs}^\text{O3}(\lambda)
                     \times L_\text{abs}^\text{O3} / \mu_0),

        where :math:`k_\text{abs}^\text{O3}` denotes the ozone
        absorption coefficients in cm-1, :math:`L_\text{abs}^\text{O3}`
        is the ozone absorption path given in cm and :math:`\mu_0` is
        the cosine of the solar zenith angle.

        Parameters
        ----------

        wvln : array-like
            wavelengths in nanometers, with shape ``(nwvln,)``

        mu0 : array-like
            cosines of the solar zenith angle, with shape ``(nscen,)``

        Returns
        -------

        trn : array-like
            ozone transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        Raises
        ------

        ValueError
            if the input ``wvln`` does not have a proper shape

        IndexError
            if the shape of ``mu0`` does not match to the number of
            scenarios in the :class:`Atmosphere` instance
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
        ozone_coef = 2.687E19 * ozone_xsec

        # Convert from ozone amount in DU to ozone absorption path in cm.
        ozone_path = (1E-3 * self.o3)[:, None]

        trn = np.exp(-ozone_coef * ozone_path / mu0)
        return trn

    def trn_oxygen(self, wvln, mu0):
        r"""Return the transmittance due to molecular oxygen absorption.

        The transmittance is computed by using the formula:

        .. math::
            T_\text{O2}(\lambda) =
                \exp(-(k_\text{abs}^\text{O2}(\lambda)
                       \times L_\text{abs}^\text{O2} / \mu_0)^a),

        where :math:`k_\text{abs}^\text{O2}` denotes the oxygen
        absorption coefficients in cm-1, :math:`L_\text{abs}^\text{O2}`
        is the oxygen absorption path given in cm, :math:`\mu_0` is the
        cosine of the solar zenith angle and :math:`a` is an empirical
        exponent (equal to 0.5641 for the molecular oxygen).

        Parameters
        ----------

        wvln : array-like
            wavelengths in nanometers, with shape ``(nwvln,)``

        mu0 : array-like
            cosines of the solar zenith angle, with shape ``(nscen,)``

        Returns
        -------

        trn : array-like
            oxygen transmittance, with shape ``(nscen, nwvln)``,
            for every scenario and wavelength

        Raises
        ------

        ValueError
            if ``wvln`` or ``mu0`` have invalid shapes
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
        """Create :class:`Atmosphere` instance from file.

        Parameters
        ----------

        path : str
            location of input file

        Returns
        -------

        atm : Atmosphere
            new :class:`Atmosphere` instance based on the input file

        Raises
        ------

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
