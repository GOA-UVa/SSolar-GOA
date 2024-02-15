#! /usr/bin/env python
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
"""Script to create example data for the tests."""

import os
import numpy as np
import solo
from solo.api import Atmosphere
from solo.api import Geometry
from solo import radtran

here = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":

    # Define test wavelengths.
    libroot = os.path.abspath(os.path.dirname(solo.__file__))
    kurucz_path = os.path.join(libroot, "dat", "kurucz.dat")
    wvln, irr0 = np.loadtxt(kurucz_path).T
    wvln_um = 0.001 * wvln

    # Define test geometry and atmosphere.
    geo0 = Geometry(lat=28.31, lon=-16.50, sza=60, day=152)
    atm0 = Atmosphere(p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05)

    # Compute optical thicknesses.
    tau_ray = atm0.tau_rayleigh(wvln_um)
    tau_aer = atm0.tau_aerosols(wvln_um)

    # Compute gas direct transmittances.
    trn1 = atm0.trn_water(wvln, geo0.mu0)
    trn2 = atm0.trn_ozone(wvln, geo0.mu0)
    trn3 = atm0.trn_oxygen(wvln, geo0.mu0)
    tdir_gas = trn1 * trn2 * trn3

    # Compute mixture transmittances with and without coupling.
    tglb_mix, tdir_mix, tdif_mix, salb_mix = atm0.trn_mixture(
        wvln_um, geo0.mu0, coupling=False, return_albedo=True)
    tglb_mix_coupled, tdir_mix_coupled, tdif_mix_coupled, salb_mix_coupled = atm0.trn_mixture(
        wvln_um, geo0.mu0, coupling=True, return_albedo=True)

    # Save results to table.
    table = np.vstack((
        wvln[None], tau_ray, tau_aer, tdir_gas,
        tdir_mix, tglb_mix, tdif_mix, salb_mix,
        tdir_mix_coupled, tglb_mix_coupled,
        tdif_mix_coupled, salb_mix_coupled)).T

    ncols = table.shape[-1]
    topfmt = "".join(["{0:>4s}"] + list(map("{%s:>20s}".__mod__, range(1, ncols))))
    txtfmt = "{0}{1}".format("%6.0f", (ncols - 1) * "%20.12E")
    header = topfmt.format(
        "wvln", "tau_ray", "tau_aer", "tdir_gas",
        "tdir_mix", "tglb_mix", "tdif_mix", "salb_mix",
        "tdir_mix_coupled", "tglb_mix_coupled",
        "tdif_mix_coupled", "salb_mix_coupled")
    opath = os.path.join(here, "transmittance.dat")
    np.savetxt(opath, table, fmt=txtfmt, header=header)

    # Compute irradiances.
    irr0_nday = irr0 * geo0.geometric_factor()
    irr_glb, irr_dir, irr_dif = radtran(geo0, atm0, coupling=True)

    # Save results to table.
    table = np.vstack((
        wvln[None], irr0_nday, irr_glb,
        irr_dir, irr_dir * geo0.mu0, irr_dif)).T

    ncols = table.shape[-1]
    topfmt = "".join(["{0:>4s}"] + list(map("{%s:>20s}".__mod__, range(1, ncols))))
    txtfmt = "{0}{1}".format("%6.0f", (ncols - 1) * "%20.12E")
    header = topfmt.format(
        "wvln", "irr0 * (r0/r)^2", "irr_glb",
        "irr_dir", "irr_dir * cos(SZA)", "irr_dif")
    opath = os.path.join(here, "irradiance.dat")
    np.savetxt(opath, table, fmt=txtfmt, header=header)

    # Integrate irradiances.
    irr0_nday_total = np.trapz(irr0_nday, wvln)
    irr_glb_total = np.trapz(irr_glb, wvln)
    irr_dir_total = np.trapz(irr_dir, wvln)
    irr_dif_total = np.trapz(irr_dif, wvln)

    # Save results to table.
    table = np.vstack((
        geo0.day, np.degrees(geo0.sza), irr0_nday_total, irr_glb_total,
        irr_dir_total, irr_dir_total * geo0.mu0, irr_dif_total)).T

    ncols = table.shape[-1]
    topfmt = "".join(["{0:>6s}", "{1:>8s}"] + list(map("{%s:>20s}".__mod__, range(2, ncols))))
    txtfmt = "{0}{1}".format("%8.0f%8.2f", (ncols - 2) * "%20.6f")
    header = topfmt.format(
        "day", "sza", "irr0 * (r0/r)^2", "irr_glb",
        "irr_dir", "irr_dir * cos(SZA)", "irr_dif")
    opath = os.path.join(here, "integrated.dat")
    np.savetxt(opath, table, fmt=txtfmt, header=header)
