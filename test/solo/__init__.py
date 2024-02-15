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
"""Template class for :mod:`solo` test cases."""

import os.path
try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
from solo.api import Geometry
from solo.api import Atmosphere


class TestSolo(unittest.TestCase):
    """Template class for :mod:`solo` test cases."""

    @staticmethod
    def one():
        """Return the one-tuple."""

        return (1,)

    def setUp(self):
        """Set up the attributes needed for the test."""

        self.delta = 5E-4

        # Read the file with reference data.
        here = os.path.dirname(__file__)
        path = os.path.join(here, "dat", "transmittance.dat")
        data = np.loadtxt(path).T

        # Extract the set of wavelengths in nanometers and microns.
        self.wvln = data[0]
        self.wvln_um = 0.001 * self.wvln

        # Create the instances of Geometry and Atmosphere.
        self.geo0 = Geometry(
            lat=28.31, lon=-16.50, sza=60, day=152)
        self.atm0 = Atmosphere(
            p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05)

        # Create vectorised instances of Geometry and Atmosphere.
        self.geo1 = Geometry(
            lat=np.asarray([np.degrees(self.geo0.lat[0]), 35.45, 40.13]),
            lon=np.asarray([np.degrees(self.geo0.lon[0]), 25.80, -9.51]),
            sza=np.asarray([np.degrees(self.geo0.sza[0]), 15.50, 30.50]),
            day=np.asarray([self.geo0.day[0], 12, 250]))
        self.atm1 = Atmosphere(
            p=np.asarray([self.atm0.p[0], 875.4, 925.3]),
            rho=np.asarray([self.atm0.rho[0], 0.35, 0.7]),
            o3=np.asarray([self.atm0.o3[0], 286., 310]),
            h2o=np.asarray([self.atm0.h2o[0], 0.15, 0.01]),
            alpha=np.asarray([self.atm0.alpha[0], 0.75, 0.9]),
            beta=np.asarray([self.atm0.beta[0], 0.10, 0.15]))

        # Store the results corresponding to the created instances.
        self.result = {
            "tau_ray": data[1],
            "tau_aer": data[2],
            "tdir_gas": data[3],
            "tdir_mix": data[4],
            "tglb_mix": data[5],
            "tdif_mix": data[6],
            "salb_mix": data[7],
            "tdir_mix_coupled": data[8],
            "tglb_mix_coupled": data[9],
            "tdif_mix_coupled": data[10],
            "salb_mix_coupled": data[11],
        }

    def tearDown(self):
        """Clean the attributes needed for the tests."""

        self.wvln = None
        self.wvln_um = None
        self.geo0 = None
        self.atm0 = None
        self.geo1 = None
        self.atm1 = None
        self.result = None
