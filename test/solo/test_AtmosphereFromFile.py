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
"""Basic tests for the :class:`Atmosphere` class."""

import os.path
try:
    import unittest2 as unittest
except ImportError:
    import unittest

import numpy as np
from solo.api import Atmosphere
from . import TestSolo


UNITTEST_FOLDER = os.path.dirname(__file__)
ATMOSPHERE_FOLDER = os.path.join(UNITTEST_FOLDER, "obj", "atm")


class TestAtmosphere(TestSolo):
    """Basic tests for the :class:`Atmosphere` class."""

    def check_atm_equal(self, atm1, atm2):
        """Generic equal check for :class:`Atmosphere` classes."""

        self.assertTrue(np.allclose(atm1.p, atm2.p))
        self.assertTrue(np.allclose(atm1.rho, atm2.rho))
        self.assertTrue(np.allclose(atm1.o3, atm2.o3))
        self.assertTrue(np.allclose(atm1.h2o, atm2.h2o))
        self.assertTrue(np.allclose(atm1.alpha, atm2.alpha))
        self.assertTrue(np.allclose(atm1.beta, atm2.beta))
        self.assertTrue(np.allclose(atm1.w0, atm2.w0))
        self.assertTrue(np.allclose(atm1.g, atm2.g))

    def test_atm11(self):
        """Test loading of `atm11.dat` from file."""

        path = os.path.join(ATMOSPHERE_FOLDER, "atm11.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05,
            w0=0.9, g=0.85)
        self.check_atm_equal(atm1, atm2)

    def test_atm12(self):
        """Test loading of `atm12.dat` from file."""

        path = os.path.join(ATMOSPHERE_FOLDER, "atm12.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05)
        self.check_atm_equal(atm1, atm2)

    def test_atm21(self):
        """Test loading of `atm21.dat` from file."""

        path = os.path.join(ATMOSPHERE_FOLDER, "atm21.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05,
            w0=0.85, g=0.95)
        self.check_atm_equal(atm1, atm2)

    def test_atm22(self):
        """Test loading of `atm22.dat` from file."""

        path = os.path.join(ATMOSPHERE_FOLDER, "atm22.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05)
        self.check_atm_equal(atm1, atm2)

    def test_atm31(self):
        """Test loading of `atm31.dat` from file."""

        path = os.path.join(ATMOSPHERE_FOLDER, "atm31.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=np.array([800, 875, 880]),
            rho=np.array([0.2, 0.3, 0.25]),
            o3=np.array([300, 320, 290]),
            h2o=np.array([0.4, 0.5, 0.4]),
            alpha=np.array([1.5, 1.4, 1.1]),
            beta=np.array([0.05, 0.01, 0.02]),
            w0=np.array([0.85, 0.8, 0.85]),
            g=np.array([0.95, 0.92, 0.93]))
        self.check_atm_equal(atm1, atm2)

    def test_atm32(self):
        """Test loading of `atm32.dat` from file."""

        path = os.path.join(ATMOSPHERE_FOLDER, "atm32.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=np.array([800, 875, 880]),
            rho=np.array([0.2, 0.3, 0.25]),
            o3=np.array([300, 320, 290]),
            h2o=np.array([0.4, 0.5, 0.4]),
            alpha=np.array([1.5, 1.4, 1.1]),
            beta=np.array([0.05, 0.01, 0.02]))
        self.check_atm_equal(atm1, atm2)


if __name__ == "__main__":
    unittest.main()
