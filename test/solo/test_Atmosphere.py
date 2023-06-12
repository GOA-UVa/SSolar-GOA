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

import os.path
import unittest
import numpy as np
from solo.api import Atmosphere
from . import TestSolo


UNITTEST_FOLDER = os.path.dirname(__file__)
ATMOSPHERE_FOLDER = os.path.join(UNITTEST_FOLDER, "obj", "atm")


class TestAtmosphere(TestSolo):

    def checkAtmEqual(self, atm1, atm2):

        self.assertTrue(np.allclose(atm1.p, atm2.p))
        self.assertTrue(np.allclose(atm1.rho, atm2.rho))
        self.assertTrue(np.allclose(atm1.o3, atm2.o3))
        self.assertTrue(np.allclose(atm1.h2o, atm2.h2o))
        self.assertTrue(np.allclose(atm1.alpha, atm2.alpha))
        self.assertTrue(np.allclose(atm1.beta, atm2.beta))
        self.assertTrue(np.allclose(atm1.w0, atm2.w0))
        self.assertTrue(np.allclose(atm1.g, atm2.g))

    def testAtm11(self):

        path = os.path.join(ATMOSPHERE_FOLDER, "atm11.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05,
            w0=0.9, g=0.85)
        self.checkAtmEqual(atm1, atm2)

    def testAtm12(self):

        path = os.path.join(ATMOSPHERE_FOLDER, "atm12.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05)
        self.checkAtmEqual(atm1, atm2)

    def testAtm21(self):

        path = os.path.join(ATMOSPHERE_FOLDER, "atm21.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05,
            w0=0.85, g=0.95)
        self.checkAtmEqual(atm1, atm2)

    def testAtm22(self):

        path = os.path.join(ATMOSPHERE_FOLDER, "atm22.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=800, rho=0.2, o3=300, h2o=0.4, alpha=1.5, beta=0.05)
        self.checkAtmEqual(atm1, atm2)

    def testAtm31(self):

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
        self.checkAtmEqual(atm1, atm2)

    def testAtm32(self):

        path = os.path.join(ATMOSPHERE_FOLDER, "atm32.dat")
        atm1 = Atmosphere.from_file(path)
        atm2 = Atmosphere(
            p=np.array([800, 875, 880]),
            rho=np.array([0.2, 0.3, 0.25]),
            o3=np.array([300, 320, 290]),
            h2o=np.array([0.4, 0.5, 0.4]),
            alpha=np.array([1.5, 1.4, 1.1]),
            beta=np.array([0.05, 0.01, 0.02]))
        self.checkAtmEqual(atm1, atm2)


if __name__ == "__main__":
    unittest.main()
