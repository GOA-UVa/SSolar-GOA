import unittest
import numpy as np
from solo.test import SoloTest


class TauTest(SoloTest):

    def testTauRayleigh_Atm0D_Val0D(self):
        shp0 = ()
        obj0 = self.result["tau_ray"]
        for squeeze in (False, True):
            shp1 = 2 * self.one(squeeze)
            obj1 = self.atm0.tau_rayleigh(self.wvln_um[0], squeeze)
            flag = np.allclose(obj1, obj0[0], self.delta)
            self.assertTupleEqual(obj1.shape, shp1)
            self.assertTrue(flag)

    def testTauRayleigh_Atm1D_Val0D(self):
        shp0 = (self.atm1.nscen,)
        obj0 = self.result["tau_ray"]
        for squeeze in (False, True):
            shp1 = shp0 + self.one(squeeze)
            obj1 = self.atm1.tau_rayleigh(self.wvln_um[0], squeeze)
            flag = np.allclose(obj1[0], obj0[0], self.delta)
            self.assertTupleEqual(obj1.shape, shp1)
            self.assertTrue(flag)

    def testTauRayleigh_Atm0D_Val1D(self):
        shp0 = (self.wvln_um.size,)
        obj0 = self.result["tau_ray"]
        for squeeze in (False, True):
            shp1 = self.one(squeeze) + shp0
            obj1 = self.atm0.tau_rayleigh(self.wvln_um, squeeze)
            flag = np.allclose(obj1, obj0, self.delta)
            self.assertTupleEqual(obj1.shape, shp1)
            self.assertTrue(flag)

    def testTauRayleigh_Atm1D_Val1D(self):
        shp0 = (self.atm1.nscen, self.wvln_um.size,)
        obj0 = self.result["tau_ray"]
        for squeeze in (False, True):
            shp1 = shp0
            obj1 = self.atm1.tau_rayleigh(self.wvln_um, squeeze)
            flag = np.allclose(obj1[0], obj0, self.delta)
            self.assertTupleEqual(obj1.shape, shp1)
            self.assertTrue(flag)

    def testTauAerosols_Atm0D_Val0D(self):
        shp0 = ()
        obj0 = self.result["tau_aer"]
        for squeeze in (False, True):
            shp1 = 2 * self.one(squeeze)
            obj1 = self.atm0.tau_aerosols(self.wvln_um[0], squeeze)
            flag = np.allclose(obj1, obj0[0], self.delta)
            self.assertTupleEqual(obj1.shape, shp1)
            self.assertTrue(flag)

    def testTauAerosols_Atm1D_Val0D(self):
        shp0 = (self.atm1.nscen,)
        obj0 = self.result["tau_aer"]
        for squeeze in (False, True):
            shp1 = shp0 + self.one(squeeze)
            obj1 = self.atm1.tau_aerosols(self.wvln_um[0], squeeze)
            flag = np.allclose(obj1[0], obj0[0], self.delta)
            self.assertTupleEqual(obj1.shape, shp1)
            self.assertTrue(flag)

    def testTauAerosols_Atm0D_Val1D(self):
        shp0 = (self.wvln_um.size,)
        obj0 = self.result["tau_aer"]
        for squeeze in (False, True):
            shp1 = self.one(squeeze) + shp0
            obj1 = self.atm0.tau_aerosols(self.wvln_um, squeeze)
            flag = np.allclose(obj1, obj0, self.delta)
            self.assertTupleEqual(obj1.shape, shp1)
            self.assertTrue(flag)

    def testTauAerosols_Atm1D_Val1D(self):
        shp0 = (self.atm1.nscen, self.wvln_um.size,)
        obj0 = self.result["tau_aer"]
        for squeeze in (False, True):
            shp1 = shp0
            obj1 = self.atm1.tau_aerosols(self.wvln_um, squeeze)
            flag = np.allclose(obj1[0], obj0, self.delta)
            self.assertTupleEqual(obj1.shape, shp1)
            self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()

