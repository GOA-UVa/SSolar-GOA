import unittest
import numpy as np
from solo.test import SoloTest


class TrnGasTest(SoloTest):

    def calcObj1(self, geo, atm, wvln):
        args = [wvln, geo.mu0]
        trn1 = atm.trn_water(*args)
        trn2 = atm.trn_ozone(*args)
        trn3 = atm.trn_oxygen(*args)
        return trn1 * trn2 * trn3

    def testTrnGas_Geo0D_Atm0D_Val0D(self):
        obj0 = self.result["tdir_gas"]
        shp1 = 2 * self.one()
        obj1 = self.calcObj1(self.geo0, self.atm0, self.wvln[0])
        flag = np.allclose(obj1, obj0[0], self.delta)
        self.assertTupleEqual(obj1.shape, shp1)
        self.assertTrue(flag)

    def testTrnGas_Geo0D_Atm0D_Val1D(self):
        shp0 = (self.wvln.size,)
        obj0 = self.result["tdir_gas"]
        shp1 = self.one() + shp0
        obj1 = self.calcObj1(self.geo0, self.atm0, self.wvln)
        flag = np.allclose(obj1, obj0, self.delta)
        self.assertTupleEqual(obj1.shape, shp1)
        self.assertTrue(flag)

    def testTrnGas_Geo0D_Atm1D_Val0D(self):
        shp0 = (self.atm1.nscen,)
        obj0 = self.result["tdir_gas"]
        shp1 = shp0 + self.one()
        obj1 = self.calcObj1(self.geo0, self.atm1, self.wvln[0])
        flag = np.allclose(obj1[0], obj0[0], self.delta)
        self.assertTupleEqual(obj1.shape, shp1)
        self.assertTrue(flag)

    def testTrnGas_Geo0D_Atm1D_Val1D(self):
        shp0 = (self.atm1.nscen, self.wvln.size)
        obj0 = self.result["tdir_gas"]
        shp1 = shp0
        obj1 = self.calcObj1(self.geo0, self.atm1, self.wvln)
        flag = np.allclose(obj1[0, :], obj0, self.delta)
        self.assertTupleEqual(obj1.shape, shp1)
        self.assertTrue(flag)

    def testTrnGas_Geo1D_Atm0D_Val0D(self):
        shp0 = (self.geo1.ngeo,)
        obj0 = self.result["tdir_gas"]
        shp1 = shp0 + self.one()
        obj1 = self.calcObj1(self.geo1, self.atm0, self.wvln[0])
        flag = np.allclose(obj1[0], obj0[0], self.delta)
        self.assertTupleEqual(obj1.shape, shp1)
        self.assertTrue(flag)

    def testTrnGas_Geo1D_Atm0D_Val1D(self):
        shp0 = (self.geo1.ngeo, self.wvln.size)
        obj0 = self.result["tdir_gas"]
        shp1 = shp0
        obj1 = self.calcObj1(self.geo1, self.atm0, self.wvln)
        flag = np.allclose(obj1[0], obj0, self.delta)
        self.assertTupleEqual(obj1.shape, shp1)
        self.assertTrue(flag)

    def testTrnGas_Geo1D_Atm1D_Val0D(self):
        shp0 = (self.geo1.ngeo,)
        obj0 = self.result["tdir_gas"]
        shp1 = shp0 + self.one()
        obj1 = self.calcObj1(self.geo1, self.atm1, self.wvln[0])
        args = [self.wvln[0], self.geo1.mu0]
        flag = np.allclose(obj1[0, 0], obj0[0], self.delta)
        self.assertTupleEqual(obj1.shape, shp1)
        self.assertTrue(flag)

    def testTrnGas_Geo1D_Atm1D_Val1D(self):
        shp0 = (self.geo1.ngeo, self.wvln.size)
        obj0 = self.result["tdir_gas"]
        shp1 = shp0
        obj1 = self.calcObj1(self.geo1, self.atm1, self.wvln)
        flag = np.allclose(obj1[0], obj0, self.delta)
        self.assertTupleEqual(obj1.shape, shp1)
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()

