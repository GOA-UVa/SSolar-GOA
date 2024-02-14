import os.path
import unittest
import numpy as np
from solo.api import Geometry
from solo.test import SoloTest


UNITTEST_FOLDER = os.path.dirname(__file__)
GEOMETRY_FOLDER = os.path.join(UNITTEST_FOLDER, "obj", "geo")


class GeoTest(SoloTest):

    def checkGeoEqual(self, geo1, geo2):

        self.assertTrue(np.allclose(geo1.day, geo2.day))
        if geo1.sec is None:
            self.assertTrue(geo1.sec is geo2.sec)
        else:
            self.assertTrue(np.allclose(geo1.sec, geo2.sec))
        if geo1.lat is None:
            self.assertTrue(geo1.lat is geo2.lat)
        else:
            self.assertTrue(np.allclose(geo1.lat, geo2.lat))
        if geo1.lon is None:
            self.assertTrue(geo1.lon is geo2.lon)
        else:
            self.assertTrue(np.allclose(geo1.lon, geo2.lon))
        self.assertTrue(np.allclose(geo1.sza, geo2.sza))
        self.assertTrue(np.allclose(geo1.mu0, geo2.mu0))

    def testGeo11(self):

        path = os.path.join(GEOMETRY_FOLDER, "geo11.dat")
        geo1 = Geometry.from_file(path)
        geo2 = Geometry(
            day=152, sec=None, lat=None, lon=None, sza=60, mode="deg")
        self.checkGeoEqual(geo1, geo2)

    def testGeo12(self):

        path = os.path.join(GEOMETRY_FOLDER, "geo12.dat")
        geo1 = Geometry.from_file(path)
        geo2 = Geometry(
            day=152, sec=25311, lat=0.49410271, lon=-0.28797933,
            sza=1.39777933, mode="rad")
        self.checkGeoEqual(geo1, geo2)

    def testGeo13(self):

        path = os.path.join(GEOMETRY_FOLDER, "geo13.dat")
        geo1 = Geometry.from_file(path)
        geo2 = Geometry(
            day=152, sec=43510, lat=0.49410271, lon=-0.28797933,
            sza=0.2546518, mode="rad")
        self.checkGeoEqual(geo1, geo2)

    def testGeo21(self):

        path = os.path.join(GEOMETRY_FOLDER, "geo21.dat")
        geo1 = Geometry.from_file(path)
        geo2 = Geometry(
            day=np.array([152, 152, 152, 152, 153]), sec=None, lat=None,
            lon=None, sza=np.array([60, 50.4, 15.1, 21, 75.]), mode="deg")
        self.checkGeoEqual(geo1, geo2)

    def testGeo22(self):

        path = os.path.join(GEOMETRY_FOLDER, "geo22.dat")
        geo1 = Geometry.from_file(path)
        geo2 = Geometry(
            day=np.array([152, 180, 235]),
            sec=np.array([25311, 5678, 47162]),
            lat=np.array([0.49410271, 0.83950337, 0.00872665]),
            lon=np.array([-0.28797933, 1.31772359, 0.6981317]),
            sza=np.array([1.39777933, 1.17809272, 0.98533964]),
            mode="rad")
        self.checkGeoEqual(geo1, geo2)

    def testGeo23(self):

        path = os.path.join(GEOMETRY_FOLDER, "geo23.dat")
        geo1 = Geometry.from_file(path)
        geo2 = Geometry(
            day=np.array([152, 180, 235]),
            sec=np.array([43510, 47175, 50820]),
            lat=np.array([0.49410271, 0.83950337, 0.00872665]),
            lon=np.array([-0.28797933, 1.31772359, 0.6981317]),
            sza=np.array([0.2546518, 1.28671359, 1.24504354]),
            mode="rad")
        self.checkGeoEqual(geo1, geo2)


if __name__ == "__main__":
    unittest.main()

