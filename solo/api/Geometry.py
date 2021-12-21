from __future__ import division
from collections import namedtuple
import numpy as np


ATTRS = ["day", "sec", "lat", "lon", "sza", "mu0"]

# Define conversion factors.
DAY_TO_RAD = 2. * np.pi / 365.
HOUR_TO_RAD = np.pi / 12.
HOUR_TO_SEC = 3600.


class Geometry(namedtuple("Geometry", ATTRS)):
    """Class to define the geometric properties of the atmospheric view.

    Every instance allows the access to the following properties:

        ngeo : int
            number of scenarios

        day : array-like, shape (ngeo,)
            Julian day ranged from 1 to 366

        sec : array-like, (ngeo,)
            UTC time in seconds (ranged from 0 to 86399)

        day_angle : array-like, shape (ngeo,)
            angle between the Earth-Sun line on 1st January and the same
            line for the Julian days corresponding to the scenarios,
            ranged from 0 to 2 * np.pi rad

        lat : array-like, shape (ngeo,)
            latitude at the viewing positions in radians, ranged from
            -np.pi / 2 to +np.pi / 2 rad

        lon : array-like, shape (ngeo,)
            longitude at the viewing positions in radians, ranged from
            -np.pi to +np.pi rad

        sza : array-like, shape (ngeo,)
            solar zenith angles in radians, ranged from 0 to +np.pi rad

        mu0 : array-like, shape (ngeo,)
            cosine of the solar zenith angles, ranged from -1 to +1
    """

    def __new__(cls, day, sec=None, lat=None, lon=None, sza=None, mode="deg"):
        """Return a new instance of Geometry.

        Receive:

            day : array-like, (ngeo,)
                Julian day
            sec : array-like, (ngeo,), optional
                UTC time in seconds (default None)
            lat : array-like, (ngeo,), optional
                latitude at the viewing positions (default None)
            lon : array-like, (ngeo,), optional
                longitude at the viewing positions (default None)
            sza : array-like, (ngeo,), optional
                solar zenith angles (default None)
            mode : str, optional
                if 'deg', input angles are provided in degrees; if 'rad',
                input angles are given in radians; otherwise, and error
                is raised (default 'deg'); note that the Geometry
                instance always stores the angles in radians

        Return:

            geo : Geometry
                instance of Geometry based on the input parameters

        Raise:

            AttributeError
                if input arguments have inconsistent or wrong shapes
            ValueError
                if mode is not equal to 'deg' or 'rad' or the input
                arguments are out of range
        """

        # Declare constructor arguments.
        args = [day, sec, lat, lon, sza]

        # Ensure that the input arguments have consistent shapes and sizes.
        set_shapes = set(np.shape(x) for x in args if x is not None)
        if len(set_shapes) > 1:
            raise AttributeError("size mismatch among input arguments")
        set_shapes = list(set_shapes)[0]
        if len(set_shapes) > 1:
            raise AttributeError("input arguments must be 0- or 1-dimensional")

        # Check that mode receives a valid value ('rad' or 'deg').
        if mode.lower() == "rad":
            to_radians = lambda x: x
        elif mode.lower() == "deg":
            to_radians = np.radians
        else:
            raise ValueError("invalid value for 'mode': {}".format(mode))

        # Check that the Julian days are within valid range.
        if np.any(day < 1) or np.any(day > 366):
            raise ValueError("Julian days out of range")
        day = np.atleast_1d(day).astype(int)

        # Check that the UTC seconds are within valid range.
        if sec is not None:
            if np.any(sec < 0) or np.any(sec > 86399):
                raise ValueError("UTC seconds out of range")
            sec = np.atleast_1d(sec).astype(int)

        # Check that the latitudes are within valid range.
        if lat is not None:
            lat = np.atleast_1d(to_radians(lat))
            if np.any(np.abs(lat) > np.pi / 2):
                raise ValueError("latitude values out of range")

        # Check that the longitudes are within valid range.
        if lon is not None:
            lon = np.atleast_1d(to_radians(lon))
            if np.any(np.abs(lon) > np.pi):
                raise ValueError("longitude values out of range")

        # Check that the solar zenith angles are within valid range.
        if sza is not None:
            sza = np.atleast_1d(to_radians(sza))
            if np.any(np.abs(sza - np.pi / 2) > np.pi / 2):
                raise ValueError("solar zenith angle values out of range")
        else:
            args = [cls, day, sec, lat, lon, sza, None]
            geo = super(Geometry, cls).__new__(*args)
            sza = geo.compute_sza()

        # Compute the cosine of the solar zenith angle.
        mu0 = np.cos(sza)

        # Return the new instance.
        args = [cls, day, sec, lat, lon, sza, mu0]
        geo = super(Geometry, cls).__new__(*args)
        return geo

    @property
    def day_angle(self):
        """Return the day angle for every Julian day.

        The day angle is defined as the angle between the Earth-Sun line
        on 1st January and the same line for the Julian days
        corresponding to the scenarios.

        Return:

            day_angle : array-like, shape (ngeo,)
                day angle for every scenario's Julian day
        """

        return (self.day - 1) * DAY_TO_RAD

    @property
    def ngeo(self):
        """Return the number of geometries stored within the instance."""

        shp = np.shape(self.mu0)
        return shp[0] if shp else 1

    def geometric_factor(self):
        """Return the factor used to correct the solar TOA irradiance.

        The solar TOA irradiance E0 is normally provided for a constant
        Earth-Sun distance, but this distance changes as a function of
        the day number, so that

            E(day) = E0 * (r0 / r(day))**2

        where E(day) is the solar TOA irradiance for the given days, E0
        is the solar TOA irradiance for the reference day, r(day) is the
        Sun-Earth distance for the given days, and r0 is the Sun-Earth
        distance for the reference day. This function returns the
        geometric factor (r0 / r(day))**2 when E0 and r0 are defined for
        a Sun-Earth distance of 1 AU.

        Return:

            geo_factor : array-like, shape (ngeo,)
                geometric factor for every scenario
        """

        # Define the coefficients of the Fourier series.
        c = [1.00011, 0.03422, 0.00128, 0.000719, 0.000077]

        day_ang1 = self.day_angle
        day_ang2 = day_ang1 * 2

        geo_factor = c[0] + c[1] * np.cos(day_ang1) + c[2] * np.sin(day_ang1) \
                          + c[3] * np.cos(day_ang2) + c[4] * np.sin(day_ang2)

        return geo_factor

    def declination(self):
        """Return the Sun declination for the current Geometry instance.

        Return:

            dec : array-like, shape (ngeo,)
                Sun declination for every scenario in radians
        """

        # Compute the day of the year in radians.
        ett1 = self.day * DAY_TO_RAD
        ett2 = 2. * ett1
        ett3 = 3. * ett1

        # Define the coefficients of the Fourier series.
        c = [0.006918, -0.399912, 0.070257, -0.006758,
             0.000907, -0.002697, 0.001480]

        # Compute the declination in radians.
        dec = c[0] + c[1] * np.cos(ett1) + c[2] * np.sin(ett1)                \
                   + c[3] * np.cos(ett2) + c[4] * np.sin(ett2)                \
                   + c[5] * np.cos(ett3) + c[6] * np.sin(ett3)

        return dec

    def equation_of_time(self):
        """Return the equation of time for the current Geometry instance.

        The equation of time (ET) computes the difference between the
        true solar time (TST), i.e. the time which tracks the diurnal
        motion of the Sun, and the mean solar time (MST), i.e. the time
        which tracks the motion of a theoretical Sun with noons always
        24 hours apart, so that

            TST(day) = MST(day) + ET(day)

        Return:

            eot : array-like, shape (ngeo,)
                equation of time values for every scenario in radians
        """

        # Compute the day of the year in radians.
        ett1 = self.day * DAY_TO_RAD
        ett2 = 2. * ett1

        # Define the coefficients of the Fourier series.
        c = [0.000075, 0.001868, -0.032077, -0.014615, -0.040849]

        # Compute the equation of time in radians.
        eot = c[0] + c[1] * np.cos(ett1) + c[2] * np.sin(ett1)                \
                   + c[3] * np.cos(ett2) + c[4] * np.sin(ett2)

        return eot

    def compute_sza(self):
        """Return the solar zenith angles for the given instance.

        In case that there is already solar zenith angles, they are just
        returned. If not, they are computed based on the datetime and the
        geolocation (latitude, longitude).

        Return:

            sza : array-like, shape (ngeo,)
                solar zenith angles

        Raise:

            ValueError
                if the latitude, longitude or datetime is missing
        """

        # Ensure that the solar zenith angle is not already computed, or that
        # the geolocation attributes are not missing.
        if self.sza is not None:
            return self.sza
        elif self.sec is None:
            raise ValueError("UTC seconds missing")
        elif self.lat is None:
            raise ValueError("latitude values missing")
        elif self.lon is None:
            raise ValueError("longitude values missing")

        # Compute the mean solar time (MST) as a function of the UTC time and
        # the time shift due to the geographic longitude.
        mst = self.sec / HOUR_TO_SEC + self.lon / HOUR_TO_RAD

        # Compute the true solar time (TST) as a function of the MST and the
        # equation of time (EOT) converted from angle to time units.
        tst = mst + self.equation_of_time() / HOUR_TO_RAD
        hour_angle = (tst - 12.) * HOUR_TO_RAD

        # Compute 'mu0' and the solar zenith angle.
        dec = self.declination()
        mu0 = + np.sin(self.lat) * np.sin(dec)                                \
              + np.cos(self.lat) * np.cos(dec) * np.cos(hour_angle)
        sza = np.arccos(mu0)

        return sza

    @staticmethod
    def from_file(path):
        """Create Geometry instance from file.

        Receive:

            path : str
                location of input file

        Return:

            geo : Geometry
                instance of Geometry based on the input file

        Raise:

            ValueError
                if the input file does not have a valid format
        """

        # Define the possible list of input arguments depending on its number.
        keys = {2: ["day", "sza"], 4: ["day", "sec", "lat", "lon"]}

        # Define the converter from time strings to seconds.
        def timestr2num(txt):
            # Try to split the hours, minutes and seconds from time string.
            try:
                nums = list(map(int, txt.decode().split(":")))
            except ValueError:
                raise ValueError("invalid UTC time format")
            # If there is only one number, it is assumed as seconds.
            if len(nums) == 1:
                return nums[0]
            # If there are 2 or three numbers, they are assumed as
            # [hours, minutes, (seconds)].
            elif len(nums) in [2, 3]:
                return sum(x * y for x, y in zip(nums, [3600, 60, 1]))
            # Any other case is not valid.
            else:
                raise ValueError("invalid UTC time format")

        # Try to open the file assuming that all the values are numbers.
        # Otherwise, raise an error.
        try:
            data = np.atleast_2d(np.loadtxt(path))
            args = data.ravel() if data.shape[0] == 1 else data.T
        # If it does not work, try to parse the second column as a timestring.
        except ValueError:
            try:
                converters = {1: timestr2num}
                data = np.atleast_2d(np.loadtxt(path, converters=converters))
                args = data.ravel() if data.shape[0] == 1 else data.T
            # If it does not work, it may be a single scenario in column form.
            except IndexError:
                data = np.loadtxt(path, dtype=np.bytes_)
                if data.shape == (4,):
                    data = np.atleast_2d(data)
                    args = [int(data[0, 0]), timestr2num(data[0, 1]),
                            float(data[0, 2]), float(data[0, 3])]
                else:
                    raise ValueError("invalid file format")
            # If nothing works, then the file cannot be imported.
            except ValueError:
                raise ValueError("invalid file format")

        # Parse the columns into a possible combination of input arguments,
        # otherwise raise an error.
        try:
            kwargs = {k: v for k, v in zip(keys[data.shape[1]], args)}
        except KeyError:
            raise ValueError("invalid file format")

        return Geometry(**kwargs)

