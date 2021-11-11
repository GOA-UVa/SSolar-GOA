
from __future__ import print_function
from __future__ import division
import os.path
import numpy as np


def radtran(geo, atm, toa_file, wvln_th, coupling=True):
    """Return the BOA irradiances based on an atmosphere and geometry.

    Receive:

        geo : Geometry
            an instance of Geometry containing the relevant information
            of the geometric parameters
        atm : Atmosphere
            an instance of Atmosphere containing the relevant information
            of the atmospheric components
        wvln_th : array-like, shape (nwvln,), optional
            wavelengths in nanometers (default None, which means that
            the wavelengths are taken from the same file where the TOA
            irradiance is stored)
        toa_file : str
            TOA spectral irradiance file to use
        coupling : bool, optional
            if True, include Rayleigh-aerosol coupling effect
            (default True)

    Return:

        irr_glb : array-like, shape (nscen, nwvln)
            BOA global irradiance for every scenario and wavelength
        irr_dir : array-like, shape (nscen, nwvln)
            BOA direct irradiance for every scenario and wavelength
        irr_dif : array-like, shape (nscen, nwvln)
            BOA diffuse irradiance for every scenario and wavelength

    Raise:

        ValueError
            if 'wvln' has an invalid shape
        TypeError
            if 'coupling' is not a boolean flag
    """

    file_toa = str(toa_file).lower()

    # Read the TOA irradiance as a function of the wavelength.
    path = os.path.join(os.path.dirname(__file__), "dat", file_toa + ".dat")
    wv, ir = np.loadtxt(path).T

    wvln0 = []
    irr0 = []
    for w, i in zip(wv, ir):
        if (w >= wvln_th[0]) & (w <= wvln_th[1]):
            wvln0.append(w)
            irr0.append(i)

    # Convert wavelengths from nanometers to microns and adjust the TOA
    # irradiance to the actual Sun-Earth distance.
    wvln_um = 1E-3 * np.asarray(wvln0, dtype=np.float64)
    irr0 = np.asarray(irr0, dtype=np.float64) * geo.geometric_factor()[:, None]

    # Compute the transmittance due to Rayleigh and aerosols.
    args = [wvln_um, geo.mu0, True, coupling]
    tglb_mix, tdir_mix, tdif_mix, atm_alb = atm.trn_mixture(*args)

    # Compute the transmittance due to gas absorption.
    args = [wvln0, geo.mu0]
    tdir_wat = atm.trn_water(*args)
    tdir_ozo = atm.trn_ozone(*args)
    tdir_oxy = atm.trn_oxygen(*args)
    tdir_gas = tdir_wat * tdir_ozo * tdir_oxy

    # Compute the amplification factor for the BOA global irradiance.
    amp_factor = 1. / (1. - atm.rho[:, None] * atm_alb)

    # Compute the BOA global, direct and diffuse irradiances.
    mu0 = geo.mu0[:, None]
    irr_glb = irr0 * mu0 * tglb_mix * tdir_gas * amp_factor
    irr_dir = irr0 * tdir_mix * tdir_gas
    irr_dif = irr_glb - irr_dir * mu0

    # If requested, squeeze the length-1 axes from the output arrays.
    out = (irr_glb, irr_dir, irr_dif, wvln0)

    return out


if __name__ == "__main__":

    import sys
    import getopt
    from solo.api import Geometry
    from solo.api import Atmosphere

    # Read arguments and options.
    optkeys = ["geo=", "atm=", "out=", "no-coupling"]
    optlist, args = getopt.getopt(sys.argv[1:], "", optkeys)

    # Parse --geo option.
    geo = [x[1] for x in optlist if x[0] == "--geo"]
    if len(geo) == 0:
        print("Error: missing --geo option")
        sys.exit(1)
    elif len(geo) != 1:
        print("Error: multiple --geo options")
        sys.exit(1)
    else:
        try:
            geo = Geometry.from_file(geo[0])
        except Exception as err:
            print("{}\nError: wrong Geometry input file".format(err))
            sys.exit(1)

    # Parse --atm option.
    atm = [x[1] for x in optlist if x[0] == "--atm"]
    if len(atm) == 0:
        print("Error: missing --atm option")
        sys.exit(2)
    elif len(atm) != 1:
        print("Error: multiple --atm options")
        sys.exit(2)
    else:
        try:
            atm = Atmosphere.from_file(atm[0])
        except Exception as err:
            print("{}\nError: wrong Atmosphere input file".format(err))
            sys.exit(2)

    # Parse --out option.
    out = [x[1] for x in optlist if x[0] == "--out"]
    if len(out) == 0:
        print("Error: missing --out option")
        sys.exit(3)
    elif len(out) != 1:
        print("Error: multiple --out options")
        sys.exit(3)
    else:
        out = str(out[0])
        out_cut = os.path.splitext(out)
        out_glb = "".join([out_cut[0], "_glb", out_cut[1]])
        out_dir = "".join([out_cut[0], "_dir", out_cut[1]])
        out_dif = "".join([out_cut[0], "_dif", out_cut[1]])

    # Parse --no-coupling option.
    coupling = [x[1] for x in optlist if x[0] == "--no-coupling"]
    coupling = not bool(coupling)

    # Run the radiative transfer solver.
    irr_glb, irr_dir, irr_dif = radtran(geo, atm, None, coupling)

    # Export the results into text files.
    np.savetxt(out_glb, irr_glb.T, fmt="%+14.6E")
    np.savetxt(out_dir, irr_dir.T, fmt="%+14.6E")
    np.savetxt(out_dif, irr_dif.T, fmt="%+14.6E")

