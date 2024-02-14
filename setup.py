#! /usr/bin/env python3

from setuptools import setup
from setuptools import find_packages
from distutils.cmd import Command
install_requires = {"install_requires": "numpy"}


class UnittestCommand(Command):
    """Custom command to run the unit tests from the library."""

    description = "run unit tests from the library"
    user_options = []

    def initialize_options(self):
        import sys
        import os.path

        name = "__init__.py"
        fold = os.path.join(os.path.dirname(__file__), "solo", "test")
        path = os.path.join(fold, name)

        version = "{}".format(3 if sys.hexversion >= 0x03000000 else "")
        python_exec = "python{}".format(version)
        self.cmdlist = [python_exec, path]

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        subprocess.check_call(self.cmdlist)


kwargs = {
    "name":
        "solo",
    "version":
        "1.0.2",
    "license":
        "GNU General Public License v2 or later (GPLv2+)",
    "description":
        "Python implementation of UVa-GOA radiative transfer model",
    "author":
        "Victor Molina Garcia",
    "author_email":
        "victor@goa.uva.es",
    "maintainer":
        "Victor Molina Garcia",
    "maintainer_email":
        "victor@goa.uva.es",
    "url":
        "https://bitbucket.org/molinav/solo",
    "package_dir":
        {"": "src"},
    "packages":
        find_packages(where="src"),
    "package_data": {
        "solo": [
            "dat/*.dat",
        ],
    },
    "cmdclass": {
        "test": UnittestCommand,
    },
}

kwargs.update(install_requires)
setup(**kwargs)

