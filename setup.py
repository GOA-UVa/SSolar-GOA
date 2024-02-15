#! /usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa: E122
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
"""solo -- Core of the radiative transfer library SSolar-GOA."""

import io
import os
import re
import sys
from setuptools import setup
from setuptools import find_packages


def get_content(name, splitlines=False):
    """Return the file contents with project root as root folder."""

    here = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(here, name)
    with io.open(path, "r", encoding="utf-8") as fd:
        content = fd.read()
    if splitlines:
        content = [row for row in content.splitlines() if row]
    return content


def get_version(pkgname):
    """Return package version without importing the file."""

    here = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(*[here, "src"] + pkgname.split(".") + ["__init__.py"])
    with io.open(path, "r", encoding="utf-8") as fd:
        pattern = r"""\n__version__[ ]*=[ ]*["']([^"]+)["']"""
        return re.search(pattern, fd.read()).group(1)


install_requires = get_content("requirements.txt", splitlines=True)
if sys.version_info[:2] == (3, 2):
    # Hack for Python 3.2 because pip < 8 cannot handle version markers.
    marker = '; python_version == "3.2"'
    install_requires = [
        item.replace(marker, "") for item in install_requires
        if item.endswith(marker)]

setup(**{
    "name":
        "solo",
    "version":
        get_version("solo"),
    "license":
        "GNU General Public License v2 or later (GPLv2+)",
    "description":
        "Core of the radiative transfer library SSolar-GOA",
    "long_description":
        get_content("README.md"),
    "long_description_content_type":
        "text/markdown",
    "url":
        "https://github.com/molinav/solo",
    "author":
        "Víctor Molina García",
    "author_email":
        "victor@goa.uva.es",
    "maintainer":
        "Víctor Molina García",
    "maintainer_email":
        "victor@goa.uva.es",
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    "keywords": [
        "atmospheric physics",
        "radiative transfer",
    ],
    "package_dir":
        {"": "src"},
    "packages":
        find_packages(where="src"),
    "package_data": {
        "solo": [
            "dat/*.dat",
        ],
    },
    "python_requires":
        ", ".join([
            ">=2.6",
            "!=3.0.*",
            "!=3.1.*",
            "<3.13",
        ]),
    "install_requires":
        install_requires,
    "extras_require": {
        "doc":
            get_content("requirements-doc.txt", splitlines=True),
        "lint":
            get_content("requirements-lint.txt", splitlines=True),
        "test":
            get_content("requirements-test.txt", splitlines=True),
    },
    "project_urls": {
        "Bug Tracker":
            "https://github.com/molinav/solo/issues",
        "Documentation":
            "https://github.com/molinav/solo/wiki",
        "Source":
            "https://github.com/molinav/solo",
    },
})
