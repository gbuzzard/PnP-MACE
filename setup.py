#!/usr/bin/env python

"""The setup script."""

import os
from ast import parse
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

name = 'pnp_mace'

# See http://stackoverflow.com/questions/2058802
with open(os.path.join(name, '__init__.py')) as f:
    version = parse(next(filter(
        lambda line: line.startswith('__version__'),
        f))).body[0].value.s


requirements = ['numpy', 'matplotlib', 'requests', 'dotmap', 'Pillow', 'bm3d']

setup_requirements = []

test_requirements = ['pytest', 'pytest-runner']

setup(
    author="Gregery T. Buzzard",
    author_email='buzzard@purdue.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Demonstrations of the PnP algorithm and MACE framework on simple image reconstruction problems. ",
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n',
    include_package_data=True,
    keywords='pnp_mace',
    name=name,
    packages=find_packages(include=['pnp_mace', 'pnp_mace.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/gbuzzard/PnP-MACE',
    version=version,
    zip_safe=False,
)
