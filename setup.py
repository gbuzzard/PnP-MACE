#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [ ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Gregery T. Buzzard",
    author_email='buzzard@purdue.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Demonstrations of the PnP algorithm and MACE framework on simple image reconstruction problems. ",
    entry_points={
        'console_scripts': [
            'pnp_mace=pnp_mace.cli:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n',
    include_package_data=True,
    keywords='pnp_mace',
    name='pnp_mace',
    packages=find_packages(include=['pnp_mace', 'pnp_mace.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/gbuzzard/pnp_mace',
    version='0.1.0',
    zip_safe=False,
)
