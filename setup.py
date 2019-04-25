#!/usr/bin/env python

from distutils.core import setup
from setuptools import find_packages

setup(
    name='dense_fusion',
    version='0.1dev',
    author='Chen Wang',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description='Estimate poses of objects',
    #long_description=open('README.md').read(),
    #package_data = {'': []},
)

