from distutils.core import setup

from setuptools import setup, find_packages
from codecs import open
from os import path
import subprocess
import re
import os

def package_files(directory='data'):
    """
    Generate a list of non-code files to be included in the package.

    By default, all files in the 'data' directory in the package root will be added.
    :param directory: The path to walk to generate the file list.
    :return: a list of filenames.
    """
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

# TODO: add .version to package MANIFEST.in ... research this file vs. package data
# TODO: get version from this file
# TODO: add check of this version against git version in Makefile

version_string="0.1"

print('Current version used by `setup.py`:', version_string)

package_name = "lal_cuda_dev"
setup(
    name=package_name,
    version=version_string,
    description="One line description of project.",
    author='Gregory B. Poole',
    author_email='gbpoole@gmail.com',
    install_requires=['Click'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    package_data={
        'lal_cuda_dev': package_files()},
    entry_points={
        'console_scripts': [
            'update_lal_cuda_docs=%s.scripts.update_lal_cuda_docs:update_lal_cuda_docs' %
            (package_name)]},
    packages=find_packages(),
)
