import os
import sys
from setuptools import setup, find_packages

# Make sure that what's in this path takes precidence
# over an installed version of the project
sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)))

import gbpBuild.project as prj
import gbpBuild.package as pkg
import gbpBuild.log as SID

# Fetch all the meta data for the project & package
this_project = prj.project(os.path.abspath(__file__))
this_package = pkg.package(os.path.abspath(__file__))

# Print project and package meta data to stdout
SID.log.comment('')
SID.log.comment(this_project)
SID.log.comment(this_package)

# This line converts the package_scripts list above into the entry point 
# list needed by Click, provided that: 
#    1) each script is in its own file
#    2) the script name matches the file name
#    3) There is only one script per file
entry_points = [ "%s=%s.scripts.%s:%s"%(script_name_i,this_package.params['name'],script_pkg_path_i,script_name_i) for script_name_i,script_pkg_path_i in this_package.collect_package_scripts() ]

# Execute setup
setup(
    name=this_package.params['name'],
    version=this_project.params['version'],
    description=this_package.params['description'],
    author=this_project.params['author'],
    author_email=this_project.params['author_email'],
    install_requires=['Click','gbpBuild','PyYAML'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    packages=find_packages(),
    entry_points={'console_scripts': entry_points},
    package_data={this_package.params['name']: this_package.package_files},
)
