import os
import sys
import importlib
from setuptools import setup, find_packages

# Make sure that what's in this path takes precidence
# over an installed version of the project
sys.path.insert(0, os.path.abspath(__file__))

# Infer the name of this package from the path of __file__
package_root_dir = os.path.abspath(os.path.dirname(__file__))
package_name = os.path.basename(package_root_dir)

# Make sure that what's in this path takes precidence
# over an installed version of the project
sys.path.insert(0, os.path.join(package_root_dir, package_name))

# Import needed internal modules
pkg = importlib.import_module(package_name)
_prj = importlib.import_module(package_name + '._internal.project')
_pkg = importlib.import_module(package_name + '._internal.package')

# Fetch all the meta data for the project & package
this_project = _prj.project(os.path.abspath(__file__))
this_package = _pkg.package(os.path.abspath(__file__))

# Print project and package meta data to stdout
pkg.log.comment('')
pkg.log.comment(this_project)
pkg.log.comment(this_package)

# This line converts the package_scripts list above into the entry point
# list needed by Click, provided that:
#    1) each script is in its own file
#    2) the script name matches the file name
#    3) There is only one script per file
pkg.log.comment("Executable scripts:")
package_scripts = this_package.collect_package_scripts()
entry_points = []
for script_name_i, script_pkg_path_i in package_scripts:
    entry_points.append(
        "%s=%s.scripts.%s:%s" %
        (script_name_i,
         this_package.params['name'],
         script_pkg_path_i,
         script_name_i))
    pkg.log.append(" %s" % (script_name_i))
pkg.log.unhang()

# Execute setup
setup(
    name=this_package.params['name'],
    version=this_project.params['version'],
    description=this_package.params['description'],
    author=this_project.params['author'],
    author_email=this_project.params['author_email'],
    install_requires=['Click'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    packages=find_packages(),
    entry_points={'console_scripts': entry_points},
    package_data={this_package.params['name']: this_package.package_files},
    include_package_data=True
)
