import shutil
import filecmp
import os
import sys
import importlib

import yaml

# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
package_name = os.path.basename(package_root_dir)

# Make sure that what's in this path takes precidence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import needed internal modules
pkg = importlib.import_module(package_name)


class package:
    """This class provides a package object, storing package parameters which
    describe the package.

    Inputs: path_call; this needs to be the FULL (i.e. absolute) path to a file or directory living somewhere in the package
    """

    def __init__(self, path_call):

        # Scan upwards from the given path until 'setup.py' is found.  That will be the package parent directory.
        self.path_package_parent = pkg.find_in_parent_path(path_call, ".package.yml")

        # Assume that the tail of the root path is the package name
        self.package_name = os.path.basename(self.path_package_parent)

        # Set the path where all the package modules start
        self.path_package_root = os.path.join(self.path_package_parent, self.package_name)

        # Read the package file
        with open_package_file(self.path_package_parent) as file_in:
            self.params = file_in.load()

        # Assemble a list of data files to bundle with the package
        self.package_files = self.collect_package_files()

    def collect_package_files(self):
        """Generate a list of non-code files to be included in the package.

        By default, all files in the 'data' directory in the package root will be added.
        :return: a list of absolute paths.
        """
        paths = []
        # Add the .project.yml and .package.yml files.  There are instances where these
        # don't get installed by default (tox virtual envs, for example) and we need
        # to make sure they are present
        paths.append(os.path.abspath(os.path.join(self.path_package_parent, ".project.yml")))
        paths.append(os.path.abspath(os.path.join(self.path_package_parent, ".project_aux.yml")))
        paths.append(os.path.abspath(os.path.join(self.path_package_parent, ".package.yml")))

        # Add the data directory
        for (path, directories, filenames) in os.walk(os.path.join(self.path_package_parent, "data"), followlinks=True):
            if(path != "__pycache__"):
                for filename in filenames:
                    paths.append(os.path.join('..', path, filename))
        return paths

    def collect_package_scripts(self):
        """Generate a list of script files associated with this package.

        By default, all files in the 'scripts' directory in the package root will be added.
        :return: a list of absolute paths.
        """
        paths = []

        # Add the scripts directory
        path_start = os.path.join(self.path_package_root, "scripts")
        for (path, directories, filenames) in os.walk(path_start, followlinks=True):
            for filename in filenames:
                filename_base = os.path.basename(filename)
                script_name, filename_extension = os.path.splitext(filename_base)
                if(filename_extension == '.py'):
                    if(script_name != '__init__'):
                        path_relative = os.path.relpath(path, path_start)
                        script_pkg_path = path_relative.replace("/", ".")
                        if(script_pkg_path == '.'):
                            script_pkg_path = ''
                        else:
                            script_pkg_path += '.'
                        script_pkg_path += script_name
                        paths.append([script_name, script_pkg_path])
        return paths

    def __str__(self):
        """Convert dictionary of package parameters to a string.

        :return: string
        """
        result = "Package information:\n"
        result += "--------------------\n"
        for k, v in sorted(self.params.items()):
            result += '   ' + k + " = " + str(v) + '\n'

        return result


class package_file():
    def __init__(self, path_package_parent):
        # File pointer
        self.fp = None

        # Assume this filename for the package file
        self.filename_package_filename = '.package.yml'

        # Set the filename of the package copy of the package file
        self.filename_package_file = os.path.join(path_package_parent, self.filename_package_filename)

    def open(self):
        try:
            self.fp = open(self.filename_package_file)
        except BaseException:
            pkg.log.error("Could not open package file {%s}." % (self.filename))
            raise

    def close(self):
        try:
            self.fp.close()
        except BaseException:
            pkg.log.error("Could not close package file {%s}." % (self.filename))
            raise

    def load(self):
        try:
            params_list = yaml.load(self.fp)
        except BaseException:
            pkg.log.error("Could not load package file {%s}." % (self.filename))
            raise
        finally:
            return {k: v for d in params_list for k, v in d.items()}


class open_package_file:
    """Open package file."""

    def __init__(self, path_call):
        self.path_call = path_call

    def __enter__(self):
        # Open the package's copy of the file
        pkg.log.open("Opening package...")
        try:
            self.file_in = package_file(self.path_call)
            self.file_in.open()
        except BaseException:
            pkg.log.error("Could not open package file.")
            raise
        finally:
            pkg.log.close("Done.")
            return self.file_in

    def __exit__(self, *exc):
        self.file_in.close()
        return False
