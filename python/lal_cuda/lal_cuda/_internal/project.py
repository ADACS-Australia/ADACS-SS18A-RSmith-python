import shutil
import filecmp
import os
import sys
import importlib

import yaml
import git

# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
package_name = os.path.basename(package_root_dir)

# Make sure that what's in this path takes precidence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import the current package
this_pkg = importlib.import_module(package_name)

# Import the internal package-helper package
_pkg = importlib.import_module(package_name + '._internal.package')

def constructor(loader, node):
    return node.value


# This hack deals with a python2.7 error with PyYaml, See here:
# https://stackoverflow.com/questions/27518976/how-can-i-get-pyyaml-safe-load-to-handle-python-unicode-tag
yaml.SafeLoader.add_constructor("tag:yaml.org,2002:python/unicode", constructor)


class project:
    """This class provides a project object, storing project parameters which
    describe the project.

    Inputs: path_call; this needs to be the path to a file or directory living somewhere in the project
    """

    def __init__(self, path_call):
        # Store the path_call
        self.path_call = path_call

        # Assume this filename for the project file
        self.filename_project_filename = '.project.yml'
        self.filename_auxiliary_filename = '.project_aux.yml'

        # Set the filename of the package copy of the project file
        package_root = this_pkg.find_in_parent_path(self.path_call, self.filename_project_filename)
        if(package_root is not None):
            self.filename_project_file = os.path.join(package_root, self.filename_project_filename)
            self.filename_auxiliary_file = os.path.abspath(
                os.path.join(
                    os.path.dirname(
                        self.filename_project_file),
                    self.filename_auxiliary_filename))
        else:
            self.filename_project_file = ''
            self.filename_auxiliary_file = ''

        # Determine if we are in a project repository.  Set to None if not.
        self.path_project_root = None
        self.filename_project_file_source = None
        try:
            with git.Repo(os.path.realpath(self.path_call), search_parent_directories=True) as git_repo:
                path_project_root_test = git_repo.git.rev_parse("--show-toplevel")
                # Check that there is a .project.yml file here.  Otherwise, we may be sitting in the path
                # of some other repo, and not a project repo
                if(not os.path.isfile(os.path.join(path_project_root_test, self.filename_project_filename))):
                    raise Exception("No project file found.")
                else:
                    self.path_project_root = path_project_root_test
                    self.filename_project_file_source = os.path.normpath(
                        os.path.join(self.path_project_root, self.filename_project_filename))
        except BaseException:
            this_pkg.log.comment("Installed environment will be assumed.")

        # Read the project file
        with open_project_file(self) as file_in:
            self.params = file_in.load()

        # Load meta data of Python packages
        self.packages = []
        for package_name in self.params['python_packages']:
            package_setup_py = os.path.abspath(os.path.join(self.params['dir_python'],package_name,'setup.py'))
            self.packages.append(_pkg.package(os.path.abspath(package_setup_py)))

    def add_packages_to_path(self):
        """Import all the python packages belonging to this project."""
        dir_file = os.path.abspath(self.path_project_root)
        count = 0
        for (directory, directories, filenames) in os.walk(dir_file):
            for filename in filenames:
                # Exclude gbpBuild (load it independently) for the case
                # that we're running this within that package.
                if(filename == "setup.py"):
                    path_package = os.path.abspath(directory)
                    sys.path.insert(0, path_package)
                    count += 1
                    break
        return count

    def __str__(self):
        """Convert dictionary of project parameters to a string.

        :return: string
        """
        result = "Project information:\n"
        result += "--------------------\n"
        for k, v in sorted(self.params.items()):
            result += '   ' + k + " = " + str(v) + '\n'

        return result


class project_file():
    def __init__(self, project):
        # Keep a record of inputs
        self.project = project

        # File pointer
        self.fp_prj = None
        self.fp_aux = None

        # Update the project file
        self.update()

    def update(self):

        # Check if we are inside a project repository...
        if(self.project.path_project_root):
            # ... if so, update the package's copy of the project file.  This is needed because if
            #    this is being run from an installed package, then there is no access to files outside
            #    of the package, and we need to work with an up-to-date copy instead.
            this_pkg.log.open("Validating package's project files...")
            try:
                flag_update = False
                if(not os.path.isfile(self.project.filename_project_file)):
                    flag_update = True
                elif(not filecmp.cmp(self.project.filename_project_file_source, self.project.filename_project_file)):
                    flag_update = True
                if(flag_update):
                    # Make a copy of the project file
                    shutil.copy2(self.project.filename_project_file_source, self.project.filename_project_file)
                    this_pkg.log.close("Updated.")
                else:
                    this_pkg.log.close("Up-to-date.")
            except BaseException:
                this_pkg.log.error("Could not update package's project file.")

            # Create a dictionary of a bunch of auxiliary project information

            # Set some project directories
            aux_params = []
            aux_params.append({'dir_docs': os.path.abspath(os.path.join(self.project.path_project_root, "docs"))})
            aux_params.append({'dir_docs_api_src': os.path.abspath(
                os.path.join(self.project.path_project_root, "docs/src"))})
            aux_params.append({'dir_docs_build': os.path.abspath(
                os.path.join(self.project.path_project_root, "docs/_build"))})
            aux_params.append({'dir_python': os.path.abspath(os.path.join(self.project.path_project_root, "python"))})

            # Get a list of python packages
            python_path = aux_params[-1]['dir_python']
            python_path_dirs = os.listdir(python_path)
            python_pkgs = []
            for python_path_dir_i in python_path_dirs:
                if(python_path_dir_i != 'build' and python_path_dir_i != '_build'):
                    python_pkgs.append(python_path_dir_i)
            aux_params.append({'python_packages': python_pkgs})

            # Check if this is a C-project (the appropriate makefile will be present if so)
            if(os.path.isfile(os.path.join(self.project.path_project_root, ".Makefile-c"))):
                aux_params.append({'is_C_project': True})
            else:
                aux_params.append({'is_C_project': False})

            # Check if this is a Python-project (the appropriate makefile will be present if so)
            if(os.path.isfile(os.path.join(self.project.path_project_root, ".Makefile-py"))):
                aux_params.append({'is_Python_project': True})
            else:
                aux_params.append({'is_Python_project': False})

            # Extract version & release from .version file.
            try:
                with open("%s/.version" % (self.project.path_project_root), "r") as fp_in:
                    version_string_source = str(fp_in.readline()).strip('\n')
                    aux_params.append({'version': version_string_source})
            except BaseException:
                this_pkg.log.comment("Project '.version' file not found.  Setting version='unset'")
                aux_params.append({'version': 'unset'})

            # TODO: Need to split version from release.
            aux_params.append({'release': version_string_source})

            # Write auxiliary parameters file
            with open(self.project.filename_auxiliary_file, 'w') as outfile:
                # UTF-8 to ensure compatability between py27 and py3X
                yaml.dump(aux_params, outfile, default_flow_style=False, encoding='utf-8')

    def open(self):
        try:
            self.fp_prj = open(self.project.filename_project_file)
            self.fp_aux = open(self.project.filename_auxiliary_file)
        except BaseException:
            this_pkg.log.error("Could not open project file {%s}." % (self.project.filename))
            raise

    def close(self):
        try:
            if(self.fp_prj is not None):
                self.fp_prj.close()
            if(self.fp_aux is not None):
                self.fp_aux.close()
        except BaseException:
            this_pkg.log.error("Could not close project file {%s}." % (self.project.filename))
            raise

    def load(self):
        try:
            params_list = []
            params_list.append(yaml.safe_load(self.fp_prj))
            params_list.append(yaml.safe_load(self.fp_aux))
            # Add a few extra things
            params_list.append([{'path_project_root': self.project.path_project_root}])
        except BaseException:
            this_pkg.log.error("Could not load project file {%s}." % (self.project.filename))
            raise
        finally:
            result = dict()
            for params in params_list:
                result.update({k: v for d in params for k, v in d.items()})
            return result


class open_project_file:
    """Open project file."""

    def __init__(self, project):
        self.project = project

    def __enter__(self):
        # Open the package's copy of the file
        this_pkg.log.open("Opening project...")
        try:
            self.file_in = project_file(self.project)
            self.file_in.open()
        except BaseException:
            this_pkg.log.error("Could not open project file.")
            raise
        finally:
            this_pkg.log.close("Done.")
            return self.file_in

    def __exit__(self, *exc):
        self.file_in.close()
        return False
