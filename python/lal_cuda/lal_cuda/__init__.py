import os
import sys
import importlib

# Make sure that what's in this path takes precidence
# over an installed version of the project
sys.path.insert(0, os.path.join(os.path.abspath(__file__), '..'))

# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
package_name = os.path.basename(package_root_dir)

# Read the package docstring.  We do things this way to seaparate package-specific
# content from implementation, to make it easier to update this file, for example
path_package_docstring = os.path.join(package_root_dir, "%s.docstring" % (package_name))
with open(path_package_docstring, "r") as fp_docstring:
    __doc__ = ''.join(fp_docstring.readlines())

# Make sure that what's in this path takes precidence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import needed internal modules
_log = importlib.import_module(package_name + '._internal.log')

#: The library log stream (see the `_internal.log` module for more details)
log = _log.log_stream()

#: The absolute path to the module root path
_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def full_path_datafile(path):
    """Return the full *INSTALLED* path to a file in the package's data
    directory.

    :param path: A path relative to the package's `/data` directory
    :return: The installed path
    """
    return os.path.join(_PACKAGE_ROOT, 'data', path)


def find_in_parent_path(path_start, filename_search, check=True):
    """Find the path to a given filename, scanning up the directory tree from
    the given path_start.  Optionally throw an error (if check=True) if not
    found.

    :param path_start: The path from which to start the search.
    :param filename: The filename to search for.
    :return: Path to the file if found, None if not found.
    """
    path_result = None
    if(os.path.isdir(path_start)):
        cur_dir = path_start
    else:
        cur_dir = os.path.dirname(path_start)

    # Scan upwards until we find the file or run out of path
    while(True):
        filename_test = os.path.join(cur_dir, filename_search)
        if(os.path.isfile(filename_test) or os.path.isdir(filename_test)):
            path_result = cur_dir
            break
        elif (cur_dir == os.sep):
            break
        else:
            cur_dir = os.path.dirname(cur_dir)

    # Check if the file has been found
    if(check and path_result is None):
        log.error("Could not find {%s} in parent directories of path {%s}." % (filename_search, path_start))

    return path_result
