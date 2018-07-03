import os
import gbpBuild.log as SID

_PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
def full_path_datafile(path):
    """
    Return the full *INSTALLED* path to a directory in the package's root directory.
    :param path: A path relative to the package root directory
    :return: The installed path
    """
    return os.path.join(_PACKAGE_ROOT, 'data', path)

def find_in_parent_path(path_start,filename_search,check=True):
    # Initialize the start
    path_result = None
    if(os.path.isdir(path_start)):
        cur_dir = path_start
    else:
        cur_dir = os.path.dirname(path_start)

    # Scan upwards until we find the file or run out of path
    while(True):
        filename_test = os.path.join(cur_dir,filename_search)
        if(os.path.isfile(filename_test)):
            path_result = cur_dir
            break
        elif (cur_dir == os.sep):
            break
        else:
            cur_dir = os.path.dirname(cur_dir)

    # Check if the file has been found
    if(check and path_result == None):
        SID.log.error("Could not find {%s} in parent directories of path {%s}."%(filename_search,path_start))

    return path_result
