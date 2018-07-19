import os
import sys
import importlib
import click

# Infer the name of this package from the path of __file__
package_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
package_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
package_name = os.path.basename(package_root_dir)

# Make sure that what's in this path takes precidence
# over an installed version of the project
sys.path.insert(0, package_parent_dir)

# Import needed internal modules
prj = importlib.import_module(package_name + '._internal.project')


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
def lal_cuda_info():
    """Print the dictionary of project parameters stored in the project
    (.project.yml) and package (.package.yml) files.

    :return: None
    """
    # Set/fetch all the project details we need
    project = prj.project(__file__)

    # Print project information
    print(project)

    # Print package information
    for package in project.packages:
        print(package)


# Permit script execution
if __name__ == '__main__':
    status = lal_cuda_params()
    sys.exit(status)
