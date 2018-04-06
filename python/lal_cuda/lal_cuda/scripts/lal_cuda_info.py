import os
import sys
import git
import glob
import click

# Find the project root directory
git_repo = git.Repo(os.path.realpath(__file__), search_parent_directories=True)
dir_root = git_repo.git.rev_parse("--show-toplevel")
dir_python = os.path.abspath(os.path.join(dir_root, "python"))

# Include the paths to local python projects (including the _dev package)
# Make sure we prepend to the list to make sure that we don't use an
# installed version.  We need access to the information in the
# project directory here.
for setup_py_i in glob.glob(dir_python + "/**/setup.py", recursive=True):
    sys.path.insert(0, os.path.abspath(os.path.dirname(setup_py_i)))

# Import the project development module
import lal_cuda_dev.project as prj
import lal_cuda_dev.docs as docs

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
def lal_cuda_params():
    """Print the dictionary of project parameters stored in the project
    .project.yml file.

    :return: None

    """
    # Set/fetch all the project details we need
    project = prj.project()

    # Print project information
    project.print2stdout()


# Permit script execution
if __name__ == '__main__':
    status = lal_cuda_params()
    sys.exit(status)
