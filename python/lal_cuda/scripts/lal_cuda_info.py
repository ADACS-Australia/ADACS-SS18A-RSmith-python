import sys
import click

import gbpBuild.project as prj
import gbpBuild.package as pkg
import gbpBuild.docs as docs

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
def lal_cuda_info():
    """Print the dictionary of project parameters stored in the project
    (.project.yml) and package (.package.yml) files.

    :return: None
    """
    # Set/fetch all the project details we need
    project = prj.project(__file__)
    package = pkg.package(__file__)

    # Print project & package information
    print(project)
    print(package)


# Permit script execution
if __name__ == '__main__':
    status = lal_cuda_params()
    sys.exit(status)
