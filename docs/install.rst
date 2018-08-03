.. _Installation:

Installation
============

This section describes how to download and install the ADACS version of the LSC Algorithm Library Suite (`LALSuite <https://wiki.ligo.org/DASWG/LALSuite>`_) as well as the Python package `lal_cuda` developed to ease testing during its development and to illustrate the use of the LALSimulation routines addressed by this project (including some minor changes to the LALSuite API).

Installing LALSuite
-------------------

The installation of the ADACS branch of LALSuite is essentially the same as the standard version it branched from (see `here <https://wiki.ligo.org/DASWG/LALSuiteInstall#Installing_from_the_git_repository>`__ for more information).  In brief: to download the ADACS version of LALSuite (to `/path/to/src/dir`; adjust according to your needs) and compile/install the resulting libraries (to `/path/to/dir/install`; adjust according to your needs), perform the following:

.. code-block:: console

    $ cd /path/to/src/dir
    $ git clone --single-branch -b ADACS https://github.com/ADACS-Australia/ADACS-SS18A-RSmith.git
    $ cd ADACS-SS18A-RSmith
    $ ./00boot
    $ ./configure --enable-cuda --enable-python --prefix=/path/to/dir/install
    $ make -j 24 install

.. note:: The GPU implementation of the code is in the 'ADACS' branch of this repository.  The `git clone` command above will ensure that you are working with it before compiling.

.. note:: The `--enable-cuda` option is required for GPU acceleration.  However, it can be omitted if an NVidia GPU is not available.

.. note:: The `--enable-python` option is required to use the `lal_cuda` Python package.

Installing the lal_cuda Python package
--------------------------------------

This package assumes that you have a working copy of the LSC Algorithm Library Suite (`LALSuite <https://wiki.ligo.org/DASWG/LALSuite>`_) installed (see `here <https://wiki.ligo.org/DASWG/LALSuiteInstall#Installing_from_the_git_repository>`__ for instructions).  Furthermore, by default it assumes that you are working with the GPU-enabled version developed by ADACS, which can be obtained `here <https://github.com/ADACS-Australia/ADACS-SS18A-RSmith>`__ .  If this version is not being used, make sure all scripts in this package are run with the '--legacy' flag and that the 'legacy' setting in the tests is enabled, if you want to use those.  Furthermore, before working with the lal_cuda library make sure that the LALSuite SWIG libraries are added to your Python path using the approach for setting environment variables appropriate for your shell.  For example, for `bash` this can be done as follows (using the install directory in the example above; adjust accordingly):

.. code-block:: console

    $ export PYTHONPATH=${PYTHONPATH}:/path/to/dir/install/lib/python2.7/site-packages/

To install `lal_cuda`, it needs to be downloaded and installed into your Python environment as follows:

.. code-block:: console

    $ cd /path/to/src/dir
    $ git clone https://github.com/ADACS-Australia/ADACS-SS18A-RSmith-python
    $ cd ADACS-SS18A-RSmith-python
    $ make init
    $ make install

.. note:: A makefile is provided in the project directory to ease the use of this software project.  Type `make help` for a list of options.
.. warning:: If the ADACS version of LALSuite is not being used with the executables in this package, make sure to use the '--legacy' flag and -- if you intend to use them -- enable the 'legacy' setting in the tests.
.. warning:: Make sure that the `make init` line is run first-thing before installing.  It will ensure that all needed dependencies are present in your current Python environment.
    Make sure to re-run this line before re-installing, if you change Python environments.
