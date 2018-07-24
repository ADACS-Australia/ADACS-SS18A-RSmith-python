Installation
============

This package assumes that you have a working copy of the LSC Algorithm Library Suite (`LALSuite <https://wiki.ligo.org/DASWG/LALSuite>`_) installed (see `here <https://wiki.ligo.org/DASWG/LALSuiteInstall#Installing_from_the_git_repository>`__ for instructions).  Furthermore, it assumes that you are working with the GPU-enabled version developed by ADACS, which can be obtained `here <https://github.com/ADACS-Australia/ADACS-SS18A-RSmith>`__ .  If this version is not being used, make sure all scripts in this package are run with the '--legacy' flag and that the 'legacy' setting in the tests is enabled, if you want to use those.

.. note:: A makefile is provided in the project directory to ease the use of this software project.  Type `make help` for a list of options.
.. warning:: If the ADACS version of LALSuite is not being used with the executables in this package, make sure to use the '--legacy' flag and -- if you intend to use them -- enable the 'legacy' setting in the tests.

Installing LALSuite
-------------------
In detail: to download the ADACS version of LALSuite (to `/path/to/src/dir`; adjust according to your needs) and compile/install the resulting libraries (to `/path/to/dir/install`; adjust according to your needs), perform the following:

.. code-block:: console

    $ cd /path/to/src/dir
    $ git clone  --single-branch -b ADACS https://github.com/ADACS-Australia/ADACS-SS18A-RSmith.git
    $ cd ADACS-SS18A-RSmith
    $ ./00boot
    $ ./configure --with-cuda --enable-python --prefix=/path/to/dir/install
    $ make -j 24 install

.. note:: The GPU implementation of the code is in the 'ADACS' branch of this repository.  The `git clone` command above will ensure that you are working with it before compiling.

.. note:: The `--enable-cuda` option is required for GPU accelleration.  However, it can be omitted if an NVidia GPU is not available.

Installing Python package
-------------------------
To install this Python package, it needs to be downloaded and installed as follows:

.. code-block:: console

    $ cd /path/to/src/dir
    $ git clone https://github.com/ADACS-Australia/ADACS-SS18A-RSmith-python
    $ cd ADACS-SS18A-RSmith-python
    $ make init
    $ make install

.. note:: This package and all needed dependancies will be installed in your current Python environment.
    Make sure you have this properly configured before doing this.

