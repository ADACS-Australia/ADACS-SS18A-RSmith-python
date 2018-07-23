Installation
============

This package assumes that you have a working copy of the LSC Algorithm Library Suite (`LALSuite <https://wiki.ligo.org/DASWG/LALSuite>`_) installed (see `here <https://wiki.ligo.org/DASWG/LALSuiteInstall#Installing_from_the_git_repository>`_ for instructions).  Note however that the code assumes that you are working with GPU-enabled version which can be cloned from `here2 <https://github.com/ADACS-Australia/ADACS-SS18A-RSmith>`_ (make sure that the code is configured with '--enable-cuda' if you want the GPU acceleration activated).  If this version is not being used, make sure all scripts are run with the '--legacy' flag and that the 'legacy' setting in the tests is enabled, if you want to use those.

Installing LALSuite
-------------------
In detail: to download the ADACS version of LALSuite to `/path/to/src/dir`, compile and install the resulting libraries to `/path/to/dir/install`, perform the following (adjust all paths according to your needs):

.. code-block:: console

    $ cd /path/to/src/dir
    $ git clone  --single-branch -b ADACS https://github.com/ADACS-Australia/ADACS-SS18A-RSmith.git
    $ cd ADACS-SS18A-RSmith
    $ ./00boot
    $ ./configure --with-cuda --enable-python --prefix=/path/to/dir/install
    $ make -j 24 install

.. note:: The GPU implementation of the code is in the 'ADACS' branch of this repository.  Make sure you ahave it checked-out before compiling.

.. note:: The `--enable-cuda` option is required for GPU accelleration.
    However, it can be omitted if an NVidia GPU is not available. 

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

