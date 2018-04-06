import numpy as np
import pylab as plt
import os
import sys
import git
import glob
import click

sys.path.insert(0, "/home/gpoole/3rd_Party/lib/python2.7/site-packages")
import lal
import lalsimulation

# Find the project root directory
git_repo = git.Repo(os.path.realpath(__file__), search_parent_directories=True)
dir_root = git_repo.git.rev_parse("--show-toplevel")
dir_python = os.path.abspath(os.path.join(dir_root, "python"))

# Include the paths to local python projects (including the _dev package)
# Make sure we prepend to the list to make sure that we don't use an
# installed version.  We need access to the information in the
# project directory here.
for root, dirs, files in os.walk(dir_python):
    if("setup.py" in files):
        sys.path.insert(0, os.path.abspath(root))

# Import the project development module
import lal_cuda_dev.project as prj
import lal_cuda_dev.docs as docs

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
def PhenomPexample():
    """The script calls a higher level function in LALSuite.  The output is two
    binary arrays corresponding to the two outputs hp_val, hc_val from
    PhenomPCore. The outputs are arrays of complex numbers; one for each
    frequency bin. In this example, the function is evaluated at 81 frequency
    bins so the output arrays should each contain 81 complex numbers. Some
    explanation of the code below:

    I'm calling (from python):

    lalsimulation.SimIMRPhenomPFrequencySequence(...)

    which is calling the C-function XLALSimIMRPhenomPFrequencySequence.  That function calls PhenomPCore which in turn calls PhenomPCoreOneFrequency.  The arguments that are passes to PhenomPCoreOneFrequency:

    (f, eta, chi1_l, chi2_l, chip, distance, M, phic, pAmp, pPhi, PCparams, pn, &angcoeffs, &Y2m, alphaNNLOoffset - alpha0, epsilonNNLOoffset, &hp_val, &hc_val, &phasing, IMRPhenomP_version, &amp_prefactors, &phi_prefactors)

    are derived from the ones passed to XLALSimIMRPhenomPFrequencySequency

        :return: None

    """

    chi1_l, chi2_l, m1, m2, chip, thetaJ, alpha0, distance, phic, fref = 0.1, 0.2, 30, 30, 0.34, 1.1, 1.5, 1000, np.pi * 0.4, 30

    m1SI = m1 * lal.lal.MSUN_SI
    m2SI = m2 * lal.lal.MSUN_SI

    flow, fhigh = 20, 100

    freqs = np.linspace(flow, fhigh, (fhigh - flow) + 1)

    H = lalsimulation.SimIMRPhenomPFrequencySequence(
        freqs,
        chi1_l,
        chi2_l,
        chip,
        thetaJ,
        m1SI,
        m2SI,
        distance *
        lal.lal.PC_SI *
        100 *
        1e6,
        alpha0,
        phic,
        fref,
        1,
        None)

    hp_val = H[0].data.data
    hc_val = H[1].data.data

    # Write results to screen
    for [f_i,hp_i,hc_i] in zip(freqs,hp_val,hc_val):
        print(f_i,hp_i.real,hp_i.imag,hc_i.real,hc_i.imag)

    # Write results to binary files
    freqs_file  = open("./freqs.dat", "wb")
    freqs.tofile(freqs_file)
    freqs_file.close()

    hp_val_file = open("./hp_val.dat", "wb")
    hp_val.tofile(hp_val_file)
    hp_val_file.close()

    hc_val_file = open("./hc_val.dat", "wb")
    hc_val.tofile(hc_val_file)
    hc_val_file.close()

# Permit script execution
if __name__ == '__main__':
    status = lal_cuda_params()
    sys.exit(status)
