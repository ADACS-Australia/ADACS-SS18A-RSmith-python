import numpy as np
import pylab as plt
import os
import sys
import git
import glob
import click
from scipy.misc import logsumexp
import emcee

import lal
import lalsimulation
import lal_cuda

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
def PhenomPCore_mcmc():

    # Initialize random seed
    np.random.seed(0)

    # Load data from file
    data_file = np.column_stack(np.loadtxt(lal_cuda.full_path_datafile(lal_cuda.full_path_datafile("H1-freqData.dat"))))
    data = data_file[1] + 1j * data_file[2]
    psd_file = np.column_stack(np.loadtxt(lal_cuda.full_path_datafile(lal_cuda.full_path_datafile("H1-PSD.dat"))))
    psd = psd_file[1]
    # data and psd start at 0Hz: remove data and psd below fmin

    # Some book keeping functions
    def mc_eta_to_m1m2(mc, eta):
        # note m1 >= m2
        if eta <= 0.25 and eta > 0.:
            root = np.sqrt(0.25 - eta)
            fraction = (0.5 + root) / (0.5 - root)
            m1 = mc * (pow(1 + 1.0 / fraction, 0.2) / pow(1.0 / fraction, 0.6))
            m2 = mc * (pow(1 + fraction, 0.2) / pow(fraction, 0.6))
            return m1, m2
        else:
            return 1., 500.

    def q_to_nu(q):
        """Convert mass ratio (which is >= 1) to symmetric mass ratio."""
        return q / (1. + q)**2.

    def htilde_of_f(freqs, m1, m2, chi1L, chi2L, chip, thetaJ, alpha, dist, fplus, fcross, phi_c):

        fref = 20.

        # 0Hz, so use this to get the wavefrom from fmin
        H = lalsimulation.SimIMRPhenomPFrequencySequence(
            freqs, chi1L, chi2L, chip, thetaJ, m1, m2, dist, alpha, phi_c, fref, 1, None, None)
        hplus = H[0].data.data
        hcross = H[1].data.data

        htilde = (fplus * hplus + fcross * hcross) * np.exp(1j * np.pi * 2. * phi_c)

        return htilde

    # This is used to check that only samples within the above ranges are evaluated in the likelihood function
    def prior(mc):
        mc_min, mc_max = 13., 40.
        if (mc >= mc_min) and (mc <= mc_max):
            return 1
        else:
            return 0

    def logprob(mc, data, psd, freqs):
        mc = mc[0]
        q = 0.5
        chi1L = 0.2
        chi2L = 0.2
        chip = 0.2
        thetaJ = 0.1
        alpha = 0.2
        dist = 1000 * 1e6 * lal.lal.PC_SI
        fplus = 1
        fcross = 1
        phi_c = 0.3

        if prior(mc):
            eta = q_to_nu(q)
            m1, m2 = mc_eta_to_m1m2(mc, eta)

            m1 *= lal.lal.MSUN_SI
            m2 *= lal.lal.MSUN_SI

            htilde = htilde_of_f(freqs, m1, m2, chi1L, chi2L, chip, thetaJ, alpha, dist, fplus, fcross, phi_c)

            deltaF = freqs[1] - freqs[0]

            # len(data)*( -0.5* ( -2*np.fft.irfft(dh)) )
            logL = -0.5 * (4 * deltaF * np.vdot(data - htilde, (data - htilde) / psd).real)

            return logL

        else:
            return -np.inf

    # Parameter ranges
    fmin = 20.
    fmax = 1024
    deltaF = 1. / 4.
    # frequencies at which waveform spectra is evaluated at
    freqs = np.linspace(fmin, fmax, int((fmax - fmin) / deltaF) + 1)

    # Initial values of parameters used by emcee
    nwalkers = 100
    ndim = 1
    nsteps = 2000

    # np.random.uniform(low=mc_min, high=mc_max, size=nwalkers)
    p0 = [[np.random.uniform(13, 40)] for i in range(nwalkers)]

    # Start sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=(data, psd, freqs))
    sampler.run_mcmc(p0, nsteps)

    # Save samples
    import pickle
    pickle.dump(sampler.flatchain, open("posterior_samples.p", "wb"))

    # Generate plot
    import matplotlib.pyplot as pl
    for i in range(ndim):
        pl.figure()
        #pl.hist(sampler.flatchain[:,i][::int(sampler.acor[i])], 100, color="k", histtype="step")
        pl.hist(sampler.flatchain[:, i][:], 100, color="k", histtype="step")
        pl.title("Dimension {0:d}".format(i))
        pl.savefig("%i.tmp.png" % i)


# Permit script execution
if __name__ == '__main__':
    status = PhenomPCore_mcmc()
    sys.exit(status)
