import numpy as np
import pylab as plt
import os
import sys
import click
import emcee
import pickle

from scipy.misc import logsumexp
from chainconsumer import ChainConsumer

import lal
import lalsimulation
import lal_cuda
import lal_cuda._internal.log as SID

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


def mc_eta_to_m1m2(mc, eta):
    """Convert eta to m2 and m1 >= m2."""
    if eta <= 0.25 and eta > 0.:
        root = np.sqrt(0.25 - eta)
        fraction = (0.5 + root) / (0.5 - root)
        m1 = mc * (pow(1 + 1.0 / fraction, 0.2) / pow(1.0 / fraction, 0.6))
        m2 = mc * (pow(1 + fraction, 0.2) / pow(fraction, 0.6))
        return m1, m2
    else:
        return 1., 500.


def q_to_nu(q):
    """Convert mass ratio (>= 1) to symmetric mass ratio."""
    return q / (1. + q)**2.


def htilde_of_f(freqs, m1, m2, chi1L, chi2L, chip, thetaJ, alpha, dist, fplus, fcross, phi_c, buf, legacy):

    fref = 20

    # 0Hz, so use this to get the wavefrom from fmin
    if(legacy):
        H = lalsimulation.SimIMRPhenomPFrequencySequence(
            freqs, chi1L, chi2L, chip, thetaJ, m1, m2, dist, alpha, phi_c, fref, 1, None)
    else:
        H = lalsimulation.SimIMRPhenomPFrequencySequence(
            freqs, chi1L, chi2L, chip, thetaJ, m1, m2, dist, alpha, phi_c, fref, 1, buf, None)
    hplus = H[0].data.data
    hcross = H[1].data.data

    htilde = (fplus * hplus + fcross * hcross) * np.exp(1j * np.pi * 2. * phi_c)

    return htilde


def prior(mc):
    """This is used to check that only samples within the above ranges are
    evaluated in the likelihood function."""
    mc_min, mc_max = 0., 40.
    if (mc >= mc_min) and (mc <= mc_max):
        return 1
    else:
        return 0


def logprob(mc, data, psd, freqs, buf, legacy):
    """Likelihood function."""
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

        htilde = htilde_of_f(freqs, m1, m2, chi1L, chi2L, chip, thetaJ, alpha, dist, fplus, fcross, phi_c, buf, legacy)

        deltaF = freqs[1] - freqs[0]

        logL = -0.5 * (4 * deltaF * np.vdot(data - htilde, (data - htilde) / psd).real)

        return logL

    else:
        return -np.inf


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--filename_plot', type=str, default=None)
@click.option('--filename_out', type=str, default="posterior_samples.p")
@click.option('--n_walkers', type=int, default=100, show_default=True)
@click.option('--n_steps', type=int, default=2000, show_default=True)
@click.option('--freqs_range', type=(float, float), default=(0., 1e10))
@click.option('--use_buffer/--no-use_buffer', default=True, show_default=True)
@click.option('--legacy/--no-legacy', default=False, show_default=True)
@click.argument(
    'data_files',
    nargs=2,
    type=click.Path(exists=True),
    required=False,
    default=[
        lal_cuda.full_path_datafile(
            lal_cuda.full_path_datafile("H1-freqData.dat")),
        lal_cuda.full_path_datafile(
            lal_cuda.full_path_datafile("H1-PSD.dat"))])
def PhenomPCore_mcmc(filename_plot, filename_out, n_walkers, n_steps, freqs_range, use_buffer, legacy, data_files):

    if(filename_plot):
        SID.log.open("Generating chain plots for {%s}..." % (filename_plot))

        # Instantiate chain consumer
        SID.log.open("Initializing chain consumer...")
        c = ChainConsumer()
        SID.log.close("Done.")

        # Load/add the given chain
        SID.log.open("Reading chain...")
        with open(filename_plot, "rb") as file_in:
            c.add_chain(pickle.load(file_in))
        SID.log.close("Done.", time_elapsed=True)

        # Create a filename base from the input filename
        filename_base = str(os.path.splitext(os.path.basename(filename_plot))[0])

        # Generate plots
        SID.log.open("Generating plot...")
        fig = c.plotter.plot(filename="%s.pdf" % (filename_base), figsize="column")
        SID.log.close("Done.", time_elapsed=True)

        SID.log.close("Done.", time_elapsed=True)

    else:
        SID.log.open("Generating MCMC chain...")
        SID.log.comment("Data file: {%s}" % (data_files[0]))
        SID.log.comment("PSD file:  {%s}" % (data_files[1]))

        # Initialize random seed
        np.random.seed(0)

        # Read 'freqData' file
        SID.log.open("Reading {%s}..." % (data_files[0]))
        data_file = np.column_stack(np.loadtxt(data_files[0]))

        # Determine the range of data that lies within our given frequency range
        idx_min = -1
        idx_max = len(data_file[0]) + 1
        for i_freq, freq_i in enumerate(data_file[0]):
            if(freq_i >= freqs_range[0] and idx_min <= 0):
                idx_min = i_freq
            if(freq_i < freqs_range[1]):
                idx_max = i_freq + 1
        if(idx_min < 0 or idx_max > len(data_file[0])):
            SID.log.error("Invalid frequency range [%le,%le]." % (freqs_range[0], freqs_range[1]))
        n_use = idx_max - idx_min
        SID.log.comment("Using %d of %d lines." % (n_use, len(data_file[0])))

        # Subselect the data
        freqs = data_file[0][idx_min:idx_max]
        data = data_file[1][idx_min:idx_max] + 1j * data_file[2][idx_min:idx_max]
        SID.log.close("Done.")

        # Read 'PSD' file
        SID.log.open("Reading {%s}..." % (data_files[1]))
        psd_file = np.column_stack(np.loadtxt(data_files[1]))
        SID.log.comment("Using %d of %d lines." % (n_use, len(psd_file[0])))
        freqs_PSD = psd_file[0][idx_min:idx_max]
        psd = psd_file[1][idx_min:idx_max]
        SID.log.close("Done.")

        # Confirm that the two data files have the same freq array
        if(not np.array_equal(freqs, freqs_PSD)):
            SID.log.error("Input data files do not have compatable frequency arrays.")

        # Initialize buffer
        buf = None
        if(not legacy and use_buffer):
            SID.log.open("Allocating buffer...")
            buf = lalsimulation.PhenomPCore_buffer_alloc(int(len(freqs)))
            SID.log.close("Done.")

        n_dim = 1

        # np.random.uniform(low=mc_min, high=mc_max, size=n_walkers)
        p0 = [[np.random.uniform(13, 40)] for i in range(n_walkers)]

        # Initialize sampler
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, logprob, args=[data, psd, freqs, buf, legacy])

        # Generate chain, printing a progress bar as it goes
        SID.log.open("Generating chain...")
        SID.log.progress_bar(sampler.sample, n_steps, p0, iterations=n_steps)
        SID.log.close("Done.")

        # Clean-up buffer
        if(buf):
            SID.log.open("Freeing buffer...")
            lalsimulation.PhenomPCore_buffer_free(buf)
            SID.log.close("Done.")

        # Save chain
        SID.log.open("Saving chains to {%s}..." % (filename_out))
        with open(filename_out, "wb") as file_out:
            pickle.dump(sampler.flatchain, file_out)
        SID.log.close("Done.", time_elapsed=True)

        SID.log.close("Done.", time_elapsed=True)


# Permit script execution
if __name__ == '__main__':
    status = PhenomPCore_mcmc()
    sys.exit(status)
