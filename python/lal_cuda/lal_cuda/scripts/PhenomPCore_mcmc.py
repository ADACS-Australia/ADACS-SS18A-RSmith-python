import numpy as np
import os
import sys
import click
import emcee
import pickle
import pylab as plt

from scipy.misc import logsumexp

import lal_cuda

# Generate mocks for these if we are building for RTD
lal = lal_cuda.import_mock_RTD("lal")
lalsimulation = lal_cuda.import_mock_RTD("lalsimulation")

# Make sure this is after the `_tkinter` import above
from chainconsumer import ChainConsumer

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
            freqs, chi1L, chi2L, chip, thetaJ, m1, m2, dist, alpha, phi_c, fref, 1, None, buf)
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


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--filename_plot', type=str, default=None, help='Specify a chain file to plot.')
@click.option('--filename_out', type=str, default="posterior_samples.p", help='Specify a file name for chain output.')
@click.option('--n_walkers', type=int, default=100, show_default=True,
              help='Specify the number of emcee walkers to use.')
@click.option('--n_steps', type=int, default=2000, show_default=True, help='Specify the number of emcee steps to take')
@click.option('--freqs_range', type=(float, float), default=(0., 1e10),
              help='Specify the frequency range of the fit as MIN MAX.')
@click.option('--use_buffer/--no-use_buffer', default=True, show_default=True, help='Use a buffer for accelleration.')
@click.option('--n_streams', type=int, default=0, show_default=True, help='Number of asynchronous streams')
@click.option('--legacy/--no-legacy', default=False, show_default=True,
              help='Specify this option if a legacy version of LALSuite (without buffer support) is being used.')
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
def PhenomPCore_mcmc(
        filename_plot,
        filename_out,
        n_walkers,
        n_steps,
        freqs_range,
        use_buffer,
        n_streams,
        legacy,
        data_files):
    """This script either generates (default) or plots (by adding the
    option: --filename_plot) an MCMC chain describing the posterior probability
    of a model (gernerated from LALSuite; see below) fit to a two-file dataset
    (given by the optional positional arguments; a default dataset stored with
    the package is used by default, if no positional arguments are given).

    The model is generated with the call:

    lalsimulation.SimIMRPhenomPFrequencySequence(...)

    which calls the C-function XLALSimIMRPhenomPFrequencySequence.  That function
    then calls PhenomPCore which in-turn calls PhenomPCoreOneFrequency.
    """
    if(filename_plot):
        lal_cuda.log.open("Generating chain plots for {%s}..." % (filename_plot))

        # Instantiate chain consumer
        lal_cuda.log.open("Initializing chain consumer...")
        c = ChainConsumer()
        lal_cuda.log.close("Done.")

        # Load/add the given chain
        lal_cuda.log.open("Reading chain...")
        with open(filename_plot, "rb") as file_in:
            c.add_chain(pickle.load(file_in))
        lal_cuda.log.close("Done.", time_elapsed=True)

        # Create a filename base from the input filename
        filename_base = str(os.path.splitext(os.path.basename(filename_plot))[0])

        # Generate plots
        lal_cuda.log.open("Generating plot...")
        fig = c.plotter.plot(filename="%s.pdf" % (filename_base), figsize="column")
        lal_cuda.log.close("Done.", time_elapsed=True)

        lal_cuda.log.close("Done.", time_elapsed=True)

    else:
        lal_cuda.log.open("Generating MCMC chain...")
        lal_cuda.log.comment("Data file: {%s}" % (data_files[0]))
        lal_cuda.log.comment("PSD file:  {%s}" % (data_files[1]))

        # Initialize random seed
        np.random.seed(0)

        # Read 'freqData' file
        lal_cuda.log.open("Reading {%s}..." % (data_files[0]))
        data_file = np.column_stack(np.loadtxt(data_files[0]))

        # Determine the range of data that lies within our given frequency range
        idx_min = -1
        idx_max = len(data_file[0]) + 1
        for i_freq, freq_i in enumerate(data_file[0]):
            if(freq_i >= freqs_range[0] and idx_min < 0):
                idx_min = i_freq
            if(freq_i <= freqs_range[1]):
                idx_max = i_freq + 1
        if(idx_min < 0 or idx_max > len(data_file[0])):
            lal_cuda.log.error("Invalid frequency range [%le,%le]." % (freqs_range[0], freqs_range[1]))
        n_use = idx_max - idx_min
        lal_cuda.log.comment("Using %d of %d lines." % (n_use, len(data_file[0])))

        # Subselect the data
        freqs = data_file[0][idx_min:idx_max]
        data = data_file[1][idx_min:idx_max] + 1j * data_file[2][idx_min:idx_max]
        lal_cuda.log.close("Done.")

        # Read 'PSD' file
        lal_cuda.log.open("Reading {%s}..." % (data_files[1]))
        psd_file = np.column_stack(np.loadtxt(data_files[1]))
        lal_cuda.log.comment("Using %d of %d lines." % (n_use, len(psd_file[0])))
        freqs_PSD = psd_file[0][idx_min:idx_max]
        psd = psd_file[1][idx_min:idx_max]
        lal_cuda.log.close("Done.")

        # Confirm that the two data files have the same freq array
        if(not np.array_equal(freqs, freqs_PSD)):
            lal_cuda.log.error("Input data files do not have compatable frequency arrays.")

        # Initialize buffer
        buf = None
        if(not legacy and use_buffer):
            lal_cuda.log.open("Allocating buffer...")
            buf = lalsimulation.PhenomPCore_buffer(int(len(freqs)), n_streams)
            lal_cuda.log.close("Done.")

        n_dim = 1

        # np.random.uniform(low=mc_min, high=mc_max, size=n_walkers)
        p0 = [[np.random.uniform(13, 40)] for i in range(n_walkers)]

        # Initialize sampler
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, logprob, args=[data, psd, freqs, buf, legacy])

        # Generate chain, printing a progress bar as it goes
        lal_cuda.log.open("Generating chain...")
        lal_cuda.log.progress_bar(sampler.sample, n_steps, p0, iterations=n_steps)
        lal_cuda.log.close("Done.")

        # Clean-up buffer
        if(buf):
            lal_cuda.log.open("Freeing buffer...")
            lalsimulation.free_PhenomPCore_buffer(buf)
            lal_cuda.log.close("Done.")

        # Save chain
        lal_cuda.log.open("Saving chains to {%s}..." % (filename_out))
        with open(filename_out, "wb") as file_out:
            pickle.dump(sampler.flatchain, file_out)
        lal_cuda.log.close("Done.", time_elapsed=True)

        lal_cuda.log.close("Done.", time_elapsed=True)


# Permit script execution
if __name__ == '__main__':
    status = PhenomPCore_mcmc()
    sys.exit(status)
