from __future__ import print_function
import numpy as np
import sys
import click
import timeit

import lal
import lalsimulation
import lal_cuda
import lal_cuda.SimIMRPhenomP as model

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--flow', type=float, default=20, show_default=True, help='Minimum frequency')
@click.option('--fhigh', type=float, default=100, show_default=True, help='Maximum frequency')
@click.option('--n_freq', type=int, default=81, show_default=True, help='Number of frequencies')
@click.option('--write2stdout/--no-write2stdout', default=True, show_default=True, help='Write to standard out?')
@click.option('--write2bin/--no-write2bin', default=False, show_default=True, help='Write to binary files?')
@click.option('--check/--no-check', default=False, show_default=True,
              help='Check result against stored results for standard parameter sets (if a match is detected).')
@click.option(
    '--timing',
    type=(
        float,
        float,
        int,
        int),
    default=(
        None,
        None,
        None,
        None),
    help='Run in timing mode (all previous parameters are ignored).  Specify run as FREQ_MIN FREQ_MAX N_FREQ N_AVG.')
@click.option('--chi1', type=float, default=0.1, show_default=True, help='Model parameter: chi1')
@click.option('--chi2', type=float, default=0.2, show_default=True, help='Model parameter: chi2')
@click.option('--m1', type=float, default=30, show_default=True, help='Model parameter: m1')
@click.option('--m2', type=float, default=30, show_default=True, help='Model parameter: m2')
@click.option('--chip', type=float, default=0.34, show_default=True, help='Model parameter: chip')
@click.option('--thetaJ', type=float, default=1.1, show_default=True, help='Model parameter: thetaJ')
@click.option('--alpha0', type=float, default=1.5, show_default=True, help='Model parameter: alpha0')
@click.option('--distance', type=float, default=1000, show_default=True,
              help='Model parameter: distance')
@click.option(
    '--phic',
    type=float,
    default=np.pi * 0.4,
    show_default=True,
    help='Model parameter: phic')
@click.option('--fref', type=float, default=30, show_default=True, help='Model parameter: fref')
@click.option('--use_buffer/--no-use_buffer', default=True, show_default=True, help='Use a buffer for accelleration.')
@click.option('--legacy/--no-legacy', default=False, show_default=True,
              help='Specify this option if a legacy version of LALSuite (without buffer support) is being used.')
def PhenomPCore(
        flow,
        fhigh,
        n_freq,
        write2stdout,
        write2bin,
        check,
        timing,
        chi1,
        chi2,
        m1,
        m2,
        chip,
        thetaj,
        alpha0,
        distance,
        phic,
        fref,
        use_buffer,
        legacy):
    """This script calls a higher-level function in LALSuite.  The output is
    two binary arrays corresponding to the two outputs hp_val, hc_val from
    PhenomPCore.

    The outputs are arrays of complex numbers; one for each frequency bin.

    The call:

    lalsimulation.SimIMRPhenomPFrequencySequence(...)

    calls the C-function XLALSimIMRPhenomPFrequencySequence.  That function then calls PhenomPCore which in-turn calls PhenomPCoreOneFrequency.
    """

    # Parse the 'timing' option.  If it is given,
    # then assume that it specifies a range of frequencies
    # to test, the number of frequencies to test, and the
    # number of calls to average results over
    if(timing[0] is not None):
        flag_timing = True
        n_freq_lo, n_freq_hi, n_n_freq, n_avg = timing
    # ... if it isn't given, just perform one run
    else:
        flag_timing = False
        n_freq_lo = n_freq
        n_freq_hi = n_freq
        n_n_freq = 1
        n_avg = 0

    # Generate timing tests
    if(flag_timing):

        # Generate the list of n_freq's that we are going to time
        n_freq_list = [int(10**(log_n_freq_i))
                       for log_n_freq_i in np.linspace(np.log10(n_freq_lo), np.log10(n_freq_hi), n_n_freq)]

        # Generate timing results for each n_freq
        n_burn = 1
        for i_n_freq, n_freq_i in enumerate(n_freq_list):

            # Initialize buffer (saves time for repeated calls)
            if(use_buffer):
                buf = lalsimulation.PhenomPCore_buffer_alloc(int(n_freq_i))
            else:
                buf = None

            # Initialize the model call (apply some unit conversions here)
            lal_inputs = model.inputs(
                chi1=chi1,
                chi2=chi2,
                m1=m1,
                m2=m2,
                chip=chip,
                thetaJ=thetaj,
                alpha0=alpha0,
                distance=distance,
                phic=phic,
                fref=fref,
                freqs=[
                    flow,
                    fhigh,
                    n_freq_i])

            # Create a timing callable
            t = timeit.Timer(lambda: lal_inputs.run(buf, legacy))

            # Burn a number of calls (to avoid contamination from Cuda context initialization if buf=None, for example)
            if(n_burn > 0):
                if(n_burn == 1):
                    lal_cuda.log.comment("Burning a call: %f seconds." % (t.timeit(number=n_burn)))
                else:
                    lal_cuda.log.comment("Burning %d calls: %f seconds." % (n_burn, t.timeit(number=n_burn)))
                n_burn = 0

            # Call the model n_avg times to generate the timing result
            wallclock_i = t.timeit(number=n_avg)

            # Print timing result
            if(len(n_freq_list) == 1):
                lal_cuda.log.comment("Average timing of %d calls: %.5f seconds." % (n_avg, wallclock_i / float(n_avg)))
            else:
                if(i_n_freq == 0):
                    print("# Column 01: Iteration")
                    print("#        02: No. of frequencies")
                    print("#        03: Total time for %d calls [s]" % (n_avg))
                    print("#        04: Avg. time per call [s]")
                print("%3d %8d %10.3le %10.3le" % (i_n_freq, n_freq_i, wallclock_i, wallclock_i / float(n_avg)))

            # Clean-up buffer
            lalsimulation.PhenomPCore_buffer_free(buf)

    # ... if n_avg<=1, then just run the model and exit.
    else:
        # Don't bother with a buffer (saves no time with just one call)
        buf = None

        # Initialize model call
        lal_inputs = model.inputs(
            chi1=chi1,
            chi2=chi2,
            m1=m1,
            m2=m2,
            chip=chip,
            thetaJ=thetaj,
            alpha0=alpha0,
            distance=distance,
            phic=phic,
            fref=fref,
            freqs=[
                flow,
                fhigh,
                n_freq])

        # Perform call
        lal_outputs = lal_inputs.run(buf=buf, legacy=legacy)

        # Write results to stdout &/or binary files
        if(write2bin):
            model.to_binary(lal_inputs, lal_outputs)
        if(write2stdout):
            print(model.to_string(lal_inputs, lal_outputs))

        # Check results against standards (if parameters match)
        if(check):
            model.calc_difference_from_reference(lal_inputs, lal_outputs)


# Permit script execution
if __name__ == '__main__':
    status = PhenomPCore()
    sys.exit(status)
