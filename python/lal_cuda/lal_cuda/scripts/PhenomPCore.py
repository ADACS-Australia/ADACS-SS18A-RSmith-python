from __future__ import print_function
import numpy as np
import sys
import click
import timeit

import lal
import lalsimulation
import lal_cuda.support as support

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--timing_test',type=(float,float,int,int), default=(None,None,None,None))
@click.option('--n_freq',     type=int,  default=81,  show_default=True)
@click.option('--chi1',       type=float,default=0.1, show_default=True)
@click.option('--chi2',       type=float,default=0.2, show_default=True)
@click.option('--m1',         type=float,default=30,  show_default=True)
@click.option('--m2',         type=float,default=30,  show_default=True)
@click.option('--chip',       type=float,default=0.34,show_default=True)
@click.option('--thetaJ',     type=float,default=1.1, show_default=True)
@click.option('--alpha0',     type=float,default=1.5, show_default=True)
@click.option('--distance',   type=float,default=1000,show_default=True)
@click.option('--phic',       type=float,default=np.pi*0.4,show_default=True)
@click.option('--fref',       type=float,default=30,       show_default=True)
@click.option('--flow',       type=float,default=20,       show_default=True)
@click.option('--fhigh',      type=float,default=100,      show_default=True)
@click.option('--write2stdout/--no-write2stdout', default=True, show_default=True)
@click.option('--write2bin/--no-write2bin',       default=False,show_default=True)
@click.option('--check/--no-check',               default=False,show_default=True)
def PhenomPCore(timing_test, n_freq, chi1, chi2, m1, m2, chip, thetaj, alpha0, distance, phic, fref, flow, fhigh, write2stdout, write2bin, check):
    """This script calls a higher-level function in LALSuite.  The output is two
    binary arrays corresponding to the two outputs hp_val, hc_val from PhenomPCore.

    The outputs are arrays of complex numbers; one for each frequency bin.

    The call:

    lalsimulation.SimIMRPhenomPFrequencySequence(...)

    calls the C-function XLALSimIMRPhenomPFrequencySequence.  That function then calls PhenomPCore which in-turn calls PhenomPCoreOneFrequency.

    """

    # Parse the 'timing_test' option, if it is given
    if(timing_test[0]!=None):
        flag_timing_test = True
        n_freq_lo,n_freq_hi,n_n_freq,n_avg = timing_test
    else:
        flag_timing_test = False
        n_freq_lo = n_freq
        n_freq_hi = n_freq
        n_n_freq  = 1
        n_avg     = 0

    # Apply some unit conversions to the input parameters
    lal_inputs = support.PhenomPCore_inputs(chi1=chi1, chi2=chi2, m1=m1, m2=m2, chip=chip, thetaJ=thetaj, alpha0=alpha0, distance=distance, phic=phic, fref=fref)

    # Generate timing information
    if(flag_timing_test):

        # Initialize buffer (saves time with repeated calls)
        buf = lalsimulation.PhenomPCore_buffer_alloc(n_freq)

        # Generate the list of n_freq's that we are going to time
        n_freq_list = [10**(log_n_freq_i) for log_n_freq_i in np.linspace(np.log10(n_freq_lo), np.log10(n_freq_hi), n_n_freq)]

        # Generate timing results for each n_freq
        n_burn=1
        for i_n_freq,n_freq_i in enumerate(n_freq_list):

            # Generate frequency array for this iteration
            if(n_freq_i<1):
                freqs = np.linspace(flow, fhigh, (fhigh - flow) + 1)
                n_freq_i=len(freqs)
            else:
                freqs = np.linspace(flow, fhigh, n_freq_i)
        
            # Create a timing callable
            t = timeit.Timer(lambda: lalsimulation.SimIMRPhenomPFrequencySequence(
                    freqs,
                    lal_inputs.chi1,
                    lal_inputs.chi2,
                    lal_inputs.chip,
                    lal_inputs.thetaJ,
                    lal_inputs.m1,
                    lal_inputs.m2,
                    lal_inputs.distance,
                    lal_inputs.alpha0,
                    lal_inputs.phic,
                    lal_inputs.fref,
                    1,
                    buf,
                    None))

            # Burn a number of calls (to avoid contamination from Cuda context initialization if buf=None, for example)
            if(n_burn>0):
                if(n_burn==1):
                    print("Burning a call: %f seconds."%(t.timeit(number=n_burn)))
                else:
                    print("Burning %d calls: %f seconds."%(n_burn,t.timeit(number=n_burn)))
                n_burn=0

            # Generate timing result
            wallclock_i = t.timeit(number=n_avg)

            # Print timing result
            if(len(n_freq_list)==1):
                print("Average timing of %d calls: %.5f seconds."%(n_avg,wallclock_i/float(n_avg)))
            else:
                if(i_n_freq==0):
                    print("# Column 01: Iteration")
                    print("#        02: No. of frequencies")
                    print("#        03: Total time for %d calls [s]"%(n_avg))
                    print("#        04: Avg. time per call [s]")
                print("%3d %8d %10.3le %10.3le"%(i_n_freq,n_freq_i,wallclock_i,wallclock_i/float(n_avg)))

        # Clean-up
        lalsimulation.PhenomPCore_buffer_free(buf)

    # ... if n_avg<=1, then just run the model and exit.
    else:
        # Don't bother with a buffer (saves no time with just one call)
        buf = None

        # Generate frequency array
        freqs = np.linspace(flow, fhigh, n_freq)

        # Perform runtime call
        H=lalsimulation.SimIMRPhenomPFrequencySequence(
            freqs,
            lal_inputs.chi1,
            lal_inputs.chi2,
            lal_inputs.chip,
            lal_inputs.thetaJ,
            lal_inputs.m1,
            lal_inputs.m2,
            lal_inputs.distance,
            lal_inputs.alpha0,
            lal_inputs.phic,
            lal_inputs.fref,
            1,
            buf,
            None)

        # Create aliases for the ouput arrays
        hp_val = H[0].data.data
        hc_val = H[1].data.data

        # Write results to stdout &/or binary files
        support.write_results_PhenomPCore(lal_inputs,freqs,hp_val,hc_val,phic,write2stdout,write2bin,filename_label=None)

        # Check results against standards (if parameters match)
        if(check):
            support.check_PhenomPCore(lal_inputs,freqs,hp_val,hc_val,phic)

# Permit script execution
if __name__ == '__main__':
    status = lal_cuda_params()
    sys.exit(status)
