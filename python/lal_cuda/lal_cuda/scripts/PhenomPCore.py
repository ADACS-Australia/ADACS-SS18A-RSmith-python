from __future__ import print_function
import numpy as np
import pylab as plt
import os
import sys
import click
import timeit
import math

import lal
import lalsimulation
import lal_cuda

def write_results(inputs_runtime,freqs,hp_val,hc_val,write2stdout,write2bin,filename_label=None):
    # Write results to screen
    if(write2stdout):
        for [f_i,hp_i,hc_i] in zip(freqs,hp_val,hc_val):
            print(f_i,hp_i.real,hp_i.imag,hc_i.real,hc_i.imag)

    # Write to binary file
    if(write2bin):

        # Write inputs
        if(filename_label):
            filename_inputs_out  = "inputs_%s.dat"%(filename_label)
            filename_outputs_out = "outputs_%s.dat"%(filename_label)
        else:
            filename_inputs_out  = "inputs.dat"
            filename_outputs_out = "outputs.dat"

        print("Writing inputs to '%s'..."%(filename_inputs_out),end='')
        inputs_file  = open(filename_inputs_out, "wb")
        inputs_runtime.tofile(inputs_file)
        freqs.tofile(inputs_file)
        inputs_file.close()
        print("Done.")

        # Write results
        print("Writing outputs to '%s'..."%(filename_outputs_out),end='')
        outputs_file = open(filename_outputs_out, "wb")
        hp_val.tofile(outputs_file)
        hc_val.tofile(outputs_file)
        outputs_file.close()
        print("Done.")

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--n_avg',   type=int,default=0,show_default=True)
@click.option('--n_freq',  type=int,default=0,show_default=True)
@click.option('--scan',    type=(float,float,int), default=(None,None,None),show_default=True)
@click.option('--chi1',    type=float,default=0.1,show_default=True)
@click.option('--chi2',    type=float,default=0.2,show_default=True)
@click.option('--m1',      type=float,default=30,show_default=True)
@click.option('--m2',      type=float,default=30,show_default=True)
@click.option('--chip',    type=float,default=0.34,show_default=True)
@click.option('--thetaJ',  type=float,default=1.1,show_default=True)
@click.option('--alpha0',  type=float,default=1.5,show_default=True)
@click.option('--distance',type=float,default=1000,show_default=True)
@click.option('--phic',    type=float,default=np.pi * 0.4,show_default=True)
@click.option('--fref',    type=float,default=30,show_default=True)
@click.option('--flow',    type=float,default=20,show_default=True)
@click.option('--fhigh',   type=float,default=100,show_default=True)
@click.option('--write2stdout/--no-write2stdout', default=False,show_default=True)
@click.option('--write2bin/--no-write2bin',       default=False,show_default=True)
def PhenomPCore(n_avg, n_freq, scan, chi1, chi2, m1, m2, chip, thetaj, alpha0, distance, phic, fref, flow, fhigh, write2stdout, write2bin):
    """This script calls a higher-level function in LALSuite.  The output is two
    binary arrays corresponding to the two outputs hp_val, hc_val from PhenomPCore.

    The outputs are arrays of complex numbers; one for each frequency bin.

    The call:

    lalsimulation.SimIMRPhenomPFrequencySequence(...)

    calls the C-function XLALSimIMRPhenomPFrequencySequence.  That function then calls PhenomPCore which in-turn calls PhenomPCoreOneFrequency.

    """

    # Check that cmdl options 'scan' and 'n_freq' have not both been specified
    # and that 'n_avg'>1 is given if 'scan' is
    if(scan[0]!=None):
        if(n_freq!=0):
            print("Command line options 'scan' and 'n_freq' can not both be specified.")
            exit(1)
        elif(n_avg<=1):
            print("'n_avg' must be >=1 if 'scan' is specified.")
            exit(1)

    # Set the array of n_freq's we are going to run
    if(scan[0]!=None):
        n_freq_lo,n_freq_hi,n_n_freq = scan
    else:
        n_freq_lo = n_freq
        n_freq_hi = n_freq
        n_n_freq  = 1

    # Apply some unit conversions to the input parameters
    m1SI = m1 * lal.lal.MSUN_SI
    m2SI = m2 * lal.lal.MSUN_SI
    distanceSI = distance*lal.lal.PC_SI*100*1e6

    # Generate timing information
    if(n_avg>1):

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
                    chi1,
                    chi2,
                    chip,
                    thetaj,
                    m1SI,
                    m2SI,
                    distanceSI,
                    alpha0,
                    phic,
                    fref,
                    1,
                    buf,
                    None))

            # Burn a number of calls (to avoid Cuda context initialization if buf=None, for example)
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
            chi1,
            chi2,
            chip,
            thetaj,
            m1SI,
            m2SI,
            distanceSI,
            alpha0,
            phic,
            fref,
            1,
            buf,
            None)

        # Create a Numpy array holding the input parameters
        inputs_runtime = np.array([chi1,chi2,chip,thetaj,m1SI,m2SI,distanceSI,alpha0,phic,fref])

        # Create aliases for the ouput arrays
        hp_val = H[0].data.data
        hc_val = H[1].data.data

        # Write results to stdout &/or binary files
        write_results(inputs_runtime,freqs,hp_val,hc_val,write2stdout,write2bin,filename_label=None)

# Permit script execution
if __name__ == '__main__':
    status = lal_cuda_params()
    sys.exit(status)
