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

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('--n_timing', default=0)
@click.option('--n_freq', default=0)
@click.option('--write2stdout/--no-write2stdout', default=False)
@click.option('--write2bin/--no-write2bin', default=False)
def PhenomPCore(n_timing,n_freq,write2stdout,write2bin):
    """This script calls a higher-level function in LALSuite.  The output is two
    binary arrays corresponding to the two outputs hp_val, hc_val from
    PhenomPCore. The outputs are arrays of complex numbers; one for each
    frequency bin. 

    The call:

    lalsimulation.SimIMRPhenomPFrequencySequence(...)

    calls the C-function XLALSimIMRPhenomPFrequencySequence.  That function then calls PhenomPCore which in-turn calls PhenomPCoreOneFrequency.  The arguments that are passed to PhenomPCoreOneFrequency:

    (f, eta, chi1_l, chi2_l, chip, distance, M, phic, pAmp, pPhi, PCparams, pn, &angcoeffs, &Y2m, alphaNNLOoffset - alpha0, epsilonNNLOoffset, &hp_val, &hc_val, &phasing, IMRPhenomP_version, &amp_prefactors, &phi_prefactors)

    are derived from the ones passed to XLALSimIMRPhenomPFrequencySequency

        :return: None

    """

    chi1_l, chi2_l, m1, m2, chip, thetaJ, alpha0, distance_PC, phic, fref = 0.1, 0.2, 30, 30, 0.34, 1.1, 1.5, 1000, np.pi * 0.4, 30

    m1SI = m1 * lal.lal.MSUN_SI
    m2SI = m2 * lal.lal.MSUN_SI
    distance = distance_PC*lal.lal.PC_SI*100*1e6

    flow, fhigh = 20, 100

    if(n_freq<1):
        freqs = np.linspace(flow, fhigh, (fhigh - flow) + 1)
        n_freq=len(freqs)
    else:
        freqs = np.linspace(flow, fhigh, n_freq)

    # Initialize buffer
    #buf = lalsimulation.PhenomPCore_buffer_alloc(n_freq)
    buf = None

    # Perform timing test if required
    if(n_timing>0):
        # Create a callable for timing purposes
        t = timeit.Timer(lambda: lalsimulation.SimIMRPhenomPFrequencySequence(
                freqs,
                chi1_l,
                chi2_l,
                chip,
                thetaJ,
                m1SI,
                m2SI,
                distance,
                alpha0,
                phic,
                fref,
                1,
                buf,
                None
            ))
        # Burning first call (to avoid CUDA context establishment, etc)
        print("Burning first call: %f seconds."%(t.timeit(number=1)))
        # Print timing results
        print("Average timing of %d subsequent calls: %.5f seconds."%(n_timing,t.timeit(number=n_timing)/float(n_timing)))

    # A Numpy array holding the input parameters
    inputs_runtime = np.array([chi1_l,chi2_l,chip,thetaJ,m1SI,m2SI,distance,alpha0,phic,fref])

    # Perform runtime call
    H=lalsimulation.SimIMRPhenomPFrequencySequence(
        freqs,
        chi1_l,
        chi2_l,
        chip,
        thetaJ,
        m1SI,
        m2SI,
        distance,
        alpha0,
        phic,
        fref,
        1,
        buf,
        None)
    hp_val = H[0].data.data
    hc_val = H[1].data.data

    # Clean-up the buffer
    lalsimulation.PhenomPCore_buffer_free(buf)

    # This flag declares whether the reference test dataset's inputs are the same as the run-time dataset
    flag_valid_ref = True

    # Load test reference dataset's inputs
    inputs_file = open(lal_cuda.full_path_datafile("inputs.dat"), "rb")
    inputs_ref=np.fromfile(inputs_file,dtype=inputs_runtime.dtype,count=len(inputs_runtime))
    freqs_ref=np.fromfile(inputs_file,dtype=freqs.dtype,count=len(freqs))
    inputs_file.close()

    # Test if the reference test dataset is the same as the run-time dataset
    if(not np.array_equal(inputs_ref,inputs_runtime)):
        flag_valid_ref = False
    if(not np.array_equal(freqs_ref,freqs)):
        flag_valid_ref = False

    # Run test if we are running with the same inputs as the reference dataset
    if(not flag_valid_ref):
        print('Inputs differ from reference ... skipping test.')
    else:
        print('Performing test...')

        # Read reference dataset's outputs
        outputs_file = open(lal_cuda.full_path_datafile("outputs.dat"), "rb")
        hp_ref=np.fromfile(outputs_file,dtype=hp_val.dtype,count=n_freq)
        hc_ref=np.fromfile(outputs_file,dtype=hc_val.dtype,count=n_freq)
        outputs_file.close()

        # Compute statistics of difference from test reference
        hpval_real_diff_avg = 0.
        hpval_imag_diff_avg = 0.
        hcval_real_diff_avg = 0.
        hcval_imag_diff_avg = 0.
        hpval_real_diff_max =-1e10 
        hpval_imag_diff_max =-1e10 
        hcval_real_diff_max =-1e10 
        hcval_imag_diff_max =-1e10 
        for (hp_val_i,hc_val_i,hp_ref_i,hc_ref_i) in zip(hp_val,hc_val,hp_ref,hc_ref):
            hpval_real_diff_i = math.fabs((hp_val_i.real-hp_ref_i.real)/hp_val_i.real)
            hpval_imag_diff_i = math.fabs((hp_val_i.imag-hp_ref_i.imag)/hp_val_i.imag)
            hcval_real_diff_i = math.fabs((hc_val_i.real-hc_ref_i.real)/hc_val_i.real)
            hcval_imag_diff_i = math.fabs((hc_val_i.imag-hc_ref_i.imag)/hc_val_i.imag)
            hpval_real_diff_avg += hpval_real_diff_i
            hpval_imag_diff_avg += hpval_imag_diff_i
            hcval_real_diff_avg += hcval_real_diff_i
            hcval_imag_diff_avg += hcval_imag_diff_i
            hpval_real_diff_max = max([hpval_real_diff_max,hpval_real_diff_i])
            hpval_imag_diff_max = max([hpval_imag_diff_max,hpval_imag_diff_i])
            hcval_real_diff_max = max([hcval_real_diff_max,hcval_real_diff_i])
            hcval_imag_diff_max = max([hcval_imag_diff_max,hcval_imag_diff_i])
        hpval_real_diff_avg /= float(len(hp_val)) 
        hpval_imag_diff_avg /= float(len(hp_val)) 
        hpval_real_diff_avg /= float(len(hp_val)) 
        hpval_imag_diff_avg /= float(len(hp_val)) 

        # Report results
        print('   Average/maximum real(hp_val) fractional difference: %.2e/%.2e'%(hpval_real_diff_avg,hpval_real_diff_max))
        print('   Average/maximum imag(hp_val) fractional difference: %.2e/%.2e'%(hpval_imag_diff_avg,hpval_imag_diff_max))
        print('   Average/maximum real(hc_val) fractional difference: %.2e/%.2e'%(hcval_real_diff_avg,hcval_real_diff_max))
        print('   Average/maximum imag(hx_val) fractional difference: %.2e/%.2e'%(hcval_imag_diff_avg,hcval_imag_diff_max))

        print("Done.")

    # Write results to screen
    if(write2stdout):
        for [f_i,hp_i,hc_i] in zip(freqs,hp_val,hc_val):
            print(f_i,hp_i.real,hp_i.imag,hc_i.real,hc_i.imag)

    # Write to binary file
    if(write2bin):
        # Write inputs
        print("Writting inputs to 'inputs.dat'...",end='')
        inputs_file  = open("./inputs.dat", "wb")
        inputs_runtime.tofile(inputs_file)
        freqs.tofile(inputs_file)
        inputs_file.close()
        print("Done.")

        # Write results
        print("Writting outputs to 'outputs.dat'...",end='')
        outputs_file = open("./outputs.dat", "wb")
        hp_val.tofile(outputs_file)
        hc_val.tofile(outputs_file)
        outputs_file.close()
        print("Done.")

# Permit script execution
if __name__ == '__main__':
    status = lal_cuda_params()
    sys.exit(status)
