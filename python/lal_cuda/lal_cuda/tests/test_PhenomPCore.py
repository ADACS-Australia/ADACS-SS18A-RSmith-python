from __future__ import print_function
import numpy as np
import pylab as plt
import os
import sys
import click
import math

def test_PhenomPCore():
    """
        :return: None

    """

    chi1_l, chi2_l, m1, m2, chip, thetaJ, alpha0, distance_PC, phic, fref = 0.1, 0.2, 30, 30, 0.34, 1.1, 1.5, 1000, np.pi * 0.4, 30

    m1SI = m1 * lal.lal.MSUN_SI
    m2SI = m2 * lal.lal.MSUN_SI
    distance = distance_PC*lal.lal.PC_SI*100*1e6

    # A Numpy array holding the input parameters
    inputs_runtime = np.array([chi1_l,chi2_l,chip,thetaJ,m1SI,m2SI,distance,alpha0,phic,fref])

    # Load test reference dataset's inputs
    inputs_file = open(lal_cuda.full_path_datafile("inputs.dat"), "rb")
    inputs_ref=dict(zip(['chi1_l','chi2_l','chip','thetaJ','m1SI','m2SI','distance','alpha0','phic','fref'],np.fromfile(inputs_file,dtype=inputs_runtime.dtype,count=len(inputs_runtime))))
    freqs_ref=np.fromfile(inputs_file,dtype=freqs.dtype,count=len(freqs))
    inputs_file.close()

    # Perform runtime call
    H=lalsimulation.SimIMRPhenomPFrequencySequence(
        inputs_ref('freqs'),
        inputs_ref('chi1_l'),
        inputs_ref('chi2_l'),
        inputs_ref('chip'),
        inputs_ref('thetaJ'),
        inputs_ref('m1SI'),
        inputs_ref('m2SI'),
        inputs_ref('distance'),
        inputs_ref('alpha0'),
        inputs_ref('phic'),
        inputs_ref('fref'),
        1,
        None)
    hp_val = H[0].data.data
    hc_val = H[1].data.data

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

    # Perform tests
    diff_threshold = 1e-10
    pytest.assume(math.fabs(hpval_real_diff_max)<diff_threshold)
    pytest.assume(math.fabs(hpval_imag_diff_max)<diff_threshold)
    pytest.assume(math.fabs(hcval_real_diff_max)<diff_threshold)
    pytest.assume(math.fabs(hcval_imag_diff_max)<diff_threshold)
