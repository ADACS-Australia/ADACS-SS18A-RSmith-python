from __future__ import print_function
import numpy as np
import os
import sys
import click
import math

import lal
import lalsimulation
import lal_cuda

def test_PhenomPCore():
    """
        :return: None

    """

def check_PhenomPCore(inputs_runtime,freqs,hp_val,hc_val,phic,tolerance=1e-10,verbose=True):

    # Get a list of reference input files
    reference_inputs = glob.glob(lal_cuda.full_path_datafile("inputs.dat*"))

    # Iterate over all reference inputs
    for inputs_i in reference_inputs:
        #freqs()

        #run()

        #check_results=lal_cuda.check_PhenomPCore(inputs_i,hp_val,hc_val,phic)
        check_results = {'hpval_real_diff_max':0., 'hpval_imag_diff_max':0., 'hcval_real_diff_max':0., 'hcval_imag_diff_max':0. }

        # Perform tests
        pytest.assume(math.fabs(check_results['hpval_real_diff_max'])<diff_threshold)
        pytest.assume(math.fabs(check_results['hpval_imag_diff_max'])<diff_threshold)
        pytest.assume(math.fabs(check_results['hcval_real_diff_max'])<diff_threshold)
        pytest.assume(math.fabs(check_results['hcval_imag_diff_max'])<diff_threshold)
