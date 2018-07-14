import glob
import pytest
import math
import os

import lalsimulation
import lal_cuda
import lal_cuda.SimIMRPhenomP as model

# Set this to True if you want to run on a pre-GPU version of lalsimulation
legacy = False


def check_SimIMRPhenomPFrequencySequence(use_buffer):
    # Intitialize an empty list of errors
    errors = []

    # Get a list of reference input files
    filename_ref_inputs = glob.glob(lal_cuda.full_path_datafile("inputs.dat*"))

    # Iterate over all reference inputs
    for filename_ref_input_i in filename_ref_inputs:

        # Initialise inputs for run
        inputs_i = model.inputs.read(filename_ref_input_i)

        # Create buffer
        if(use_buffer and not legacy):
            buf = lalsimulation.PhenomPCore_buffer(inputs_i.n_freqs)
            n_check = 3
        else:
            buf = None
            n_check = 1

        # If buf is being used, check that multiple calls are identical
        flag_check = False
        for i_check in range(n_check):
            # Perform run
            outputs_i = inputs_i.run(buf=buf, legacy=legacy)

            # Calculate difference from stored reference
            diff_i = model.calc_difference_from_reference(inputs_i, outputs_i, verbose=False)

            # Perform tests
            if(i_check == 0):
                tolerance = 1e-6
                if math.fabs(diff_i['hpval_real_diff_max']) > tolerance:
                    errors.append("%s: hpval_real_diff_max=%le > %le" %
                                  (os.path.basename(filename_ref_input_i), diff_i['hpval_real_diff_max'], tolerance))
                    flag_check = True
                if math.fabs(diff_i['hpval_imag_diff_max']) > tolerance:
                    errors.append("%s: hpval_imag_diff_max=%le > %le" %
                                  (os.path.basename(filename_ref_input_i), diff_i['hpval_imag_diff_max'], tolerance))
                    flag_check = True
                if math.fabs(diff_i['hcval_real_diff_max']) > tolerance:
                    errors.append("%s: hcval_real_diff_max=%le > %le" %
                                  (os.path.basename(filename_ref_input_i), diff_i['hcval_real_diff_max'], tolerance))
                    flag_check = True
                if math.fabs(diff_i['hcval_imag_diff_max']) > tolerance:
                    errors.append("%s: hcval_imag_diff_max=%le > %le" %
                                  (os.path.basename(filename_ref_input_i), diff_i['hcval_imag_diff_max'], tolerance))
                    flag_check = True
                outputs_0 = outputs_i
            else:
                if(outputs_i != outputs_0):
                    errors.append("%s: outputs_i!=outputs_0 for i=%d" % (inputs_i, i_check))
                    flag_check = True

            # Stop looping if an error occurs for this parameter set
            if(flag_check):
                break

        # Clean-up buffer
        if(buf):
            lalsimulation.free_PhenomPCore_buffer(buf)
            buf = None

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))


def test_SimIMRPhenomPFrequencySequence_without_buffer():
    check_SimIMRPhenomPFrequencySequence(use_buffer=False)


if(not legacy):
    def test_SimIMRPhenomPFrequencySequence_with_buffer():
        check_SimIMRPhenomPFrequencySequence(use_buffer=True)
