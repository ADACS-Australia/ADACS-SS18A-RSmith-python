import glob
import pytest
import math
import os

import lalsimulation
import lal_cuda
import lal_cuda.SimIMRPhenomP as model

# Set this to True if you want to run on a pre-GPU version of lalsimulation
legacy = False


def check_against_reference(use_buffer, n_streams, filename_ref_inputs):

    # Initialise inputs for run
    inputs_i = model.inputs.read(filename_ref_inputs)

    # Create buffer (if needed)
    if(use_buffer and not legacy):
        buf = lalsimulation.PhenomPCore_buffer(inputs_i.n_freqs, n_streams)
    else:
        buf = None

    # Perform multiple iterations because there can be
    # noise in async GPU implementations and because the
    # buffer can be compromised in successive calls, etc.
    n_check = 3

    # If buf is being used, check that multiple calls are identical
    flag_check = False
    errors = []
    for i_check in range(n_check):
        # Perform run
        outputs_i = inputs_i.run(buf=buf, legacy=legacy)

        # Calculate *fractional* difference from stored reference
        diff_i = model.calc_difference_from_reference(inputs_i, outputs_i, verbose=False)

        # Perform tests
        tolerance = 1e-6
        if math.fabs(diff_i['hpval_real_diff_max']) > tolerance:
            errors.append("%s: hpval_real_diff_max=%le > %le" %
                          (os.path.basename(filename_ref_inputs), diff_i['hpval_real_diff_max'], tolerance))
            flag_check = True
        if math.fabs(diff_i['hpval_imag_diff_max']) > tolerance:
            errors.append("%s: hpval_imag_diff_max=%le > %le" %
                          (os.path.basename(filename_ref_inputs), diff_i['hpval_imag_diff_max'], tolerance))
            flag_check = True
        if math.fabs(diff_i['hcval_real_diff_max']) > tolerance:
            errors.append("%s: hcval_real_diff_max=%le > %le" %
                          (os.path.basename(filename_ref_inputs), diff_i['hcval_real_diff_max'], tolerance))
            flag_check = True
        if math.fabs(diff_i['hcval_imag_diff_max']) > tolerance:
            errors.append("%s: hcval_imag_diff_max=%le > %le" %
                          (os.path.basename(filename_ref_inputs), diff_i['hcval_imag_diff_max'], tolerance))
            flag_check = True

    # Clean-up buffer (if used)
    if(buf):
        lalsimulation.free_PhenomPCore_buffer(buf)
        buf = None

    # assert no error message has been registered, else print messages
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

# *** Set the grid of tests to perform ***


# ... n_streams ...
if(legacy):
    n_streams_grid_test_SimIMRPhenomPFrequencySequence = [None]
else:
    n_streams_grid_test_SimIMRPhenomPFrequencySequence = [0, 1, 8, 16]

# ... use_buffer ...
if(legacy):
    use_buffer_grid_test_SimIMRPhenomPFrequencySequence = [False]
else:
    use_buffer_grid_test_SimIMRPhenomPFrequencySequence = [False, True]

# ... reference inputs ...
filename_ref_inputs_grid_test_SimIMRPhenomPFrequencySequence = glob.glob(lal_cuda.full_path_datafile("inputs.dat*"))

# *** Perform tests ***


@pytest.mark.parametrize("use_buffer", use_buffer_grid_test_SimIMRPhenomPFrequencySequence)
@pytest.mark.parametrize("n_streams", n_streams_grid_test_SimIMRPhenomPFrequencySequence)
@pytest.mark.parametrize("filename_ref_inputs", filename_ref_inputs_grid_test_SimIMRPhenomPFrequencySequence)
def test_ref_inputs(use_buffer, n_streams, filename_ref_inputs):
    check_against_reference(use_buffer=use_buffer, n_streams=n_streams, filename_ref_inputs=filename_ref_inputs)
