"""This submodule handles all interaction with the LALSuite C API.

It provides a class for inputs and a class for outputs and functions for
reading/writing/comparing them.  The inputs method has a 'run' method
for calling the C API.
"""

import numpy as np
import glob
import math

# Import the C API
import lal
import lalsimulation

import lal_cuda

# Define a set of default model parameters.  These are the ones in the
# 'default' fiducial inputs file in the 'data' directory.
chi1_default = 0.1
chi2_default = 0.2
m1_default = 30
m2_default = 30
chip_default = 0.34
thetaJ_default = 1.1
alpha0_default = 1.5
distance_default = 1000
phic_default = np.pi * 0.4
fref_default = 30
flow_default = 20
fhigh_default = 80


def to_string(inputs, outputs):
    """Convert a paired set of inputs/outputs to an ASCII table.

    :param inputs: An instance of the inputs class.
    :param outputs: An associated instance of the outputs class.
    :return: A list of strings.
    """
    r_val = '# Column 01: frequency\n'
    r_val += '#        02: hp - real\n'
    r_val += '#        03: hp - imagingary\n'
    r_val += '#        04: hc - real\n'
    r_val += '#        05: hc - imaginary\n'
    for f_i, hp_i, hc_i in zip(inputs.freqs, outputs.hp, outputs.hc):
        r_val += "%8.2f %12.5e %12.5e %12.5e %12.5e\n" % (f_i, hp_i.real, hp_i.imag, hc_i.real, hc_i.imag)
    return r_val


def to_binary(inputs, outputs, filename_label=None):
    """Convert a paired set of inputs/outputs to binary files.

    :param inputs: An instance of the inputs class.
    :param outputs: An associated instance of the outputs class.
    :param filename_label: An optional label for the output files.
    :return: None
    """
    inputs.write(filename_label, filename_label=filename_label)
    outputs.write(filename_label, filename_label=filename_label)


def calc_frac_diff(x, y):
    """Calculate the fractional difference between two numbers.

    If the reference value is 0 and the value to check is not, returns 1.

    :param x: Value to check
    :param y: Reference value
    :return: Value
    """
    if(y != 0):
        return math.fabs((x - y) / y)
    elif(x != 0.):
        return 1.
    else:
        return 0.


def calc_difference_from_reference(inputs, outputs, verbose=True):
    """Look for a match between the given inputs and the stored reference
    inputs.  If found return a dictionary with absolute differences from the
    given outputs, otherwise print a warning to stdout if verbose=True.

    All reference inputs/outputs were computed by running the
    PhenomPCore script of this package against a version of LALSuite
    compiled from the commit with hash "3494e18e6d".

    :param inputs: An instance of the inputs class
    :param outputs: An associated instance of the outputs class
    :param verbose: Boolean controlling whether logging information is reported
    :return: None
    """

    # Get a list of reference input/output files
    filename_ref_inputs = glob.glob(lal_cuda.full_path_datafile("inputs.dat*"))
    filename_ref_outputs = [
        filename_ref_input_i.replace(
            "inputs.dat",
            "outputs.dat") for filename_ref_input_i in filename_ref_inputs]

    # Look to see if the given inputs are in the stored reference inputs
    filename_ref_output = None
    for filename_ref_input_i, filename_ref_output_i in zip(filename_ref_inputs, filename_ref_outputs):
        inputs_i = inputs.read(filename_ref_input_i)

        # Check to see if this set of inputs matches the set that has been passed
        if(inputs_i == inputs):
            inputs_ref = inputs_i
            filename_ref_output = filename_ref_output_i
            break

    # Perform check if a match has been found
    if(not filename_ref_output):
        lal_cuda.log.warning(
            "Checking could not be performed: reference data set with given inputs (%s) not found." %
            (inputs))
    else:
        if(verbose):
            lal_cuda.log.open('Performing test...')

        # Read reference dataset's outputs
        outputs_ref = outputs.read(filename_ref_output)

        # Compute statistics of difference from test reference
        hpval_real_diff_avg = 0.
        hpval_imag_diff_avg = 0.
        hcval_real_diff_avg = 0.
        hcval_imag_diff_avg = 0.
        hpval_real_diff_max = 0.
        hpval_imag_diff_max = 0.
        hcval_real_diff_max = 0.
        hcval_imag_diff_max = 0.
        for (hp_i, hc_i, hp_ref_i, hc_ref_i) in zip(outputs.hp, outputs.hc, outputs_ref.hp, outputs_ref.hc):
            hpval_real_diff_i = calc_frac_diff(hp_i.real, hp_ref_i.real)
            hpval_imag_diff_i = calc_frac_diff(hp_i.imag, hp_ref_i.imag)
            hcval_real_diff_i = calc_frac_diff(hc_i.real, hc_ref_i.real)
            hcval_imag_diff_i = calc_frac_diff(hc_i.imag, hc_ref_i.imag)
            hpval_real_diff_avg += hpval_real_diff_i
            hpval_imag_diff_avg += hpval_imag_diff_i
            hcval_real_diff_avg += hcval_real_diff_i
            hcval_imag_diff_avg += hcval_imag_diff_i
            hpval_real_diff_max = max([hpval_real_diff_max, hpval_real_diff_i])
            hpval_imag_diff_max = max([hpval_imag_diff_max, hpval_imag_diff_i])
            hcval_real_diff_max = max([hcval_real_diff_max, hcval_real_diff_i])
            hcval_imag_diff_max = max([hcval_imag_diff_max, hcval_imag_diff_i])
        hpval_real_diff_avg /= float(len(outputs.hp))
        hpval_imag_diff_avg /= float(len(outputs.hp))
        hcval_real_diff_avg /= float(len(outputs.hc))
        hcval_imag_diff_avg /= float(len(outputs.hc))

        # Report results
        if(verbose):
            lal_cuda.log.comment('   Average/maximum real(hp) fractional difference: %.2e/%.2e' %
                                 (hpval_real_diff_avg, hpval_real_diff_max))
            lal_cuda.log.comment('   Average/maximum imag(hp) fractional difference: %.2e/%.2e' %
                                 (hpval_imag_diff_avg, hpval_imag_diff_max))
            lal_cuda.log.comment('   Average/maximum real(hc) fractional difference: %.2e/%.2e' %
                                 (hcval_real_diff_avg, hcval_real_diff_max))
            lal_cuda.log.comment('   Average/maximum imag(hc) fractional difference: %.2e/%.2e' %
                                 (hcval_imag_diff_avg, hcval_imag_diff_max))
            lal_cuda.log.close("Done.")

        return {
            'hpval_real_diff_avg': hpval_real_diff_avg,
            'hpval_real_diff_max': hpval_real_diff_max,
            'hpval_imag_diff_avg': hpval_imag_diff_avg,
            'hpval_imag_diff_max': hpval_imag_diff_max,
            'hcval_real_diff_avg': hcval_real_diff_avg,
            'hcval_real_diff_max': hcval_real_diff_max,
            'hcval_imag_diff_avg': hcval_imag_diff_avg,
            'hcval_imag_diff_max': hcval_imag_diff_max}


class outputs(object):
    """This class manages the output (hp and hc complex arrays) from a LALSUite
    model call.

    An instance can be created using the default constructor or the
    :func:`read` method.  An instance can be written using the
    :func:`write` method.  Equivalence of two instances is defined by
    the element-wise equivilance of their hp and hc arrays.
    """

    def __init__(self, return_from_SimIMRPhenomPFrequencySequence=None, hp=None, hc=None):
        """Create an instance of the outputs class.  Optionally pass complex
        arrays hp and hc to initialize from.

        :param return_from_SimIMRPhenomPFrequencySequence: The data structure returned from the LALSUite C API.
        :param hp: Complex floating point array
        :param hc: Complex floating point array
        """
        if((isinstance(hp, np.ndarray)) and (isinstance(hc, np.ndarray)) and (isinstance(return_from_SimIMRPhenomPFrequencySequence, type(None)))):
            self.hp = hp
            self.hc = hc
        elif((isinstance(hp, type(None))) and (isinstance(hc, type(None))) and not (isinstance(return_from_SimIMRPhenomPFrequencySequence, type(None)))):
            self.hp = return_from_SimIMRPhenomPFrequencySequence[0].data.data
            self.hc = return_from_SimIMRPhenomPFrequencySequence[1].data.data
        else:
            lal_cuda.log.error("Invalid inputs to SimIMRPhenomPFrequencySequence outputs constructor.")
            exit(1)

    @classmethod
    def read(cls, filename_datafile_in):
        """Create an instance of the outputs class from a binary file.

        :param filename_datafile_in: Filename to read from.
        :return: An instance of the outputs class.
        """
        with open(lal_cuda.full_path_datafile(filename_datafile_in), "rb") as outputs_file:
            n_freqs = np.asscalar(np.fromfile(outputs_file, dtype=np.int32, count=1))
            hp = np.fromfile(outputs_file, dtype=np.complex128, count=n_freqs)
            hc = np.fromfile(outputs_file, dtype=np.complex128, count=n_freqs)
        return(cls(hp=hp, hc=hc))

    def write(self, filename_outputs_out, filename_label=None, verbose=True):
        """Write the instance of the output class to a binary file.

        :param filename_outputs_out: Filename to write to.
        :param filename_label: Filename modifier.
        :param verbose: Boolean flag indicating whether to write activity to the log.
        :return: None
        """
        # Set filename
        if(filename_label):
            filename_outputs_out = "outputs.dat." + filename_label
        else:
            filename_outputs_out = "outputs.dat"

        if(verbose):
            lal_cuda.log.open("Writing outputs to '%s'..." % (filename_outputs_out), end='')
        with open(filename_outputs_out, "wb") as outputs_file:
            np.array([len(self.hp)], dtype=np.int32).tofile(outputs_file)
            self.hp.tofile(outputs_file)
            self.hc.tofile(outputs_file)
        if(verbose):
            lal_cuda.log.close("Done.")

    def __eq__(self, other):
        """Test for equivilance of two sets of outputs."""
        return np.array_equal(self.hp, other.hp) and np.array_equal(self.hc, other.hc)

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)


class inputs(object):

    def __init__(
            self,
            chi1=chi1_default,
            chi2=chi2_default,
            m1=m1_default,
            m2=m2_default,
            chip=chip_default,
            thetaJ=thetaJ_default,
            alpha0=alpha0_default,
            distance=distance_default,
            phic=phic_default,
            fref=fref_default,
            mode=1,
            freqs=[
                flow_default,
                fhigh_default,
                -1],
            freqs_from_range=True,
            convert_units=True):
        """Create an instance of the inputs class, for a given set of model
        parameters.

        :param chi1: See LALSuite documentation for a description of this model parameter.
        :param chi2: See LALSuite documentation for a description of this model parameter.
        :param m1: See LALSuite documentation for a description of this model parameter.
        :param m2: See LALSuite documentation for a description of this model parameter.
        :param chip: See LALSuite documentation for a description of this model parameter.
        :param thetaJ: See LALSuite documentation for a description of this model parameter.
        :param alpha0: See LALSuite documentation for a description of this model parameter.
        :param distance: See LALSuite documentation for a description of this model parameter.
        :param phic: See LALSuite documentation for a description of this model parameter.
        :param fref: See LALSuite documentation for a description of this model parameter.
        :param mode: See LALSuite documentation for a description of this model parameter.
        :param freqs: Frequency array (either an element-wise array, or a 3-element description of range)
        :param freqs_from_range: Set to True if the freqs describes a range, rather than an element-wise array
        :param convert_units:
        """
        self.chi1 = chi1
        self.chi2 = chi2
        self.m1 = m1
        self.m2 = m2
        self.distance = distance
        self.thetaJ = thetaJ
        self.alpha0 = alpha0
        self.chip = chip
        self.phic = phic
        self.fref = fref
        self.mode = mode

        # Perform unit conversions, if requested
        if(convert_units):
            self.m1 = self.m1 * lal.lal.MSUN_SI
            self.m2 = self.m2 * lal.lal.MSUN_SI
            self.distance = self.distance * lal.lal.PC_SI * 100 * 1e6

        # Generate frequency array
        if(freqs_from_range):
            flow = freqs[0]
            fhigh = freqs[1]
            n_freqs = freqs[2]
            # If n_freqs<1, then assum dfreq=1.
            if(n_freqs < 1):
                self.freqs = np.linspace(flow, fhigh, (fhigh - flow) + 1)
                self.n_freqs = len(self.freqs)
            else:
                self.n_freqs = n_freqs
                self.freqs = np.linspace(flow, fhigh, self.n_freqs)
        # If freqs_from_range is false, then assume that freqs specifies a list of frequencies
        else:
            self.freqs = freqs
            self.n_freqs = len(self.freqs)

    def np_floats(self):
        """A numpy array of all floating-point inputs.

        :return: A numpy array of floats.
        """
        # A numpy-array packaging of the floating-point input parameters
        return np.array([self.chi1, self.chi2, self.chip, self.thetaJ, self.m1, self.m2,
                         self.distance, self.alpha0, self.phic, self.fref], dtype=np.float64)

    def np_ints(self):
        """A numpy array of all integer inputs.

        :return: A numpy array of integers.
        """
        # A numpy-array packaging of the integer input parameters
        return np.array([self.mode, self.n_freqs], dtype=np.int32)

    @classmethod
    def read(cls, filename_datafile_in):
        """Create an instance of a inputs method from a binary file.

        :param filename_datafile_in: Filename storing inputs.
        :return: A object of class inputs
        """
        with open(lal_cuda.full_path_datafile(filename_datafile_in), "rb") as inputs_file:
            # Read floating-point parameters
            inputs_np_floats = np.fromfile(inputs_file, dtype=np.float64, count=len(cls().np_floats()))
            chi1 = inputs_np_floats[0]
            chi2 = inputs_np_floats[1]
            chip = inputs_np_floats[2]
            thetaJ = inputs_np_floats[3]
            m1 = inputs_np_floats[4]
            m2 = inputs_np_floats[5]
            distance = inputs_np_floats[6]
            alpha0 = inputs_np_floats[7]
            phic = inputs_np_floats[8]
            fref = inputs_np_floats[9]
            # Read integer-type parameters
            inputs_np_ints = np.fromfile(inputs_file, dtype=np.int32, count=len(cls().np_ints()))
            mode = int(inputs_np_ints[0])
            n_freqs = int(inputs_np_ints[1])
            # Read frequency array
            freqs = np.fromfile(inputs_file, dtype=np.float64, count=n_freqs)
        return(cls(chi1=chi1, chi2=chi2, m1=m1, m2=m2, chip=chip, thetaJ=thetaJ, alpha0=alpha0, distance=distance, phic=phic, fref=fref, mode=mode, freqs=freqs, freqs_from_range=False, convert_units=False))

    def write(self, filename_inputs_out, filename_label=None, verbose=True):
        """Write an instance of an object of class inputs to a binary file.

        :param filename_inputs_out: Filename to write to
        :param filename_label: Filename modifier.
        :param verbose: Boolean flag indicating whether to write activity to the log.
        :return:
        """
        # Set filename
        if(filename_label):
            filename_inputs_out = "inputs.dat." + filename_label
        else:
            filename_inputs_out = "inputs.dat"

        if(verbose):
            lal_cuda.log.open("Writing inputs to '%s'..." % (filename_inputs_out), end='')
        with open(filename_inputs_out, "wb") as inputs_file:
            self.np_floats().tofile(inputs_file)
            self.np_ints().tofile(inputs_file)
            self.freqs.tofile(inputs_file)
        if(verbose):
            lal_cuda.log.close("Done.")

    def run(self, buf=None, legacy=False):
        """Call the C-compiled model in lalsuite.

        If legacy is true, then assume that the compiled version of
        lalsuite we are using does not have PhenomP buffer support.

        :param buf: A buffer, as generated by the ADACS version of LALSuite
        :param legacy: True if using a version of LALSuite not compiled with the ADACS GPU buffer support
        :return: An instance of the outputs class
        """
        if(legacy):
            return(outputs(return_from_SimIMRPhenomPFrequencySequence=lalsimulation.SimIMRPhenomPFrequencySequence(
                self.freqs,
                self.chi1,
                self.chi2,
                self.chip,
                self.thetaJ,
                self.m1,
                self.m2,
                self.distance,
                self.alpha0,
                self.phic,
                self.fref,
                self.mode,
                None)))
        # ... else, assume that we are working with a version of PhenomP that does have buffer support
        else:
            return(outputs(return_from_SimIMRPhenomPFrequencySequence=lalsimulation.SimIMRPhenomPFrequencySequence(
                self.freqs,
                self.chi1,
                self.chi2,
                self.chip,
                self.thetaJ,
                self.m1,
                self.m2,
                self.distance,
                self.alpha0,
                self.phic,
                self.fref,
                self.mode,
                None,
                buf)))

    def __str__(self):
        """Return a string representation of the parameter set."""
        return "chi1=%e chi2=%e m1=%e m2=%e distance=%e thetaJ=%e alpha0=%e chip=%e phic=%e fref=%e mode=%d freqs=[%e...%e]" % (
            self.chi1, self.chi2, self.m1 / lal.lal.MSUN_SI, self.m2 / lal.lal.MSUN_SI, self.distance / (lal.lal.PC_SI * 100 * 1e6), self.thetaJ, self.alpha0, self.chip, self.phic, self.fref, self.mode, self.freqs[0], self.freqs[-1])

    def __eq__(self, other):
        """Test for equivilance of two sets of inputs."""
        return np.array_equal(
            self.np_floats(),
            other.np_floats()) and np.array_equal(
            self.np_ints(),
            other.np_ints()) and np.array_equal(
            self.freqs,
            other.freqs)

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)
