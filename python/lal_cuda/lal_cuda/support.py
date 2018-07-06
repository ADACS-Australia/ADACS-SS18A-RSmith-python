from __future__ import print_function
import numpy as np
import sys
import glob
import math

import lal
import lalsimulation
import lal_cuda

chi1_default    = 0.1
chi2_default    = 0.2
m1_default      = 30
m2_default      = 30
chip_default    = 0.34
thetaJ_default  = 1.1
alpha0_default  = 1.5
distance_default= 1000
phic_default    = np.pi*0.4
fref_default    = 30

class PhenomPCore_inputs(object):

    def __init__(self, chi1=chi1_default, chi2=chi2_default, m1=m1_default, m2=m2_default, chip=chip_default, thetaJ=thetaJ_default, alpha0=alpha0_default, distance=distance_default, phic=phic_default, fref=fref_default,convert_units=True):
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
        if(convert_units):
            self.m1 = self.m1 * lal.lal.MSUN_SI
            self.m2 = self.m2 * lal.lal.MSUN_SI
            self.distance = self.distance * lal.lal.PC_SI*100*1e6

    def np(self):
        # A numpy-array packaging of the inputs
        return np.array([self.chi1,self.chi2,self.chip,self.thetaJ,self.m1,self.m2,self.distance,self.alpha0,self.phic,self.fref])

    @classmethod
    def fromfile(cls,inputs_file):
        inputs_np=np.fromfile(inputs_file,dtype=cls().np().dtype,count=len(cls().np()))
        chi1 = inputs_np[0]
        chi2 = inputs_np[1]
        chip = inputs_np[2]
        thetaJ = inputs_np[3]
        m1 = inputs_np[4]
        m2 = inputs_np[5]
        distance = inputs_np[6]
        alpha0 = inputs_np[7]
        phic = inputs_np[8]
        fref = inputs_np[9]
        return(cls(chi1=chi1, chi2=chi2, m1=m1, m2=m2, chip=chip, thetaJ=thetaJ, alpha0=alpha0, distance=distance, phic=phic, fref=fref,convert_units=False))

    def __str__(self):
        return "chi1=%le chi2=%le m1=%le m2=%le distance=%le thetaJ=%le alpha0=%le chip=%le phic=%le fref=%le"%(self.chi1, self.chi2, self.m1 / lal.lal.MSUN_SI, self.m2 / lal.lal.MSUN_SI, self.distance / (lal.lal.PC_SI*100*1e6), self.thetaJ, self.alpha0, self.chip, self.phic, self.fref)

    def __eq__(self, other):
        """Test for equivilance of two sets of inputs"""
        return self.chi1==other.chi1 and self.chi2==other.chi2 and self.m1==other.m1 and self.m2==other.m2 and self.distance==other.distance and self.thetaJ==other.thetaJ and self.alpha0==other.alpha0 and self.chip==other.chip and self.phic==other.phic and self.fref==other.fref

    def __ne__(self, other):
        """Overrides the default implementation (unnecessary in Python 3)"""
        return not self.__eq__(other)

def write_results_PhenomPCore(inputs_runtime,freqs,hp_val,hc_val,phic,write2stdout,write2bin,filename_label=None):

    # Write results to screen
    if(write2stdout):
        for [f_i,hp_i,hc_i] in zip(freqs,hp_val,hc_val):
            print(f_i,hp_i.real,hp_i.imag,hc_i.real,hc_i.imag)

    # Write to binary file
    if(write2bin):

        # Set filenames
        if(filename_label):
            filename_inputs_out  = "inputs.dat."+filename_label
            filename_outputs_out = "outputs.dat."+filename_label
        else:
            filename_inputs_out  = "inputs.dat"
            filename_outputs_out = "outputs.dat"

        # Write inputs
        print("Writing inputs to '%s'..."%(filename_inputs_out),end='')
        inputs_file  = open(filename_inputs_out, "wb")
        inputs_runtime.np().tofile(inputs_file)
        freqs.tofile(inputs_file)
        inputs_file.close()
        print("Done.")

        # Write results
        print("Writing outputs to '%s'..."%(filename_outputs_out),end='')
        outputs_file = open(filename_outputs_out, "wb")
        hp_val.tofile(outputs_file)
        hc_val.tofile(outputs_file)
        phic.tofile(outputs_file)
        outputs_file.close()
        print("Done.")

def check_PhenomPCore(inputs_runtime,freqs,hp_val,hc_val,phic,tolerance=None,verbose=True):

    # Get a list of reference input files
    reference_inputs = glob.glob(lal_cuda.full_path_datafile("inputs.dat*"))
    reference_outputs = [ reference_input_i.replace("inputs.dat","outputs.dat") for reference_input_i in reference_inputs ]

    # Look to see if the given inputs are in the stored reference inputs
    outputs_ref_file = None
    for reference_inputs_i,reference_outputs_i in zip(reference_inputs,reference_outputs):
        with open(lal_cuda.full_path_datafile("inputs.dat"), "rb") as inputs_file:
            inputs_i=PhenomPCore_inputs.fromfile(inputs_file)
            freqs_ref=np.fromfile(inputs_file,dtype=freqs.dtype,count=len(freqs))
            n_freq = len(freqs_ref)

        # Check to see if this set of inputs matches the set that has been passed
        if(inputs_i==inputs_runtime):
            inputs_ref = inputs_i
            outputs_ref_file = reference_outputs_i
            break

    # Perform check if a match has been found
    if(not outputs_ref_file):
        print("Checking could not be performed: reference data set with given inputs (%s) not found."%(inputs_runtime))
    else:
        if(verbose):
            print('Performing test...')

        # Read reference dataset's outputs
        outputs_file = open(lal_cuda.full_path_datafile(outputs_ref_file), "rb")
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
            hcval_imag_diff_max = max([hcval_imag_diff_max,hcval_imag_diff_i])
        hpval_real_diff_avg /= float(len(hp_val))
        hpval_imag_diff_avg /= float(len(hp_val))
        hpval_real_diff_avg /= float(len(hp_val))
        hpval_imag_diff_avg /= float(len(hp_val))

        # Report results
        if(verbose):
            print('   Average/maximum real(hp_val) fractional difference: %.2e/%.2e'%(hpval_real_diff_avg,hpval_real_diff_max))
            print('   Average/maximum imag(hp_val) fractional difference: %.2e/%.2e'%(hpval_imag_diff_avg,hpval_imag_diff_max))
            print('   Average/maximum real(hc_val) fractional difference: %.2e/%.2e'%(hcval_real_diff_avg,hcval_real_diff_max))
            print('   Average/maximum imag(hx_val) fractional difference: %.2e/%.2e'%(hcval_imag_diff_avg,hcval_imag_diff_max))
            print("Done.")

        # Check values against a tolerance (if given)
        return {'hpval_real_diff_avg':hpval_real_diff_avg,'hpval_real_diff_max':hpval_real_diff_max,'hpval_imag_diff_avg':hpval_imag_diff_avg,'hpval_imag_diff_max':hpval_imag_diff_max,'hcval_real_diff_avg':hcval_real_diff_avg,'hcval_real_diff_max':hcval_real_diff_max,'hcval_imag_diff_avg':hcval_imag_diff_avg,'hcval_imag_diff_max':hcval_imag_diff_max}
