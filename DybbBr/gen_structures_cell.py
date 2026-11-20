
#!/THL6/home/jiangh/members/lisx/install_lisx/anaconda3/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from numpy import polyfit, poly1d
import argparse
from pathlib import Path
import shutil 
import os 
import sys
import re
import yaml
from matplotlib import pyplot as plt

from ase import Atoms
from ase.io import read
from ase.visualize import view
from ase.build import fcc111, add_adsorbate
from ase.io import write
from ase.io import vasp


##################################### constants ####################################
# conversion factor from m (reduced mass, amu) and w (frequency, cm^-1) to the amplitude of a harmonic oscillator x ( Angstrom)
hbar = 1.05457266
amu = 1.66053886
c = 2.9979
x_fac = np.sqrt( hbar / (amu * 2.0 * np.pi * c) ) * 10 ** 1.5
symbols = {1:"H", 6:"C", 7:"N", 8:"O", 14:"Si", 17:"Cl", 66:"Dy"} 
atMass = {"H":1.008, "C":12.01, "N":14.017, "O":16.0, "Si":28.09, "Cl":35.45, "Dy":573.0} 

def count_string2(string, filename):
    """
        return the number of appearances of string in the file filename
    """
    cmd = "n=`grep " + '"' + string + '" ' + filename + " | wc -l` && echo $n > tmp.dat"
    os.system(cmd)
    with open("tmp.dat") as f:
        n = int(f.readline().split()[0])
    os.system("rm tmp.dat")
    return n 

def count_string(string, filename):
    """
        return the number of appearances of string in the file filename
    """
    with open(filename) as f:
        data = f.read()
    n = len(data.split(string))-1
    return n
    
class mol_freq(object):
    """
        the object corresponds to freq calculation of one molecule 
        in Gaussian.
    """ 
    def __init__(self,outfile,posfile):
        self.outfile = outfile
        self.posfile = posfile
    
    def read_geometry(self):
        """
            read in the equilibrium geometry from the associated log file
        """
        self.orig = read(self.posfile)
        self.crds = np.array(self.orig.positions)
    
    def read_modes(self):
        """
            read in normal modes information from the output file
            including the frequencies (cm-1), reduced masses (amu), force constants (mDyne/A) 
            and atom displacement vectors (unitless). 
        """
        n_freq_analysis = count_string("cm-1",self.outfile)
        freqs = np.empty([n_freq_analysis], dtype = float)
        vecs = np.empty([n_freq_analysis,self.orig.get_number_of_atoms(),3], dtype = float)
        print("""
There are %4d frequency analysis in the log file, and we always use the first part.
When this number is greater than 1, the first part contains more digit numbers for the 
displacement vectors. 
""" % n_freq_analysis) 
        if n_freq_analysis == 1:
            print("\nThe vectors have only 2 digits, more accurate results could be printed in log by specifying freq=hpmodes in Gaussian input file!\n")      
        with open(self.outfile, "r") as f:
            for i in range(0,n_freq_analysis):
                line = f.readline()
                freqs[i] = float(line.split()[7])
                line = f.readline()
                for j in range(0,self.orig.get_number_of_atoms()):
                    vecs[i,j,...] = f.readline().split()[3:]
                line = f.readline()
        self.freqs, self.vecs = freqs, vecs 
        umasses = np.empty([n_freq_analysis], dtype = float)
        for i in range(0,n_freq_analysis):
            mass = 0
            for j in range(0,self.orig.get_number_of_atoms()):
                mass += (self.vecs[i,j,...] ** 2).sum()/atMass[symbols[self.orig.get_atomic_numbers()[j]]]
            umasses[i] = 1/mass
        self.masses = umasses

    def perturb(self, npts, mdp):
        """
            to perturb the equilibrium structures along each normal modes. 
            The maximum displacements are the amplitude of the zero-point vibration.
            The number of points are controled by npts. 
        """
        if npts % 2 != 0:
            print("The number of points along a given mode should be an odd number!")
            return
        # check if the output directory already exists
        if Path("./perb").exists():
            print("./perb directory already exist! We now delete it...")
            shutil.rmtree("./perb") 
        Path("perb").mkdir(parents=True)
        scalings = list(np.linspace(-mdp, mdp, npts + 1)) 
        scalings.pop( npts // 2) # remove the middle 0 
        scalings = np.array(scalings)
        Path("./perb/eq").mkdir(parents=True) 
        outfile = "./perb/eq/eq.xyz"
        new_crd = self.crds
        with open(outfile, "w") as f:
            # print out coordinates 
            print(self.orig.get_number_of_atoms(), file=f)
            print("eq", file=f) 
            for i in range(self.orig.get_number_of_atoms()):
                print(" %6s %12.6f %12.6f %12.6f" % (symbols[self.orig.get_atomic_numbers()[i]], new_crd[i,0], new_crd[i,1], new_crd[i, 2]), file=f)
        for imod in range(self.freqs.size):
            Path("./perb/" + str(imod+1)).mkdir(parents=True) 
            # we are dealing with the imod-th mode 
            x = x_fac / np.sqrt(self.masses[imod] * self.freqs[imod])
            norm = np.sqrt( ( self.vecs[imod, :, :] ** 2).sum() ) # in case the vectors are not strictly normalized 
            for ipt in range(npts):
                # the ipt-th displacement
                prefix = "./perb/" + str(imod+1) + "/" + str(ipt+1) 
                Path(prefix).mkdir(parents=True)  
                new_crd = self.crds + self.vecs[imod, :, :]/norm  * x * scalings[ipt]
                outfile = prefix + "/" + "mol_{x}_{y}.xyz".format(x=imod+1,y=ipt+1)
                with open(outfile, "w") as f:
                    # print out coordinates 
                    print(self.orig.get_number_of_atoms(), file=f)
                    print("ModeNo. %5d  StructureNo. %4d scaling= %4.1f%%" % (imod+1, ipt + 1, scalings[ipt] * 100), file=f) 
                    for i in range(self.orig.get_number_of_atoms()):
                        print(" %6s %12.6f %12.6f %12.6f" % (symbols[self.orig.get_atomic_numbers()[i]], new_crd[i,0], new_crd[i,1], new_crd[i, 2]), file=f)
        # "echo out some information..."
        print("All %5d modes are perturbed, the results stored in ./perb/" % self.freqs.size) 
        print("Structures are generated at ", end=" ")
        percents = ["%4.1f%%  " % x for x in scalings * 100] 
        print(" ".join(percents), " of the amplitude of the zero-point motion.")  
        
# the actual work is done here. 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
        A script to generate structures perturbed along normal modes 
        provided by a freq calculation done in Gaussian.""")
    parser.add_argument("--out", dest="outfile", required=True, help="The filename of vasp outcar.")
    parser.add_argument("--pos", dest="posfile", required=True, help="The filename of poscar.")
    parser.add_argument("--npts", "-np", dest="npts", type=int, default=10, help="Number of points along each modes. Default is 10.")
    parser.add_argument("--maxdp", "-mdp", dest="mdp", type=float, default=1, help="The max displacement of each modes. Default is 1.")
    parser.add_argument("--verbose", "-v", action="store_true", help="print out more information of the structures for debugging purpose.")
    args = parser.parse_args()
    
    mol = mol_freq(args.outfile,args.posfile) 
    mol.read_geometry()
    mol.read_modes()
    if args.verbose:
        print("atom numbers:\n", mol.orig.get_number_of_atoms(), "\n")
        print("atom equilibrium coordinates (A):\n", mol.crds, "\n")  
        print("freqs (cm-1):\n", mol.freqs, "\n")
        print("displacement vecs:\n", mol.vecs, "\n") 
        # check if the displacemnet vectors are normalized 
        for i in range(mol.freqs.size):
            print("The mode No. %4d vector norm is: %10.5f" % ((i+1), np.sqrt(( mol.vecs[i, :, :] * mol.vecs[i, :, :] ).sum())) )
    # perturbation starts here.
    mol.perturb(args.npts,args.mdp)  
