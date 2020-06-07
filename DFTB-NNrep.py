import sys
import numpy as np
import logging
from os import environ, listdir, mkdir, chdir

import torch
from schnetpack.interfaces.ase_interface import SpkCalculator

from ase.io import read, write
from ase.units import Hartree, Bohr
from ase.optimize import QuasiNewton, BFGS
from ase.vibrations import Vibrations
from ase.calculators.dftb import Dftb
import ase.calculators.mixing 

try:
  idir = sys.argv[1] ;  odir = sys.argv[2]
except:
  print("Usage:", sys.argv[0], "MolsDir OutDir")
  sys.exit(1)

# Load NN model for DFTB3 repulsive energies and forces
logging.info("get model")
model_dir = 'dftbnn_model/'
path_to_model = model_dir+'best_model'
model = torch.load(path_to_model, map_location='cuda')

# SchNet calculator
SchNetcalc = SpkCalculator(model, device='cuda',energy="ErepD3", forces="FOR3")
 
flist = listdir(idir)

for iFile, fname in enumerate(flist):
    sysname = fname[:-4]
    print(sysname)
    mkdir(odir+sysname)
    chdir(odir+sysname)
    atoms = read(idir+fname)
    nAtoms = len(atoms)

# DFTB calculator
    DFTBcalc = Dftb(label='current_dftb',
                atoms=atoms,
                run_manyDftb_steps=True,
                Hamiltonian_SCC = 'Yes',
                Hamiltonian_MaxSCCIterations = '4000',
#                Hamiltonian_Filling = ' Fermi{ Temperature[K]= 100 }',
                Hamiltonian_ThirdOrderFull = 'Yes',
                Hamiltonian_SCCTolerance = '1E-8',
                Hamiltonian_PolynomialRepulsive_ = '',
                Hamiltonian_PolynomialRepulsive_setForAll = '{Yes}',
                Analysis_ ='',
                Analysis_CalculateForces = 'Yes')

# Mixing calculators
    QMMMcalc =  ase.calculators.mixing.SumCalculator([DFTBcalc,SchNetcalc], atoms)

    atoms.set_calculator(QMMMcalc)

# Optmizing geometry
    qn = BFGS(atoms, trajectory='mol.traj')
    qn.run(fmax=0.0001)
    write('final.xyz', atoms)

# Computing vibrational modes
    vib = Vibrations(atoms, delta=0.01,nfree=4)
    vib.run()
    vib.summary()
    vb_ev = vib.get_energies()
    vb_cm = vib.get_frequencies()
    ENE = float(atoms.get_total_energy())

    o1 = open('info-modes.dat', 'w')
    o1.write("# Potential energy:   " + "{: >24}".format(ENE) + "\n")
    for i in range(0, len(vb_ev)):
      o1.write("{: >24}".format(vb_ev.real[i]) + "{: >24}".format(vb_ev.imag[i]) + "{: >24}".format(vb_cm.real[i]) + "{: >24}".format(vb_cm.imag[i]) + "\n")

    vib.write_jmol()

    o1.close()
    chdir(odir)

