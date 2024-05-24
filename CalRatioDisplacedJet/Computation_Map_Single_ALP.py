#!/usr/bin/env python
import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import scipy
import time
import readMapNew as rmN
import tqdm
import mplhep as hep
import lhe_parser as lhe
import hepmc_parser as hepmc
import Computation_Functions as cmfp
import random
import math
import os
import glob

#mass_Phi = int(sys.argv[1])
#mass_S = int(sys.argv[2])
mode  = sys.argv[1]
mass_ALP = float(sys.argv[2])
mass_Phi = -1

slug = f'mALP{mass_ALP:.6g}_{mode}' 
InDir = sys.argv[3]
OutDir = sys.argv[4]

hasHEPData=0

random.seed(123)
hep.style.use("ATLAS") # Define a style for the plots

#Path Pythia8 file
file_selection = f"{OutDir}/Script_{slug}/Events/run_01/tag_1_pythia8_events.hepmc.gz"

#Path MadGraph file
MG_file_selection = f"{OutDir}/Script_{slug}/Events/run_01/unweighted_events.lhe.gz"

#HEP data file
#file_HEP = glob.glob(f"{InDir}/recastingCodes/CalRatioDisplacedJet/ATLAS_data/HEPData*_{mass_Phi}_{mass_S}.root")

#print (f"{InDir}/recastingCodes/CalRatioDisplacedJet/ATLAS_data/HEPData*_{mass_Phi}_{mass_S}.root")
#if len(file_HEP): 
#  hasHEPData=1
#  file_HEP=file_HEP[0]
#  #HEP limits file
#  file_HEP_limit = glob.glob(f"{InDir}/recastingCodes/CalRatioDisplacedJet/ATLAS_data/HEP_Limits/HEPData*_{mass_Phi}_{mass_S}.root")[0]
#  print("HAS HEPDATA!")

#factor = 0.048 if mass_Phi==125 else 1

#Constant
c = 3e8# Light velocity in m/s

os.system("mkdir -p Plots_ALP")

#tauN=np.geomspace(0.1,1e2, 30)
tauN=np.geomspace(1e-4,1e2, 100)

print('Start with MG')
#MG
MG_events = lhe.EventFile(MG_file_selection) # Open LHE file
Nevent = (len(MG_events)) # get the NEvents from the length of the file

#px, py, pz, pdg, E, MASS = cmfp.parsing_LHE(MG_events) #Parsing the LHE file
#MG_px_V, MG_py_V,MG_pz_V,MG_E_V,MG_mass_V,MG_pdg_V_1 = cmfp.recover_MG_V(px, py, pz, E, MASS, pdg) # Separate data from V and ALP
#MG_pT_V,MG_eta_V, MG_gamma_V = cmfp.kinematics_MG_V(MG_px_V,MG_py_V,MG_pz_V,MG_E_V) # Computing kinematics for V
#MG_px_ALP, MG_py_ALP,MG_pz_ALP,MG_E_ALP,MG_mass_ALP,MG_pdg_ALP_1 = cmfp.recover_MG_ALP(px, py, pz, E, MASS, pdg) # Separate data from V and ALP
#MG_pT_ALP,MG_eta_ALP, MG_gamma_ALP = cmfp.kinemamtics_MG_ALP(MG_px_ALP,MG_py_ALP,MG_pz_ALP,MG_E_ALP) # Computing kinematics for ALP


print('Now Pythia')
#Pythia
events = hepmc.HEPMC_EventFile(file_selection) # Open HEPMC file
px_TOT, py_TOT, pz_TOT, E_TOT, mass_TOT,pdg_TOT = cmfp.parsing_hepmc_ALP(events) # Parsing the HEPMC file
print(len(px_TOT))
print(pdg_TOT)
px_tot, py_tot, pz_tot, E_tot, mass_tot, pdg_tot = cmfp.conversion_one_list(px_TOT, py_TOT, pz_TOT, E_TOT, mass_TOT, pdg_TOT) # Obtaining data in one list
px_V, px_ALP, py_V, py_ALP, pz_V, pz_ALP, pdg_tot_V, pdg_tot_ALP, E_V, E_ALP, mass_V, mass_alp = cmfp.recover(px_tot, py_tot, pz_tot, E_tot, mass_tot, pdg_tot) # Separate data from V and ALP
pdg_tot_ALP = np.array(pdg_TOT)[:,0]
#exit(1)
beta_V, gamma_V, pT_V, eta_V = cmfp.kinematics_DH1(px_V, py_V, pz_V, E_V) # Computing kinematics for V
beta_ALP, gamma_ALP, pT_ALP, eta_ALP = cmfp.kinematics_DH1(px_ALP, py_ALP, pz_ALP, E_ALP) # Computing kinematics for ALP
Lxy_tot_V, Lz_tot_V = cmfp.decaylenghtDH1(px_V, py_V, pz_V, E_V, gamma_V, tauN) # Computing the decay lenght for V
Lxy_tot_ALP, Lz_tot_ALP = cmfp.decaylenghtDH1(px_ALP, py_ALP, pz_ALP, E_ALP, gamma_ALP, tauN) # Computing the decay lenght for ALP

#MG_Lxy_tot_V, MG_Lz_tot_V = cmfp.decaylenght_MG_V(MG_px_V, MG_py_V, MG_pz_V, E_V, MG_gamma_V, tauN) # Computing decay lenght for V
#MG_Lxy_tot_ALP, MG_Lz_tot_ALP = cmfp.decaylenght_MG_ALP(MG_px_ALP, MG_py_ALP, MG_pz_ALP, E_ALP, MG_gamma_ALP, tauN) # Computing decay lenght for ALP


#HEP data
data_HEP, branch_HEP_limit = None, None
#if hasHEPData:
#  data_HEP, branch_HEP_limit = cmfp.elem_list(file_HEP, file_HEP_limit) # Recover public data from ATLAS to compare the results
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

#######################################################Computing the efficiencies and ploting the results###########################################################

map_eff_highETX = np.zeros(len(tauN))
#eff_highETX = cmfp.eff_bdt_High(pT_V, eta_V, Lxy_tot_V, Lz_tot_V, abs(pdg_tot_V), pT_ALP, eta_ALP, Lxy_tot_ALP, Lz_tot_ALP, abs(pdg_tot_ALP),  E_V, E_ALP,tauN, Nevent,  mass_Phi, mass_S) # Compute the efficiency from Pythia
eff_highETX = cmfp.eff_bdt_WALP(pT_V, eta_V, Lxy_tot_V, Lz_tot_V, abs(pdg_tot_V), pT_ALP, eta_ALP, Lxy_tot_ALP, Lz_tot_ALP, abs(pdg_tot_ALP),  E_V, E_ALP,tauN, Nevent,  mass_Phi, mass_ALP) # Compute the efficiency from Pythia
cmfp.plt_eff(map_eff_highETX, eff_highETX, tauN, data_HEP, mass_Phi, mass_ALP, model="ALP") # Ploting and saving a comparison of all the results of efficiencies
#cmfp.plt_cross(eff_highETX, tauN, mass_Phi, mass_S, branch_HEP_limit, factor, hepdata_eff=data_HEP)# Ploting and saving a comparison of the limits obtained with the map and by ATLAS.
