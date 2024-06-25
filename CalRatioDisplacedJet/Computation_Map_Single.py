#!/usr/bin/env pythonMG_pdg_DH2_1
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

mass_Phi = int(sys.argv[1])
mass_S = int(sys.argv[2])

InDir = sys.argv[3]
OutDir = sys.argv[4]
model = sys.argv[5]
mode = sys.argv[6]

hasHEPData=0

random.seed(123)
hep.style.use("ATLAS") # Define a style for the plots

#Path Pythia8 file
file_selection = f"{OutDir}/Script_mH{mass_Phi}_mS{mass_S}/Events/run_01/tag_1_pythia8_events.hepmc.gz"

#Path MadGraph file
MG_file_selection = f"{OutDir}/Script_mH{mass_Phi}_mS{mass_S}/Events/run_01/unweighted_events.lhe.gz"

#HEP data file
file_HEP = glob.glob(f"{InDir}/recastingCodes/CalRatioDisplacedJet/ATLAS_data/HEPData*_{mass_Phi}_{mass_S}.root")

if len(file_HEP): 
  hasHEPData=1
  file_HEP=file_HEP[0]
  #HEP limits file
  file_HEP_limit = glob.glob(f"{InDir}/recastingCodes/CalRatioDisplacedJet/ATLAS_data/HEP_Limits/HEPData*_{mass_Phi}_{mass_S}.root")[0]

factor = 0.048 if mass_Phi==125 else 1

#Constant
c = 3e8# Light velocity in m/s

os.system(f"mkdir -p Plots_High/HAHM_{mode}_version/{mass_Phi}_{mass_S}/Cross_section")
os.system(f"mkdir -p Plots_High/HAHM_{mode}_version/{mass_Phi}_{mass_S}/Efficiencies")
os.system(f"mkdir -p Plots_Low/HAHM_{mode}_version/{mass_Phi}_{mass_S}/Cross_section")
os.system(f"mkdir -p Plots_Low/HAHM_{mode}_version/{mass_Phi}_{mass_S}/Efficiencies")

tauN=np.geomspace(0.1,1e2, 200)

print('Start with MG')
#MG
MG_events = lhe.EventFile(MG_file_selection) # Open LHE file
Nevent = (len(MG_events)) # get the NEvents from the length of the file

px, py, pz, pdg, E, MASS = cmfp.parsing_LHE(MG_events) #Parsing the LHE file
MG_px_DH1, MG_py_DH1,MG_pz_DH1,MG_E_DH1,MG_mass_DH1,MG_pdg_DH1_1 = cmfp.recover_MG_DH1(px, py, pz, E, MASS, pdg) # Separate data from DH1 and DH2
MG_pT_DH1,MG_eta_DH1, MG_gamma_DH1 = cmfp.kinematics_MG(MG_px_DH1,MG_py_DH1,MG_pz_DH1,MG_E_DH1) # Computing kinematics for DH1
MG_px_DH2, MG_py_DH2,MG_pz_DH2,MG_E_DH2,MG_mass_DH2,MG_pdg_DH2_1 = cmfp.recover_MG_DH2(px, py, pz, E, MASS, pdg) # Separate data from DH1 and DH2
MG_pT_DH2,MG_eta_DH2, MG_gamma_DH2 = cmfp.kinematics_MG(MG_px_DH2,MG_py_DH2,MG_pz_DH2,MG_E_DH2) # Computing kinematics for DH2


print('Now Pythia')
#Pythia
events = hepmc.HEPMC_EventFile(file_selection) # Open HEPMC file
px_TOT, py_TOT, pz_TOT, E_TOT, mass_TOT,pdg_TOT = cmfp.parsing_hepmc(events) # Parsing the HEPMC file
px_tot, py_tot, pz_tot, E_tot, mass_tot, pdg_tot = cmfp.conversion_one_list(px_TOT, py_TOT, pz_TOT, E_TOT, mass_TOT, pdg_TOT) # Obtaining data in one list
px_DH1, px_DH2, py_DH1, py_DH2, pz_DH1, pz_DH2, pdg_tot_DH1, pdg_tot_DH2, E_DH1, E_DH2, mass_DH1, mass_DH2 = cmfp.recover(px_tot, py_tot, pz_tot, E_tot, mass_tot, pdg_tot) # Separate data from DH1 and DH2
beta_DH1, gamma_DH1, pT_DH1, eta_DH1 = cmfp.kinematics(px_DH1, py_DH1, pz_DH1, E_DH1) # Computing kinematics for DH1
beta_DH2, gamma_DH2, pT_DH2, eta_DH2 = cmfp.kinematics(px_DH2, py_DH2, pz_DH2, E_DH2) # Computing kinematics for DH2


Lxy_tot_DH1, Lz_tot_DH1 = cmfp.decaylength(px_DH1, py_DH1, pz_DH1, E_DH1, gamma_DH1, tauN) # Computing the decay length for DH1
Lxy_tot_DH2, Lz_tot_DH2 = cmfp.decaylength(px_DH2, py_DH2, pz_DH2, E_DH2, gamma_DH2, tauN) # Computing the decay length for DH2
MG_Lxy_tot_DH1, MG_Lz_tot_DH1 = cmfp.decaylength(MG_px_DH1, MG_py_DH1, MG_pz_DH1, E_DH1, MG_gamma_DH1, tauN) # Computing decay length for DH1
MG_Lxy_tot_DH2, MG_Lz_tot_DH2 = cmfp.decaylength(MG_px_DH2, MG_py_DH2, MG_pz_DH2, E_DH2, MG_gamma_DH2, tauN) # Computing decay length for DH2

#HEP data
data_HEP, branch_HEP_limit = None, None
if hasHEPData:
  cmfp.elem_list(file_HEP, file_HEP_limit) # Recover public data from ATLAS to compare the results

########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

#######################################################Computing the efficiencies and ploting the results###########################################################

if mass_Phi >= 400: # Condition if the sample is "High-ET" or " Low-ET"
    print('Map eval with MG')
    MG_eff_highETX = cmfp.eff_map_MG_high(MG_pT_DH1, MG_eta_DH1,MG_Lxy_tot_DH1, MG_Lz_tot_DH1, MG_pdg_DH1_1, MG_pT_DH2, MG_eta_DH2, MG_Lxy_tot_DH2, MG_Lz_tot_DH2, MG_pdg_DH2_1, tauN, Nevent,  mass_Phi, mass_S) # Compute the efficiency from MG
    print('Map eval with Pythia')
    eff_highETX = cmfp.eff_map_High(pT_DH1, eta_DH1, Lxy_tot_DH1, Lz_tot_DH1, abs(pdg_tot_DH1), pT_DH2, eta_DH2, Lxy_tot_DH2, Lz_tot_DH2, abs(pdg_tot_DH2), tauN, Nevent,  mass_Phi, mass_S) # Compute the efficiency from Pythia
    print('Plotting...')
    cmfp.plt_eff_high(MG_eff_highETX, eff_highETX, tauN, data_HEP, mass_Phi, mass_S,Nevent,mode ) # Ploting and saving a comparison of all the results of efficiencies
    cmfp.plt_cross_High(eff_highETX, tauN, mass_Phi, mass_S, branch_HEP_limit, factor,Nevent,mode)# Ploting and saving a comparison of the limits obtained with the map and by ATLAS.

else:
    MG_eff_lowETX = cmfp.eff_map_MG_low(MG_pT_DH1, MG_eta_DH1,MG_Lxy_tot_DH1, MG_Lz_tot_DH1, MG_pdg_DH1_1, MG_pT_DH2, MG_eta_DH2, MG_Lxy_tot_DH2, MG_Lz_tot_DH2, MG_pdg_DH2_1, tauN, Nevent, mass_Phi, mass_S)
    eff_lowETX = cmfp.eff_map_Low(pT_DH1, eta_DH1, Lxy_tot_DH1, Lz_tot_DH1, abs(pdg_tot_DH1), pT_DH2, eta_DH2, Lxy_tot_DH2, Lz_tot_DH2, abs(pdg_tot_DH2), tauN, Nevent, mass_Phi, mass_S)
    cmfp.plt_eff_low(MG_eff_lowETX, eff_lowETX, tauN, data_HEP, mass_Phi, mass_S,Nevent,mode)
    cmfp.plt_cross_Low(eff_lowETX, tauN, mass_Phi, mass_S, branch_HEP_limit, factor,Nevent,mode)
