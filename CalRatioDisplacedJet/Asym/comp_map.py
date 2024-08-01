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
import hepmc_parser as hepmc
import lhe_parser as lhe
import comp_function as cmfp
import random
import math
import os
import glob




mass_Phi = int(sys.argv[1])
mass_S1 = int(sys.argv[2])
mass_S2 = int(sys.argv[3])
nevent = int(sys.argv[4]) 

InDir = sys.argv[5]
OutDir = sys.argv[6]

model = sys.argv[7]
mode = sys.argv[8]

ct1 = sys.argv[9]
ct2= sys.argv[10]
Lambda = sys.argv[11]
k = sys.argv[12]

hasHEPData=0

random.seed(123)
hep.style.use("ATLAS") # Define a style for the plots

#Path Pythia8 file
file_selection = f"{OutDir}/Script_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}/Events/run_01/tag_1_pythia8_events.hepmc.gz"
print("DEBUG we are opening this file", file_selection)



factor = 0.048 if mass_Phi==125 else 1

#Constant
c = 3e8# Light velocity in m/s

os.system(f"mkdir -p Plots_High/HAHM_{mode}_version/{mass_Phi}_{mass_S1}/Cross_section")
os.system(f"mkdir -p Plots_High/HAHM_{mode}_version/{mass_Phi}_{mass_S1}/Efficiencies")
os.system(f"mkdir -p Plots_Low/HAHM_{mode}_version/{mass_Phi}_{mass_S1}/Cross_section")
os.system(f"mkdir -p Plots_Low/HAHM_{mode}_version/{mass_Phi}_{mass_S1}/Efficiencies")

print('Now Pythia')
#Pythia
events = hepmc.HEPMC_EventFile(file_selection) # Open HEPMC file
result = cmfp.parsing_hepmc_generic(events, verbose=1)

for particleName, particleData in result.items():
  print(particleName)
  particleData = cmfp.kinematics(particleData)
  print('particleData: ', particleData)
  ct = 9999
  if particleName == 'llp1': ct=float(ct1)
  if particleName == 'LLP2': ct=float(ct2)

  ctLxy, ctLz = cmfp.decayLength(particleData, [ct])
  print(f"Longueurs de vie transverses pour {particleName}:")
  print(ctLxy)
  print(f"Longueurs de vie en z pour {particleName}:")
  print(ctLz)


########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

#######################################################Computing the efficiencies and ploting the results###########################################################
#print('len results : ', len(result))
#print('results llp1 pT : ', result['llp1']['pT'])

# Save efficiency to a file
output_file = f"./Asym/Results/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}_nevents{nevent}.txt"

if mass_Phi >= 400: # Condition if the sample is "High-ET" or " Low-ET"
    print('Map eval with Pythia')
    eff_highETX = cmfp.eff_map(result, selection='high-ET') # Compute the efficiency from Pythia
    print('efficacité :', eff_highETX)
    efficiency = float(eff_highETX)
    #print(os.getcwd())
    #np.savetxt(f'./Asym/Results/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}_nevents{nevent}.txt', [{mass_Phi}], [{mass_S1}],[{mass_S2}],[{ct1}],[{ct2}],[{nevent}],[eff_highETX] )
#    cmfp.plt_eff_high(MG_eff_highETX, eff_highETX, tauN, data_HEP, mass_Phi, mass_S,Nevent,mode ) # Ploting and saving a comparison of all the results of efficiencies
#    cmfp.plt_cross_High(eff_highETX, tauN, mass_Phi, mass_S, branch_HEP_limit, factor,Nevent,mode)# Ploting and saving a comparison of the limits obtained with the map and by ATLAS.

else:
    print('Map eval with Pythia')
    eff_lowETX = cmfp.eff_map(result, selection='low-ET')
    print('efficacité :', eff_lowETX)
    efficiency = float(eff_lowETX)
    #np.savetxt(f'./Asym/Results/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}_nevents{nevent}.txt', [eff_lowETX] )
#    cmfp.plt_eff_low(MG_eff_lowETX, eff_lowETX, tauN, data_HEP, mass_Phi, mass_S,Nevent,mode)
#    cmfp.plt_cross_Low(eff_lowETX, tauN, mass_Phi, mass_S, branch_HEP_limit, factor,Nevent,mode)

#print('Plotting...')

data_to_save = np.array([[mass_Phi, mass_S1, mass_S2, ct1, ct2, nevent, efficiency]],dtype=float)

# Save the array to the file
np.savetxt(output_file, data_to_save, header="mass_Phi mass_S1 mass_S2 ct1 ct2 nevent efficiency")

print(f"Efficiency saved to {output_file}")

