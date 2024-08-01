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

mass_Phi=1000
mass_S1=275
mass_S2=275
nevent=5000
InDir = "/users/divers/atlas/millot/home2/MG5_aMC_v3_4_2"  
OutDir = "/users/divers/atlas/millot/scratch/Recasting_HAHM_gluons_version"
ctaus = np.geomspace(0.1, 100, 200)


hasHEPData=0

random.seed(123)

#Path Pythia8 file
file_selection = f"{OutDir}/Script_mH{mass_Phi}_mS{mass_S1}/Events/run_01/tag_1_pythia8_events.hepmc.gz"
print("DEBUG we are opening this file", file_selection)

factor = 0.048 if mass_Phi==125 else 1

#Constant
c = 3e8# Light velocity in m/s



#pythia
events = hepmc.HEPMC_EventFile(file_selection) # Open HEPMC file
result = cmfp.parsing_hepmc_generic(events, verbose=1)
ctLxyDict = {}
ctLzDict = {}
for particleName, particleData in result.items():
  print(particleName)
  particleData = cmfp.kinematics(particleData)
  print('particleData: ', particleData)
  ctLxy, ctLz = cmfp.decayLength(particleData, ctaus)
  ctLxyDict[particleName] = ctLxy
  ctLzDict[particleName] = ctLz
  print(f"Longueurs de vie transverses pour {particleName}:")
  print(ctLxy)
  print(f"Longueurs de vie en z pour {particleName}:")
  print(ctLz)

# print("Table length:")
# print(f"ctLxyDict['llp1']: {len(ctLxyDict.get('llp1', []))}")
# print(f"ctLzDict['llp1']: {len(ctLzDict.get('llp1', []))}")
# print(f"ctLxyDict['LLP2']: {len(ctLxyDict.get('LLP2', []))}")
# print(f"ctLzDict['LLP2']: {len(ctLzDict.get('LLP2', []))}")


# Output file name
output_filename = f"./Results_ancien/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_nevents{nevent}.txt"
# Open the file in write mode
with open(output_filename, 'w') as f:
    # Écrire les en-têtes des colonnes
    f.write("massPhi\tmass_S1\tmass_S2\tctau\tnevents\tefficiency\n")

    # Browse the efficiency values and write to the file
    for ict, ctau in enumerate(ctaus):
      result['llp1']['Lxy'] = ctLxyDict['llp1'][ict]
      result['llp1']['Lz'] = ctLzDict['llp1'][ict]
      result['LLP2']['Lxy'] = ctLxyDict['LLP2'][ict]
      result['LLP2']['Lz'] = ctLzDict['LLP2'][ict]
      #print('LC DEBUG', ict, ctau)
      #print(list(result['LLP2'].Lz))

      # Calculation of efficiency trhough the map
      if mass_Phi >= 400: # Condition if the sample is "High-ET" or " Low-ET"
       print('Map eval with Pythia')
       eff_highETX = cmfp.eff_map(result, selection='high-ET') # Compute the efficiency from Pythia
       print('efficacité :', eff_highETX)
       efficiency = float(eff_highETX)

      else:
       print('Map eval with Pythia')
       eff_lowETX = cmfp.eff_map(result, selection='low-ET')
       print('efficacité :', eff_lowETX)
       efficiency = float(eff_lowETX)
      # Write the information in the file
      f.write(f"{mass_Phi}\t{mass_S1}\t{mass_S2}\t{ctau}\t{nevent}\t{efficiency}\n")

print(f"The efficiency values have been saved in file {output_filename}")


