#!/usr/bin/env python3
import os
import numpy as np
import random
random.seed(123)
# The number of events you want to generate for each sample
nevent = 5000

#The Phi and S particles masses models you want to generate, i.e [m_Phi, m_S]
#WARNING: If you plan to use your own machine, run the different masses one by one
#or consider using the "Writing_Scripts_MG+P8.py" code (that loop over the chosen masses)
masses = [  
          #[200,50,50]                       
          [1000, 275,275],   #please note that here the number of events is 5000 and not 10000
          #[400, 100,100] ,
          #[125,55,55]
          #[600, 150,150]        
          ]

#The full path to where the MadGraph folder is!
#InDir = "/users/divers/atlas/haddad/home2/MG5_aMC_v3_4_2"  
InDir = "/users/divers/atlas/millot/home2/MG5_aMC_v3_4_2"  

#The full path to where you want all your outputs to be saved 
#OutDir = "/users/divers/atlas/haddad/scratch/Recasting"  
OutDir = "/users/divers/atlas/millot/scratch/Recasting_HAHM_asym_version"  # "/users/divers/atlas/millot/scratch/Recasting_HAHM_past_version" or "/users/divers/atlas/millot/scratch/Recasting_HAHM_gluons_version"

model = "HAHM_asym_interaction_order_without_top_quark_UFO" # HAHM_asym_interaction_order_without_top_quark_UFO 
mode = "new" 
#ctaus = [
        #[2,2],
        #[10,10],
        #]

ctaus = [ [ct,ct] for ct in np.geomspace(0.1,1e2, 200) ] 
Lambda ="100"
k="1"


for mass_Phi, mass_S1, mass_S2 in masses:
  for ct1,  ct2 in ctaus :
    f = open(f"{OutDir}/Job_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}.sh",'w')
    
    #These two lines will setup ATLAS and Athena in order to use an updated verision of python (>= 3.7)
    #To be removed if you haven't access to cvmfs, or simply if you are using your up-to-date python  
    f.write("source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh\n")
    f.write("asetup AnalysisBase,24.2.35\n")
    f.write(f"cd {InDir}\n")
    f.write(f"pip install numpy -t $PWD\n")
    f.write(f"export PYTHONPATH={InDir}:$PYTHONPATH\n")
    f.write(f"export TQDM_DISABLE=1\n")
    f.write(f"python3 -c 'import numpy as np ; print(\"NUMPY PATH\", np.__path__)'\n")
    #Or avtivate your generated python environment
    ##f.write(f"{InDir}/env/Scripts/activate")
    
    f.write(f"python3 {InDir}/recastingCodes/CalRatioDisplacedJet/Asym/writing_script.py {mass_Phi} {mass_S1} {mass_S2} {nevent} {InDir} {OutDir} {model} {mode} {ct1} {ct2} {Lambda} {k}\n")
    f.write(f"cd {OutDir}\n")
    f.write(f"{InDir}/bin/mg5_aMC -f {OutDir}/script_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}.txt\n")
    f.write(f"cd {InDir}/recastingCodes/CalRatioDisplacedJet/\n")
    f.write(f"python3 {InDir}/recastingCodes/CalRatioDisplacedJet/Asym/comp_map.py {mass_Phi} {mass_S1} {mass_S2} {nevent} {InDir} {OutDir} {model} {mode} {ct1} {ct2} {Lambda} {k} \n")
    f.close()
    
    os.system(f"chmod +x {OutDir}/Job_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}.sh")
    
    #Depending on where you want to run MadGraph, use the first command line to run locally
    #Or the second one to run in your lab servers (to be updated accordingly, here is for LPC)
    ##os.system(f"{InDir}/bin/mg5_aMC -f {OutDir}/script_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}.txt")
    os.system(f"qsub -q prod2C7@clratlserv04 -o {OutDir} -e {OutDir} {OutDir}/Job_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_{ct1}_ct2_{ct2}.sh")
    
    
    
