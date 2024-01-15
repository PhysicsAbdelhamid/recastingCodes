#!/usr/bin/env python3
import os

# The number of events you want to generate for each sample
nevent = 20000

#The Phi and S particles masses models you want to generate, i.e [m_Phi, m_S]
#WARNING: If you plan to use your own machine, run the different masses one by one
#or consider using the "Writing_Scripts_MG+P8.py" code (that loop over the chosen masses)
masses = [                         
          [60, 5],
          [125, 55],
          [200, 50],
          [600, 150],
          [1000, 275],
          ]

#The full path to where the MadGraph folder is!
InDir = "/users/divers/atlas/haddad/home2/MG5_aMC_v3_4_2"  

#The full path to where you want all your outputs to be saved 
OutDir = "/users/divers/atlas/haddad/scratch/Recasting"  


for mass_Phi, mass_S in masses:
    
    f = open(f"{OutDir}/Job_mH{mass_Phi}_mS{mass_S}.sh",'w')
    
    #These two lines will setup ATLAS and Athena in order to use an updated verision of python (>= 3.7)
    #Consider removing if you haven't access to cvmfs, or simply if your python version is up-to-date   
    f.write("source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh\n")
    f.write("asetup AnalysisBase,24.2.35\n")
    
    f.write(f"python3 {InDir}/recastingCodes/CalRatioDisplacedJet/Writing_Scripts_MG+P8_Single.py {mass_Phi} {mass_S} {nevent} {InDir} {OutDir}\n")
    f.write(f"cd {OutDir}\n")
    f.write(f"{InDir}/bin/mg5_aMC -f {OutDir}/script_mH{mass_Phi}_mS{mass_S}.txt")
    #Uncomment this last line if you want to process the data and obtain the plots and limits as well
    #The limits and values will be saved in text files so that you do not have to redo the entire run.
    ##f.write(f"python3 {InDir}/recastingCodes/CalRatioDisplacedJet/Computation_Map_Single.py {mass_Phi} {mass_S} {nevent} {InDir} {OutDir}\n")
    f.close()
    
    os.system(f"chmod +x {OutDir}/Job_mH{mass_Phi}_mS{mass_S}.sh" )
    
    #Depending on where you want to run MadGraph, use the first command line to run locally
    #Or the second one to run in your lab servers -to be updated accordingly, here is for LPC-
    ##os.system(f"{InDir}/bin/mg5_aMC -f {OutDir}/script_mH{mass_Phi}_mS{mass_S}.txt")
    os.system(f"qsub -q prod1C7@clratlserv04 -o {OutDir} -e {OutDir} {OutDir}/Job_mH{mass_Phi}_mS{mass_S}.sh")
    
    
    
