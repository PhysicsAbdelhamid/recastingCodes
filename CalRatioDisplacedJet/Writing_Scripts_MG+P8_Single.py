# This code write script that MG and Pythia will read to generate events
import os
import sys

mass_Phi = int(sys.argv[1])
mass_S = int(sys.argv[2])
nevent = int(sys.argv[3])

InDir = sys.argv[4]
OutDir = sys.argv[5]


f = open(f"{OutDir}/script_mH{mass_Phi}_mS{mass_S}.txt", 'w') # Creation of the Script to follow for MG.
f.write(f"import model {InDir}/HAHM_MG5model_v3/HAHM_variableMW_v3_UFO \n") # Import the model.
f.write(f"define f = u c d s u~ c~ d~ s~ b b~ e+ e- mu+ mu- ta+ ta- t t~ \n") # Define a fermion.
f.write(f"generate g g > h HIG=1 HIW=0 QED=0 QCD=0, (h > h2 h2, h2 > f f) \n") # Generate the process.
f.write(f"output Script_mH{mass_Phi}_mS{mass_S} \n")
f.write(f"launch Script_mH{mass_Phi}_mS{mass_S} \n")
f.write(f"shower=Pythia8 \n") #Add Pythia
f.write(f"0 \n") #Launch the computation
f.write(f"set nevents = {nevent} \n" ) # change the number of event
f.write(f"set mhsinput {mass_S} \n") # Set a mass for the LLP.
f.write(f"set mhinput {mass_Phi} \n") # Set a mass for the Heavy Neutral Boson.
f.write(f"set epsilon 1e-10 \n") # Set the couplings for epsilon.
f.write(f"set kap 1e-4 \n") # Set the couplings for kappa.
f.write(f"set time_of_flight 0 \n" ) # Set the time of flight of the particle.
f.write(f"set event_norm = sum \n" )
f.write(f"set ptj = 0 \n" ) # Disable the cuts.
f.write(f"set ptb = 0 \n" )
f.write(f"set pta = 0 \n" )
f.write(f"set ptl = 0 \n" )
f.write(f"set etaj = -1 \n" )
f.write(f"set etab = -1 \n" )
f.write(f"set etaa = -1 \n" )
f.write(f"set etal = -1 \n" )
f.write(f"set drjj = 0 \n" )
f.write(f"set drbb = 0 \n" )
f.write(f"set drll = 0 \n" )
f.write(f"set draa = 0 \n" )
f.write(f"set drbj = 0 \n" )
f.write(f"set draj = 0 \n" )
f.write(f"set drjl = 0 \n" )
f.write(f"set drab = 0 \n" )
f.write(f"set drbl = 0 \n" )
f.write(f"set dral = 0 \n" )

if mass_Phi <= 125: 
    f.write(f"set cut_decays = F\n" )
    f.write(f"set wzp = Auto\n"  ) # Set the calculation of the Width.
    f.write(f"set wh = Auto\n" )
    f.write(f"set wt = Auto\n" )
    f.write(f"set whs = Auto\n" )
    f.write(f"set lhe_version = 3.0\n" )
else:
    f.write(f"set use_syst = T \n" )
    f.write(f"set wzp = 5\n" )
    f.write(f"set wh = 5\n" )
    f.write(f"set wt = Auto\n" )
    f.write(f"set whs = 5\n" )

f.write(f"0 \n") #launch the generation
f.write(f"exit")
