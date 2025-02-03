# This code write script that MG and Pythia will read to generate events
import os
import sys

mode = sys.argv[1]
mass_Chi = float(sys.argv[2])
nevent = int(sys.argv[3])
slug = f'mChi{mass_Chi:.6g}_{mode}' 
InDir = sys.argv[4]
OutDir = sys.argv[5]

f = open(f"{OutDir}/script_{slug}.txt", 'w')
f.write(f"set auto_convert_model T \n")
f.write(f"import model sm \n")
f.write(f"import model {InDir}/WIMP_BG_higgsportal_full_loop \n")
f.write(f"define p = g u c d s u~ c~ d~ s~ \n")
f.write(f"define l- = e- mu- ta- \n")
f.write(f"define l+ = e+ mu+ ta+ \n")
f.write(f"define vl = ve vm vt \n")
f.write(f"define vl~ = ve~ vm~ vt~ \n")
f.write(f"define j = g u c d s u~ c~ d~ s~ \n")
f.write(f"define f = u c d s u~ c~ d~ s~ b b~ e+ e- mu+ mu- ta+ ta- t t~ \n")
if mode == "Tau":
    f.write(f"generate p p > xchi xchi , xchi > ta- ta+ vl , xchi > ta+ ta- vl~ \n")
if mode == "cbs":
    f.write(f"generate p p > xchi xchi , xchi > c b s      , xchi > c~ b~ s~ \n")
if mode == "bb":
    f.write(f"generate p p > xchi xchi , xchi > vl b b~    , xchi > vl~ b~ b \n")
f.write(f"output Script_{slug} \n")
f.write(f"launch Script_{slug} \n")
f.write(f"shower=Pythia8 \n")  # Add Pythia
f.write(f"0 \n")  # Launch the computation
f.write(f"set nevents = {nevent} \n")  # Generate 10% extra events?
f.write(f"set Mxchi {mChi} \n")  # Set mass of chi
f.write(f"set epsilon 1e-10 \n") # Set the couplings for epsilon.
f.write(f"set kap 1e-4 \n") # Set the couplings for kappa.
f.write(f"set time_of_flight 0 \n" ) # Set the time of flight of the particle.
f.write(f"set event_norm = sum \n")
f.write(f"set lhe_version = 3.0\n" )
f.write(f"set cut_decays = F \n")
f.write(f"set pdlabel = pdflabel \n")  # Disable the cuts.
f.write(f"set lhaid = lhaid \n")  
  
f.write(f"set ptj = 0 \n")
f.write(f"set ptb = 0 \n")
f.write(f"set pta = 0 \n")
f.write(f"set ptl = 0 \n")
f.write(f"set etaj = -1 \n")
f.write(f"set etab = -1 \n")
f.write(f"set etaa = -1 \n")
f.write(f"set etal = -1 \n")
f.write(f"set drjj = 0 \n")
f.write(f"set drbb = 0 \n")
f.write(f"set drll = 0 \n")
f.write(f"set draa = 0 \n")
f.write(f"set drbj = 0 \n")
f.write(f"set draj = 0 \n")
f.write(f"set drjl = 0 \n")
f.write(f"set drab = 0 \n")
f.write(f"set drbl = 0 \n")
f.write(f"set dral = 0 \n")   
 
f.write(f"set use_syst = T \n")
f.write(f"set sys_scalefact 1 0.5 2 \n")
f.write(f"set sys_pdf NNPDF31_lo_as_0118 \n")    
f.write(f"0 \n")  # Launch the generation
f.write(f"exit \n")
