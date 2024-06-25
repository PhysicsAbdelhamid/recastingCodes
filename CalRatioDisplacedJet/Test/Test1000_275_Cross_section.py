import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.stats
import scipy
import time
import readMapNew as rmN
import tqdm
import mplhep as hep
import glob
import random
import os
random.seed(123)
hasHEPData=0

fichier_root = uproot.open("ATLAS_data/HEP_Limits/HEPData-ins2043503-v3-Figure_6f_of_Aux._Mat..root")
objet_graphe = fichier_root["Figure 6f of Aux. Mat./Graph1D_y1;1"]
#objet_graphe = fichier_root["Figure 2c of Aux. Mat. 200_50/Graph1D_y1;1"]


effMG_pythia = "Plots_High/HAHM_old_version/1000_275/Cross_section/Cross_section_mH1000_mS275_nevents10000.txt"
effMG_pythia_data = np.loadtxt(effMG_pythia)

ctau = "Plots_High/HAHM_old_version/1000_275/Cross_section/Cross_section_mH1000_mS275_nevents10000_ctau.txt"
ctau_data = np.loadtxt(ctau)

################## PLOT EFFICIENCY ##################
fig, ax = plt.subplots()

   ################## Plot efficiency from MG ##################
#plt.plot(ctau_data,effMG_data, 'k--',linewidth=2, label = 'MG')

    ################## Plot efficiency from MG+Pythia8 ##################
plt.plot(ctau_data,effMG_pythia_data, 'r', linewidth=2 ,label = 'Map results')
    

   ################## Plot efficiency from HEP data ##################
plt.plot(objet_graphe.values(axis='both')[0],objet_graphe.values(axis='both')[1], 'b',label='Expected results from simulation by ATLAS')

  ################ Uncertainties from HEP ##################

 
    ################## Uncertainties from Map ##################
#plt.fill_between(ctau_data, np.array(effMG_pythia_data) + 0.25* np.array(effMG_pythia_data), np.array(effMG_pythia_data) - 0.25*np.array(effMG_pythia_data), color = (1, 0.5, 0.5) , label='MG+Pythia8, with error bands ',alpha=.7)
#plt.fill_between(objet_graphe.values(axis='both')[0], objet_graphe.values(axis='both')[1] +  objet_graphe.errors('high')[1] , objet_graphe.values(axis='both')[1] - objet_graphe.errors('high')[1] , color = (0.3, 0.3, 1), label = r'ATLAS, with $\pm$ 1 $\sigma$ error bands',alpha=.7)
    ################## Limits of validity ##################
#ax.hlines(y=(0.25*(max(effMG_pythia_data))), xmin=0, xmax=1e2, linewidth=2, color='g', label = 'Limits of validity' )

    # place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.53, 0.83, f" $ m_Φ $ = 1000 GeV, $m_S$ = 275 GeV" , transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
ax.text(0.735, 0.74, f"$Nevents$ = 10000", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

#ax.fill_between(x, 0.25*(max(effMG_pythia_data)), color='black', alpha=.2, hatch="/", edgecolor="black", linewidth=1.0) # adding hatch
plt.ylim(10e-4,10e1) # start at 0
plt.xlim(1e-1, 68.432)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'c$\tau$ [m]', fontsize=13)
plt.ylabel(r'95% CL limit on $\sigma \times B$ [pb]', fontsize=13 )
plt.legend( fontsize = 9, loc=1) # set the legend in the upper right corner
plt.savefig("Test4.png")
plt.close()