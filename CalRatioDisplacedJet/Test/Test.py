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

#HEP data file
file_HEP = glob.glob("ATLAS_data/HEP_Limits/HEPData-ins2043503-v3-Figure_6f_of_Aux._Mat..root")

#fichier_root = uproot.open("ATLAS_data/HEP_Limits/HEPData-ins2043503-v3-Figure_2f_of_Aux._Mat._1000_275.root")
#objet_graphe = fichier_root["Figure 2f of Aux. Mat./Graph1D_y1;1"]

# Ouvrir le fichier ROOT
fichier_root = uproot.open("ATLAS_data/HEP_Limits/HEPData-ins2043503-v3-Figure_6f_of_Aux._Mat..root")

# Accéder à l'objet contenant le graphe
objet_graphe = fichier_root["Figure 6f of Aux. Mat./Graph1D_y1;1"]

# Afficher les noms des membres de l'objet
print("Noms des membres de l'objet :", objet_graphe.member_names)

print("Liste des attributs:", dir(objet_graphe))

# Accéder aux valeurs de l'objet
valeurs = objet_graphe.values()
erreur = objet_graphe.errors('high')

# Récupérer les valeurs de f(x)
valeurs_fx = valeurs[1]
valeurs_x = valeurs[0]
erreur1 = erreur[1]

# Convertir les valeurs en une liste
liste_valeurs = list(valeurs_fx)
liste_valeurs_x = list(valeurs_x)
liste_erreur1 = erreur1
# Afficher les premiers éléments de la liste de valeurs
print("Liste de valeurs de f(x) :", liste_valeurs)  # Affiche les 10 premières valeurs de f(x)
print("liste de valeurs de x :", liste_valeurs_x )
print("erreur :", liste_erreur1)
#ctau = objet_graphe.members
#print ("ctau?", ctau)

#axe des ordonnées
cross_section_file2 = "Plots_Low/HAHM_old_version/200_50/Efficiencies/Efficiency_comparaison_mH200_mS50_nevents10000_MG+pythia.txt"
cross_section_data2 = np.loadtxt(cross_section_file2)

#axe des abscisse
ctau_file2 = "Plots_Low/HAHM_old_version/200_50/Efficiencies/Efficiency_comparaison_mH200_mS50_nevents10000_ctau.txt"
ctau_data2 = np.loadtxt(ctau_file2)

# Plot des données
#plt.plot(ctau_data2, cross_section_data2, label='cross_section_data2')
plt.plot(liste_valeurs_x, liste_valeurs, label='liste_valeurs')

# Labels et légendes
plt.xlabel('ctau_data2')
plt.ylabel('Valeurs')
plt.title('Plot des données')
plt.legend()

# Afficher le graphique
plt.show()
plt.savefig('Test_test.png')

