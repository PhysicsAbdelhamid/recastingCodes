
import numpy as np
import matplotlib.pyplot as plt

# Charger les données de la première cross section
cross_section_file1 = "Plots_High/HAHM_new_version/1000_275/Efficiencies/Efficiency_comparaison_mH1000_mS275_nevents5000MG+pythia.txt"
cross_section_data1 = np.loadtxt(cross_section_file1)

# Charger les données de la première ctau
ctau_file1 = "Plots_High/HAHM_new_version/1000_275/Efficiencies/Efficiency_comparaison_mH1000_mS275_nevents5000_ctau.txt"
ctau_data1 = np.loadtxt(ctau_file1)

# Charger les données de la deuxième cross section
cross_section_file2 = "Plots_High/HAHM_old_version/1000_275/Efficiencies/Efficiency_comparaison_mH1000_mS275_nevents5000MG+pythia.txt"
cross_section_data2 = np.loadtxt(cross_section_file2)

# Charger les données de la deuxième ctau
ctau_file2 = "Plots_High/HAHM_old_version/1000_275/Efficiencies/Efficiency_comparaison_mH1000_mS275_nevents5000_ctau.txt"
ctau_data2 = np.loadtxt(ctau_file2)

# Créer une figure et deux axes
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

# Tracer les deux cross sections en fonction de ctau (bleu pour la première, rouge pour la deuxième)
ax1.plot(ctau_data1, cross_section_data1, linestyle='-', color='blue', label='HAHM_simplified_UFO')
ax1.plot(ctau_data2, cross_section_data2, linestyle='-', color='red', label='HAHM_variableMW_v3_UFO')

ax1.set_xlabel(r'c$\tau$ [m]')
ax1.set_ylabel(r'Efficiency')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True)
ax1.legend()

ax1.text(0.0, 1.20, f" $ m_Φ $ = 1000 GeV, $m_S$ = 275 GeV" , transform=ax1.transAxes, fontsize=14, verticalalignment='top')
ax1.text(0.0, 1.10, f"$Nevents$ = 5000", transform=ax1.transAxes, fontsize=14, verticalalignment='top')

# Calculer le ratio des cross sections
ratio_cross_section = cross_section_data1 / cross_section_data2

# Tracer le ratio sur le deuxième axe (vert)
ax2.plot(ctau_data1, ratio_cross_section, linestyle='-', color='green', label='ratio')
ax2.set_xlabel(r'c$\tau$ [m]')
ax2.set_ylabel('Ratio of efficiencies')
ax2.set_xscale('log')

ax2.set_ylim(0, 2)
ax2.axhline(y=1, color='black', linestyle='--')
ax2.grid(True)
ax2.legend()

plt.show()
plt.savefig('Comparaison/Efficiencies/Efficiencies_mH1000_mS275_nevents5000.png')