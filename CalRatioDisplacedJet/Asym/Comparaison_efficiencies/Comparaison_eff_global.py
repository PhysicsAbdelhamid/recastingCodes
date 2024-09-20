import numpy as np
import matplotlib.pyplot as plt
import glob

# Set the parameters
mass_Phi=600
mass_S1=150
mass_S2=150
nevent=10000

###################################################################################################################################################
# import data from lifetime parametrized model 

files = glob.glob(f"../Results/Results_new/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_*_ct2_*_nevents{nevent}.txt")
print(files)
ct1Values = []
ct2Values = []
effValues = []
for file in files:
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            # Ignore header lines or invalid lines
            if line.startswith("#") or len(line.strip().split()) != 7:
                print(f"Ignoring line: {line.strip()}")
                continue
            
            # Extract row values
            parts = line.strip().split()
            if len(parts) != 7:
                print(f"Unexpected number of values: {len(parts)}. Line content: {line.strip()}")
                continue

            mH, mS1, mS2, ct1, ct2, nevent, eff = parts

            try:
                # Convert ct1 and eff to float and add them to the lists
                ct1Values.append(float(ct1))
                ct2Values.append(float(ct2))
                effValues.append(float(eff))
            except ValueError as ve:
                print(f"ValueError: {ve}. Line content: {line.strip()}")

# Associate ct1Values and effValues
combined = list(zip(ct1Values, ct2Values, effValues))
print(combined)
# Sort by ct1Values
sorted_combined = sorted(combined, key=lambda x: x[0])
# Separate sorted values
sorted_ct1Values, sorted_ct2Values, sorted_effValues = zip(*sorted_combined)
# Convert sorted results into lists
sorted_ct1Values = list(sorted_ct1Values)
sorted_ct2Values = list(sorted_ct2Values)
sorted_effValues = list(sorted_effValues)

#print("Sorted ct1Values:", sorted_ct1Values)
#print("Sorted ct2Values:", sorted_ct2Values)
#print("Corresponding sorted effValues:", sorted_effValues)

# Interpoler les valeurs de l'efficacité précédente pour obtenir les points correspondants
#interp_eff_values = np.interp(sorted_ct1Values, ctau_data1, cross_section_data1)
# Charger les données de la première cross section
#cross_section_file1 = "../Plots_High/HAHM_new_version/400_100/Efficiencies/Efficiency_comparaison_mH400_mS100_nevents10000MG+pythia.txt"
#cross_section_data1 = np.loadtxt(cross_section_file1)
# Charger les données de la première ctau
#ctau_file1 = "../Plots_High/HAHM_new_version/400_100/Efficiencies/Efficiency_comparaison_mH400_mS100_nevents10000_ctau.txt"
#ctau_data1 = np.loadtxt(ctau_file1)

######################################################################################################################################################
# import data from original version (David UFO simplified version)
nevent = "{:.0f}".format(float(nevent))
file1 = f"../Results_ancien/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_nevents{nevent}.txt"
ctau_values = []
efficiency_values = []

with open(file1, 'r') as f:
     next(f)
     for line in f:
             # Split the line into columns
             columns = line.split()
             if len(columns) >= 6:
                 ctau = float(columns[3])
                 efficiency = float(columns[5])
               
                 # Adding values to lists
                 ctau_values.append(ctau)
                 efficiency_values.append(efficiency)

 # Calculate the efficiency ratio
ratio_efficiency = np.array(sorted_effValues) / efficiency_values

 # Create a figure and two axes
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})

#Draw the two cross sections as a function of ctau (blue for the first, red for the second).
ax1.plot(sorted_ct1Values, sorted_effValues, linestyle='-', color='blue', label=f'lifetime parametrized version, nevents={nevent}')
ax1.plot(ctau_values, efficiency_values, linestyle='-', color='red', label=f'original version, nevents={nevent}')
ax1.set_xlabel(r'c$\tau$ [m]')
ax1.set_ylabel(r'Efficiency')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True)
ax1.legend()

ax1.text(0.0, 1.20, f" $ m_Φ $ = {mass_Phi} GeV, $m_{{LLP1}}$ = {mass_S1} GeV, $m_{{LLP2}}$ = {mass_S2} GeV" , transform=ax1.transAxes, fontsize=14, verticalalignment='top')
#ax1.text(0.0, 1.10, f"$Nevents$ = 5000", transform=ax1.transAxes, fontsize=14, verticalalignment='top')
# Tracer le ratio sur le deuxième axe (vert)
ax2.plot(sorted_ct1Values, ratio_efficiency, linestyle='-', color='green', label='ratio')
ax2.set_xlabel(r'c$\tau$ [m]')
ax2.set_ylabel('Ratio of efficiencies')
ax2.set_xscale('log')

ax2.set_ylim(0, 2)
ax2.axhline(y=1, color='black', linestyle='--')
ax2.grid(True)
ax2.legend()

plt.show()
plt.savefig(f'Comparaison_global_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}_new.png')
plt.savefig(f'Comparaison_global_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}_new.pdf')
plt.close()