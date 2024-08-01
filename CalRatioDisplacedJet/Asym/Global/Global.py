import glob
import matplotlib.pyplot as plt

# Set the parameters
mass_Phi=200
mass_S1=50
mass_S2=50
nevent=10000


###################################################################################################################################################

# FOR LIFETIME PARAMETRIZED MODEL 
files = glob.glob(f"../Results/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_*_ct2_*_nevents{nevent}.txt")
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
print('sorted_combined',sorted_combined)
# Separate sorted values
sorted_ct1Values, sorted_ct2Values, sorted_effValues = zip(*sorted_combined)
# Convert sorted results into lists
sorted_ct1Values = list(sorted_ct1Values)
sorted_ct2Values = list(sorted_ct2Values)
sorted_effValues = list(sorted_effValues)

#print("Sorted ct1Values:", sorted_ct1Values)
#print("Sorted ct2Values:", sorted_ct2Values)
#print("Corresponding sorted effValues:", sorted_effValues)
nevent = "{:.0f}".format(float(nevent))
#PLOT EFFICIENCY 
fig, ax = plt.subplots()
plt.plot(sorted_ct1Values, sorted_effValues, 'b',linewidth=2, label='lifetime parametrized values')
# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.55, 0.75, f" $ m_Φ $ = {mass_Phi} GeV, $m_{{LLP1}}$ = {mass_S1} GeV, $m_{{LLP2}}$ = {mass_S2} GeV " , transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
ax.text(0.55, 0.65, f"$Nevents$ = {nevent}", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
#x = np.linspace(0,100)
#plt.ylim(0) # start at 0
plt.xscale('log')
plt.xlabel(r'c$\tau$ [m]',fontsize=10)
plt.ylabel(r'Efficiency',fontsize=10)
plt.title('Efficiency vs ct1')
plt.legend( fontsize = 10, loc=1) # set the legend in the upper right corner
plt.savefig(f"Global_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}_lifetime_parametrized.png")
#plt.savefig(f"Global_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}_lifetime_parametrized.pdf")
plt.close()


##################################################################################################################################################

#FOR ORIGNIAL VERSION (DAVID UFO SIMPLIFIED VERSION)
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

# PLOT EFFICIENCY 
fig, ax = plt.subplots()
plt.plot(ctau_values, efficiency_values, linestyle='-', color='red', label=f'original version, nevents={nevent}')
# place a text box in upper left in axes coords
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(0.55, 0.75, f" $ m_Φ $ = {mass_Phi} GeV, $m_{{LLP1}}$ = {mass_S1} GeV, $m_{{LLP2}}$ = {mass_S2} GeV " , transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
ax.text(0.55, 0.65, f"$Nevents$ = {nevent}", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
#x = np.linspace(0,100)
#plt.ylim(0) # start at 0
plt.xscale('log')
plt.xlabel(r'c$\tau$ [m]',fontsize=10)
plt.ylabel(r'Efficiency',fontsize=10)
plt.title('Efficiency vs ct1')
plt.legend( fontsize = 10, loc=1) # set the legend in the upper right corner
plt.savefig(f"Global_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}_original_version.png")
#plt.savefig(f"Global_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}_original_version.pdf")
plt.close()
