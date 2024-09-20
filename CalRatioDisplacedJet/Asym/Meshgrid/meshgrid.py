import numpy as np
import matplotlib.pyplot as plt
import glob

# Set the parameters
mass_Phi=200
mass_S1=50
mass_S2=50
nevent=10000

###################################################################################################################################################
# import data from lifetime parametrized model 

files = glob.glob(f"../Results/Results_new/Efficiency_mH{mass_Phi}_mS1_{mass_S1}_mS2_{mass_S2}_ct1_*_ct2_*_nevents{nevent}.txt")
#if not files:
#    print("No files found. Check the directory path and the file naming convention.")
#else:
#    print(f"Files found: {files}")
#print(len(files))

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
                print("ct1 :",ct1Values)
                print("*****************************************************")
                print("ct2 :",ct2Values)
                print("*****************************************************")
                print("efficiency :",effValues)
            except ValueError as ve:
                print(f"ValueError: {ve}. Line content: {line.strip()}")
            
# Convert lists to numpy arrays
ct1Values = np.array(ct1Values)
ct2Values = np.array(ct2Values)
effValues = np.array(effValues)

# Create the scatter plot
plt.figure(figsize=(10, 8))
sc = plt.scatter(ct1Values, ct2Values, c=effValues, cmap='viridis', marker='s', edgecolor='k')
plt.colorbar(sc, label='Efficiency')
plt.xlabel('ct1')
plt.ylabel('ct2')
plt.title('Efficiency Plot: ct1 vs ct2 with Color Gradient for Efficiency')
plt.grid(True)
plt.savefig(f'meshgrid_mH{mass_Phi}_ms1_{mass_S1}_ms2_{mass_S2}_nevents{nevent}.png')
plt.show()
