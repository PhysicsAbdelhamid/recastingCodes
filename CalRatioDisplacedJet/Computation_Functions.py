import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import scipy
import time
import readMapNew as rmN
import tqdm
import mplhep as hep
import lhe_parser as lhe
import hepmc_parser as hepmc
import uproot
import random
import os
import pickle
random.seed(123)
hep.style.use("ATLAS")

# set cst
c = 3e8# Light velocity in m/s

#All plots are made with 10 000 events, if you want to try with other numbers of events, you will have to change the number of events in lines 285-314-551-580 for the calculation of the efficiency.

#########################################################################################
#Parsing the hepmc file from the hadronization of the MG outputs to recover the data from the process.
#########################################################################################

def parsing_hepmc(events):

    px_TOT = []
    py_TOT = []
    pz_TOT = []
    E_TOT = []
    mass_TOT = []
    pdg_TOT = []

    for ie , event in enumerate(events):
        count=0
        for id, vertex in event.vertex.items():
            if [p.pdg for p in vertex.incoming] == [25] and [p.pdg for p in vertex.outcoming] == [35, 35]: # PDGID 25 = Higgs, PDGID 35 Dark Higgs
                px_TOT.append(list(p.px for p in vertex.outcoming)) # recover the x momenta in GeV
                py_TOT.append(list(p.py for p in vertex.outcoming)) # recover the y momenta in GeV
                pz_TOT.append(list(p.pz for p in vertex.outcoming)) # recover the z momenta in GeV
                E_TOT.append(list(p.E for p in vertex.outcoming)) # recover the Energy in GeV
                mass_TOT.append(list(p.mass for p in vertex.outcoming)) # recover the mass in GeV

            if [p.pdg for p in vertex.incoming] == [35]: # PDGID 35 Dark Higgs
                pdg_TOT.append((list(p.pdg for p in vertex.outcoming))) # recover the PDG ID of the particle produced

                count = count+1
                if count==2: ##
                    break
                    pass

    return px_TOT, py_TOT, pz_TOT, E_TOT, mass_TOT, pdg_TOT

def parsing_hepmc_ALP(events):

    px_TOT = []
    py_TOT = []
    pz_TOT = []
    E_TOT = []
    mass_TOT = []
    pdg_TOT = []

    for ie , event in enumerate(events):
        count=0
        for id, vertex in event.vertex.items():
            particles = sorted([p for p in vertex.outcoming], key = lambda k : k.pdg)
            particlesIn = sorted([p for p in vertex.incoming], key = lambda k : k.pdg)
            if 9000005 in [p.pdg for p in particles]  and len(particles)==2 : # PDGID 25 = Higgs, PDGID 35 Dark Higgs
                px_TOT.append(list(p.px for p in particles)) # recover the x momenta in GeV
                py_TOT.append(list(p.py for p in particles)) # recover the y momenta in GeV
                pz_TOT.append(list(p.pz for p in particles)) # recover the z momenta in GeV
                E_TOT.append(list(p.E for p in particles)) # recover the Energy in GeV
                mass_TOT.append(list(p.mass for p in particles)) # recover the mass in GeV

            if [p.pdg for p in vertex.incoming] == [9000005] and len(particles)==2: # PDGID 35 Dark Higgs
                pdg_TOT.append((list(p.pdg for p in vertex.outcoming))) # recover the PDG ID of the particle produced

                count = count+1
                if count==2: ##
                    break
                    pass

    return px_TOT, py_TOT, pz_TOT, E_TOT, mass_TOT, pdg_TOT

#########################################################################################
#The data recovered are list of list, we need to convert them into one list to be able to separate the contribution of each LLP.
#########################################################################################

def conversion_one_list(px_TOT, py_TOT, pz_TOT, E_TOT, mass_TOT, pdg_TOT):

    px_tot = []
    for i in range(len(px_TOT)):
        for y in range(len(px_TOT[i])):
            px_tot.append(px_TOT[i][y])

    py_tot = []
    for i in range(len(py_TOT)):
        for y in range(len(py_TOT[i])):
            py_tot.append(py_TOT[i][y])

    pz_tot = []
    for i in range(len(pz_TOT)):
        for y in range(len(pz_TOT[i])):
            pz_tot.append(pz_TOT[i][y])

    E_tot = []
    for i in range(len(E_TOT)):
        for y in range(len(E_TOT[i])):
            E_tot.append(E_TOT[i][y])

    mass_tot = []
    for i in range(len(mass_TOT)):
        for y in range(len(mass_TOT[i])):
            mass_tot.append(mass_TOT[i][y])

    pdg_tot = []
    for i in range(len(pdg_TOT)):
        for y in range(len(pdg_TOT[i])):
            pdg_tot.append(pdg_TOT[i][y])

    return px_tot, py_tot, pz_tot, E_tot, mass_tot, pdg_tot

#########################################################################################
# Recovering the data from each LLP (px,py,pz,E,mass,PDG ID).
#########################################################################################

def recover(px_tot, py_tot, pz_tot, E_tot, mass_tot,pdg_tot):

    px_DH1 = []
    px_DH2 = []

    py_DH1 = []
    py_DH2 = []

    pz_DH1 = []
    pz_DH2 = []

    E_DH1 = []
    E_DH2 = []

    mass_DH1 = []
    mass_DH2 = []

    pdg_tot_DH1 = []
    pdg_tot_DH2 = []

    for i in range(0, len(px_tot),2): # in the list, each even value is for DH1
        px_DH1.append(px_tot[i])
        py_DH1.append(py_tot[i])
        pz_DH1.append(pz_tot[i])
        E_DH1.append(E_tot[i])
        mass_DH1.append(mass_tot[i])

    for i in range(1, len(px_tot),2): # in the list, each odd value is for DH2
        px_DH2.append(px_tot[i])
        py_DH2.append(py_tot[i])
        pz_DH2.append(pz_tot[i])
        E_DH2.append(E_tot[i])
        mass_DH2.append(mass_tot[i])

    for i in range(0, len(pdg_tot),4): # recover the PDG ID of the particle produced by decay of DH1
        pdg_tot_DH1.append(pdg_tot[i])

    for i in range(2, len(pdg_tot),4): # recover the PDG ID of the particle produced by decay of DH2
        pdg_tot_DH2.append(pdg_tot[i])

    # Convert all lists into arrays

    px_DH1 = np.array(px_DH1)/c # GeV/c
    py_DH1 = np.array(py_DH1)/c
    pz_DH1 = np.array(pz_DH1)/c
    E_DH1 = np.array(E_DH1)
    mass_DH1 = np.array(mass_DH1)
    pdg_tot_DH1 = np.array(pdg_tot_DH1)

    px_DH2 = np.array(px_DH2)/c
    py_DH2 = np.array(py_DH2)/c
    pz_DH2 = np.array(pz_DH2)/c
    E_DH2 = np.array(E_DH2)
    mass_DH2 = np.array(mass_DH2)
    pdg_tot_DH2 = np.array(pdg_tot_DH2)

    return px_DH1, px_DH2, py_DH1, py_DH2, pz_DH1, pz_DH2, pdg_tot_DH1, pdg_tot_DH2, E_DH1, E_DH2, mass_DH1, mass_DH2

#########################################################################################
# Computation of the kinematics variable for LLP1 (velocities, beta, gamma, pT the transverse momenta, eta the pseudo-rapidity).
#########################################################################################

def kinematics_DH1(px_DH1, py_DH1, pz_DH1, E_DH1):

    vx_DH1 = (px_DH1*c**2)/E_DH1 #compute the velocities in each direction
    vy_DH1 = (py_DH1*c**2)/E_DH1
    vz_DH1 = (pz_DH1*c**2)/E_DH1
    beta_DH1 = np.sqrt(vx_DH1**2 + vy_DH1**2 + vz_DH1**2)/c # compute beta
    gamma_DH1 = 1/(np.sqrt(1-beta_DH1**2)) # compute gamma

    pT_DH1 = np.sqrt(px_DH1**2 + py_DH1**2)*c # compute the transverse momenta
    eta_DH1 = np.arctanh(pz_DH1/(np.sqrt(px_DH1**2 + py_DH1**2 + pz_DH1**2))) # compute the pseudorapidity

    return beta_DH1, gamma_DH1, pT_DH1, eta_DH1


#########################################################################################
# Computation of the kinematics variable for LLP2 (velocities, beta, gamma, pT the transverse momenta, eta the pseudo-rapidity).
#########################################################################################

def kinematics_DH2(px_DH2, py_DH2, pz_DH2, E_DH2):

    vx_DH2 = (px_DH2*c**2)/E_DH2 #compute the velocities in each direction
    vy_DH2 = (py_DH2*c**2)/E_DH2
    vz_DH2 = (pz_DH2*c**2)/E_DH2
    beta_DH2 = np.sqrt(vx_DH2**2 + vy_DH2**2 + vz_DH2**2)/c # compute beta
    gamma_DH2 = 1/(np.sqrt(1-beta_DH2**2)) # compute gamma

    pT_DH2 = np.sqrt(px_DH2**2 + py_DH2**2)*c # compute the transverse momenta
    eta_DH2 = np.arctanh(pz_DH2/(np.sqrt(px_DH2**2 + py_DH2**2 + pz_DH2**2))) # compute the pseudorapidity

    return beta_DH2, gamma_DH2, pT_DH2, eta_DH2

#########################################################################################
# lifetime function.
#########################################################################################

def lifetime(avgtau = 4.3):
    import math
    avgtau = avgtau / c
    t = random.random()
    return -1.0 * avgtau * math.log(t)

#########################################################################################
# Decay lenght computation for LLP1.
#########################################################################################

def decaylenghtDH1(px_DH1, py_DH1, pz_DH1, E_DH1, gamma_DH1, tauN):

    Lx_tot_DH1 = []
    Ly_tot_DH1 = []
    Lz_tot_DH1 = []
    Lxy_tot_DH1 = []

    for ctau in range(len(tauN)):

        Lx_DH1 = []
        Ly_DH1 = []
        Lz_DH1 = []
        Lxy_DH1 = []

        for i in range(len(gamma_DH1)):
            lt = lifetime(tauN[ctau]) # set mean lifetime
            Lx_DH1.append((px_DH1[i]/E_DH1[i])*c**2 * lt * gamma_DH1[i]) # compute the decay lenght in x,y,z
            Ly_DH1.append((py_DH1[i]/E_DH1[i])*c**2 * lt * gamma_DH1[i])
            Lz_DH1.append((abs(pz_DH1[i])/E_DH1[i])*c**2 * lt  * gamma_DH1[i] )
            Lxy_DH1.append(np.sqrt((Lx_DH1[i])**2 + (Ly_DH1[i])**2)) # compte the transverse decay lenght

        Lx_tot_DH1.append(Lx_DH1)
        Ly_tot_DH1.append(Ly_DH1)
        Lz_tot_DH1.append(Lz_DH1)
        Lxy_tot_DH1.append(Lxy_DH1)
    return Lxy_tot_DH1, Lz_tot_DH1

#########################################################################################
# Decay lenght computation for LLP2.
#########################################################################################

def decaylenghtDH2(px_DH2, py_DH2, pz_DH2, E_DH2, gamma_DH2, tauN):

    Lx_tot_DH2 = []
    Ly_tot_DH2 = []
    Lz_tot_DH2 = []
    Lxy_tot_DH2 = []

    for ctau in range(len(tauN)):
        Lx_DH2 = []
        Ly_DH2 = []
        Lz_DH2 = []
        Lxy_DH2 = []

        for i in range(len(gamma_DH2)):
                lt = lifetime(tauN[ctau]) # set mean lifetime
                Lx_DH2.append((px_DH2[i]/E_DH2[i])*c**2 * lt * gamma_DH2[i]) # compute the decay lenght in x,y,z
                Ly_DH2.append((py_DH2[i]/E_DH2[i])*c**2 * lt * gamma_DH2[i])
                Lz_DH2.append((abs(pz_DH2[i])/E_DH2[i])*c**2 * lt* gamma_DH2[i])
                Lxy_DH2.append(np.sqrt((Lx_DH2[i])**2 + (Ly_DH2[i])**2)) # compte the transverse decay lenght

        Lx_tot_DH2.append(Lx_DH2)
        Ly_tot_DH2.append(Ly_DH2)
        Lz_tot_DH2.append(Lz_DH2)
        Lxy_tot_DH2.append(Lxy_DH2)
    return Lxy_tot_DH2, Lz_tot_DH2

#########################################################################################
# Computation of the efficiency with the map from the data obtained with MG+Pythia8 for the high-ET samples (mH >= 400GeV).
#########################################################################################

def eff_map_High(pT_DH1, eta_DH1, Lxy_tot_DH1, Lz_tot_DH1, pdg_tot_DH1, pT_DH2, eta_DH2, Lxy_tot_DH2, Lz_tot_DH2, pdg_tot_DH2, tauN, nevent, mass_phi, mass_s):

    eff_highETX = []

    for index in tqdm.tqdm(range(len(tauN))):
        queryMapResult = []
        for iEvent in range(len(pT_DH1)):
            queryMapResult.append(rmN.queryMapFromKinematics(pT_DH1[iEvent],
                                                            eta_DH1[iEvent],
                                                            Lxy_tot_DH1[index][iEvent],
                                                            Lz_tot_DH1[index][iEvent],
                                                            abs(pdg_tot_DH1[iEvent]),
                                                            pT_DH2[iEvent],
                                                            eta_DH2[iEvent],
                                                            Lxy_tot_DH2[index][iEvent],
                                                            Lz_tot_DH2[index][iEvent],
                                                            abs(pdg_tot_DH2[iEvent]),
                                                            selection = "high-ET"))
        eff_highETX.append(sum(queryMapResult))
    queryMapResult = np.array(queryMapResult) #convertion into array
    eff_highETX = np.array(eff_highETX) #convertion into array
    eff_highETX = eff_highETX/nevent #efficiency/(nbr of event)

    Data_Eff_High = np.column_stack(eff_highETX)
    np.savetxt(f'./Plots_High/Efficiencies_Text_{mass_phi}_{mass_s}.txt', Data_Eff_High)

    return eff_highETX

#########################################################################################
# Computation of the efficiency with the map from the data obtained with MG+Pythia8 for the low-ET samples (mH <= 400GeV).
#########################################################################################

def eff_map_Low(pT_DH1, eta_DH1, Lxy_tot_DH1, Lz_tot_DH1, pdg_tot_DH1, pT_DH2, eta_DH2, Lxy_tot_DH2, Lz_tot_DH2, pdg_tot_DH2, tauN,nevent, mass_phi, mass_s):

    eff_lowETX = []

    for index in tqdm.tqdm(range(len(tauN))):
        queryMapResult = []
        for iEvent in range(len(pT_DH1)):
            queryMapResult.append(rmN.queryMapFromKinematics(pT_DH1[iEvent],
                                                            eta_DH1[iEvent],
                                                            Lxy_tot_DH1[index][iEvent],
                                                            Lz_tot_DH1[index][iEvent],
                                                            abs(pdg_tot_DH1[iEvent]),
                                                            pT_DH2[iEvent],
                                                            eta_DH2[iEvent],
                                                            Lxy_tot_DH2[index][iEvent],
                                                            Lz_tot_DH2[index][iEvent],
                                                            abs(pdg_tot_DH2[iEvent]),
                                                            selection = "low-ET"))
        eff_lowETX.append(sum(queryMapResult))
    queryMapResult = np.array(queryMapResult) #convertion into array
    eff_lowETX = np.array(eff_lowETX) #convertion into array
    eff_lowETX = eff_lowETX/nevent #efficiency/(nbr of event)

    Data_Eff_Low = np.column_stack(eff_lowETX)
    np.savetxt(f'./Plots_Low/Efficiencies_Text_{mass_phi}_{mass_s}.txt', Data_Eff_Low)

    return eff_lowETX

def eff_bdt_WALP(pT_V, eta_V, Lxy_tot_V, Lz_tot_V, pdg_tot_V, pT_ALP, eta_ALP, Lxy_tot_ALP, Lz_tot_ALP, pdg_tot_ALP, E_V, E_ALP, tauN, nevent, mass_phi, mass_s):
      return eff_bdt_tauN(pT_V, eta_V, Lxy_tot_V, Lz_tot_V, pdg_tot_V, pT_ALP, eta_ALP, Lxy_tot_ALP, Lz_tot_ALP, pdg_tot_ALP, E_V, E_ALP,tauN, nevent, mass_phi, mass_s, "WALP")

def eff_bdt_High(pT_DH1, eta_DH1, Lxy_tot_DH1, Lz_tot_DH1, pdg_tot_DH1, pT_DH2, eta_DH2, Lxy_tot_DH2, Lz_tot_DH2, pdg_tot_DH2, E_DH1, E_DH2, tauN, nevent, mass_phi, mass_s):
    return eff_bdt_tauN(pT_DH1, eta_DH1, Lxy_tot_DH1, Lz_tot_DH1, pdg_tot_DH1, pT_DH2, eta_DH2, Lxy_tot_DH2, Lz_tot_DH2, pdg_tot_DH2, E_DH1, E_DH2,tauN, nevent, mass_phi, mass_s, "sel1h_036")

def eff_bdt_Low(pT_DH1, eta_DH1, Lxy_tot_DH1, Lz_tot_DH1, pdg_tot_DH1, pT_DH2, eta_DH2, Lxy_tot_DH2, Lz_tot_DH2, pdg_tot_DH2,E_DH1, E_DH2, tauN, nevent, mass_phi, mass_s):
    return eff_bdt_tauN(pT_DH1, eta_DH1, Lxy_tot_DH1, Lz_tot_DH1, pdg_tot_DH1, pT_DH2, eta_DH2, Lxy_tot_DH2, Lz_tot_DH2, pdg_tot_DH2,E_DH1, E_DH2, tauN, nevent, mass_phi, mass_s, "sel1l_027")
    
def eff_bdt_tauN(pT_DH1, eta_DH1, Lxy_tot_DH1, Lz_tot_DH1, pdg_tot_DH1, pT_DH2, eta_DH2, Lxy_tot_DH2, Lz_tot_DH2, pdg_tot_DH2, E_DH1, E_DH2, tauN, nevent, mass_phi, mass_s, sel):
   eff = []
   scaler_mean, scaler_std, clf =  load_model(sel)
   llp1_ET =  np.sqrt(pT_DH1**2 + mass_s**2)
   llp2_ET =  np.sqrt(pT_DH2**2 + mass_s**2)
   for index in tqdm.tqdm(range(len(tauN))):
        if    np.array(Lxy_tot_DH2[index])[np.array(eta_DH2) > 1.5].mean() < 1*0.25 : thisEffA= -1
        elif  np.array(Lxy_tot_DH2[index])[np.array(eta_DH2) > 1.5].mean() > 4*4 : thisEffA= -1
        elif  np.array(Lz_tot_DH2[index])[np.array(eta_DH2) < 1.5].mean()  < 3*0.25 : thisEffA= -1
        elif  np.array(Lz_tot_DH2[index])[np.array(eta_DH2) < 1.5].mean()  > 7*4 : thisEffA= -1
        else:
          thisEffA = bdt_eval(pT_DH1, eta_DH1, Lxy_tot_DH1[index], Lz_tot_DH1[index], pdg_tot_DH1, pT_DH2, eta_DH2, Lxy_tot_DH2[index], Lz_tot_DH2[index], pdg_tot_DH2, llp1_ET, llp2_ET,E_DH1, E_DH2, clf, scaler_mean, scaler_std, sel=sel) 
        eff.append(thisEffA)
   return eff

def load_model(sel):
   modelDir = "models"
   scaler_mean =  np.load(f"{modelDir}/{sel}_scaler_mean.npy")
   scaler_std =  np.load(f"{modelDir}/{sel}_scaler_std.npy")
   f = open(f'{modelDir}/{sel}_model.pkl', 'rb')
   clf = pickle.load(f)
   reg = None
   try:
     fw = open(f'{modelDir}/{sel}_model_weights.pkl', 'rb')
     reg = pickle.load(fw)
   except:
     pass
   return scaler_mean, scaler_std, [clf, reg]

def bdt_eval(llp1_pT, llp1_eta, llp1_Lxy, llp1_Lz, llp1_child_pdgId, llp2_pT, llp2_eta, llp2_Lxy, llp2_Lz, llp2_child_pdgId, llp1_ET, llp2_ET, llp1_E, llp2_E,clf=None, mean=None, std=None, sel=None):
  if clf is None:
    mean, std, clfAndReg =  load_model(sel)
  clf, reg = clf # bundled with regressor if it exists

  #X = np.array([np.array(llp1_Lxy)*1000, np.array(llp2_Lxy)*1000, np.array(llp1_Lz)*1000, np.array(llp2_Lz)*1000, llp1_eta, llp2_eta, llp1_pT*1000, llp2_pT*1000, llp1_child_pdgId, llp2_child_pdgId]).T
  #X = np.array([np.array(llp1_Lxy)*1000, np.array(llp2_Lxy)*1000, np.array(llp1_Lz)*1000, np.array(llp2_Lz)*1000, llp1_eta, llp2_eta, llp1_pT*1000, llp2_pT*1000,  llp1_ET*1000, llp2_ET*1000, llp1_child_pdgId, llp2_child_pdgId]).T
  #print("LC DEBUG Lxy", np.array(llp2_Lxy).mean()) 
  #print("LC DEBUG Lz", np.array(llp2_Lz).shape) 
  #print("LC DEBUG eTA", np.array(llp2_eta).shape) 
  #print("LC DEBUG eT", np.array(llp2_ET).shape) 
  #print("LC DEBUG pdg", np.array(llp2_child_pdgId).shape) 
  if sel=="WALP":
    X = np.array([np.array(llp2_Lxy)*1000, np.array(llp2_Lz)*1000, llp2_eta, llp2_pT*1000, llp2_ET*1000, llp2_child_pdgId,llp1_eta, np.array(llp1_pT)*1000]).T # WALP
  else:
    X = np.array([np.array(llp1_Lxy)*1000, np.array(llp2_Lxy)*1000, np.array(llp1_Lz)*1000, np.array(llp2_Lz)*1000, llp1_eta, llp2_eta, llp1_pT*1000, llp2_pT*1000,  llp1_ET*1000, llp2_ET*1000, llp1_child_pdgId, llp2_child_pdgId]).T
  #print("MEANs", list(mean))
  #print("MEAN our sample", X.T.mean(axis=1))

  #print("STDs", list(std))
  #print("STD our sample", X.T.std(axis=1))
  X = (X - mean) /std
  pred_proba = clf.predict_proba(X).T
  den = len(pred_proba[1])
  #if reg is not None: 
  #  weights = reg.predict(X)
  #  pred_proba *= weights
  #  den = sum(weights)
  #print("LC DEBUG X rescaled",  X)
  #print("LC DEBUG len of proba ",  pred_proba)
  return sum(pred_proba[1])/den

   

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################

#########################################################################################
#Parsing the lhe file from the MG output to recover the data from the process.
#########################################################################################


def parsing_LHE(MG_events):
    px = []
    py = []
    pz = []
    pdg = []
    E = []
    MASS = []
    for event in MG_events:
        for particle in event:
            pdg.append(particle.pdg)
            if particle.pdg == 35: # PDGID 35 Dark Higgs
                px.append(particle.px)
                py.append(particle.py)
                pz.append(particle.pz)
                E.append(particle.E)
                MASS.append(particle.mass)

    px = np.array(px)/c # GeV/c
    py = np.array(py)/c
    pz = np.array(pz)/c

    return px, py, pz, pdg, E, MASS


#########################################################################################
# Recovering the data from MG (LHE) (PDG ID, px,py,pz,E,mass).
#########################################################################################

def recover_MG_DH1(px, py, pz, E, MASS, pdg):

    MG_pdg_DH1_1 = []
    for i in range(5,len(pdg),9):
        MG_pdg_DH1_1.append(pdg[i]) #List with the PDG ID of the particle produced by the decay of the LLP1

    MG_E_DH1 = []
    for i in range(0,len(px),2):
        MG_E_DH1.append(E[i]) #List with the energy of the LLP1

    MG_px_DH1 = []
    for i in range(0,len(px),2):
        MG_px_DH1.append(px[i]) #List with x momenta from LLP1

    MG_py_DH1 = []
    for i in range(0,len(px),2):
        MG_py_DH1.append(py[i]) #List with y momenta from LLP1

    MG_pz_DH1 = []
    for i in range(0,len(px),2):
        MG_pz_DH1.append(pz[i]) #List with z momenta from LLP1

    MG_mass_DH1 = []
    for i in range(0,len(px),2):
        MG_mass_DH1.append(MASS[i]) #List with the mass from LLP1

    MG_px_DH1 = np.array(MG_px_DH1) # convertion into arrays
    MG_py_DH1 = np.array(MG_py_DH1)
    MG_pz_DH1 = np.array(MG_pz_DH1)
    MG_E_DH1 = np.array(MG_E_DH1)
    MG_mass_DH1 = np.array(MG_mass_DH1)

    return MG_px_DH1, MG_py_DH1,MG_pz_DH1,MG_E_DH1,MG_mass_DH1,MG_pdg_DH1_1

#########################################################################################
# Computation of the kinematics variable for LLP1 (velocities, beta, gamma, pT the transverse momenta, eta the pseudo-rapidity).
#########################################################################################

def kinematics_MG_DH1(MG_px_DH1,MG_py_DH1,MG_pz_DH1,MG_E_DH1 ):

    MG_vx_DH1 = (MG_px_DH1*c**2)/MG_E_DH1 #compute the velocities in each direction
    MG_vy_DH1 = (MG_py_DH1*c**2)/MG_E_DH1
    MG_vz_DH1 = (MG_pz_DH1*c**2)/MG_E_DH1
    MG_beta_DH1 = np.sqrt(MG_vx_DH1**2 + MG_vy_DH1**2 + MG_vz_DH1**2)/c # compute beta
    MG_gamma_DH1 = 1/(np.sqrt(1-MG_beta_DH1**2)) # compute gamma
    MG_pT_DH1 = np.sqrt(MG_px_DH1**2 + MG_py_DH1**2)*c # compute the transverse momenta
    MG_eta_DH1 = np.arctanh(MG_pz_DH1/(np.sqrt(MG_px_DH1**2 + MG_py_DH1**2 + MG_pz_DH1**2))) # compute the pseudorapidity

    return MG_pT_DH1,MG_eta_DH1, MG_gamma_DH1

#########################################################################################
# Recovering the data from LLP2 (PDG ID, px,py,pz,E,mass).
#########################################################################################

def recover_MG_DH2(px, py, pz, E, MASS, pdg):

    MG_pdg_DH2_1 = []
    for i in range(7,len(pdg),9):
        MG_pdg_DH2_1.append(pdg[i]) #List with the PDG ID of the particle produced by the decay of the LLP2

    MG_E_DH2 = []
    for i in range(1,len(px),2):
        MG_E_DH2.append(E[i]) #List with the energy of the LLP2

    MG_px_DH2 = []
    for i in range(1,len(px),2):
        MG_px_DH2.append(px[i]) #List with x momenta from LLP2

    MG_py_DH2 = []
    for i in range(1,len(px),2):
        MG_py_DH2.append(py[i]) #List with y momenta from LLP2

    MG_pz_DH2 = []
    for i in range(1,len(px),2):
        MG_pz_DH2.append(pz[i]) #List with z momenta from LLP2

    MG_mass_DH2 = []
    for i in range(1,len(px),2):
        MG_mass_DH2.append(MASS[i]) #List with the mass from LLP2

    MG_px_DH2 = np.array(MG_px_DH2) # convertion into arrays
    MG_py_DH2 = np.array(MG_py_DH2)
    MG_pz_DH2 = np.array(MG_pz_DH2)
    MG_E_DH2 = np.array(MG_E_DH2)
    MG_mass_DH2 = np.array(MG_mass_DH2)

    return MG_px_DH2, MG_py_DH2,MG_pz_DH2,MG_E_DH2,MG_mass_DH2,MG_pdg_DH2_1

#########################################################################################
# Computation of the kinematics variable for LLP2 (velocities, beta, gamma, pT the transverse momenta, eta the pseudo-rapidity).
#########################################################################################

def kinemamtics_MG_DH2(MG_px_DH2,MG_py_DH2,MG_pz_DH2,MG_E_DH2):

    MG_vx_DH2 = (MG_px_DH2*c**2)/MG_E_DH2 #compute the velocities in each direction
    MG_vy_DH2 = (MG_py_DH2*c**2)/MG_E_DH2
    MG_vz_DH2 = (MG_pz_DH2*c**2)/MG_E_DH2
    MG_beta_DH2 = np.sqrt(MG_vx_DH2**2 + MG_vy_DH2**2 + MG_vz_DH2**2)/c # compute beta
    MG_gamma_DH2 = 1/(np.sqrt(1-MG_beta_DH2**2)) # compute gamma

    MG_pT_DH2 = np.sqrt(MG_px_DH2**2 + MG_py_DH2**2)*c # compute the transverse momenta
    MG_eta_DH2 = np.arctanh(MG_pz_DH2/(np.sqrt(MG_px_DH2**2 + MG_py_DH2**2 + MG_pz_DH2**2))) # compute the pseudorapidity

    return MG_pT_DH2,MG_eta_DH2, MG_gamma_DH2

#########################################################################################
# Decay lenght computation for LLP1.
#########################################################################################

def decaylenght_MG_DH1(MG_px_DH1, MG_py_DH1, MG_pz_DH1, E_DH1, MG_gamma_DH1, tauN):

    MG_Lx_tot_DH1 = []
    MG_Ly_tot_DH1 = []
    MG_Lz_tot_DH1 = []
    MG_Lxy_tot_DH1 = []

    for ctau in range(len(tauN)):

        MG_Lx_DH1 = []
        MG_Ly_DH1 = []
        MG_Lz_DH1 = []
        MG_Lxy_DH1 = []

        for i in range(len(MG_gamma_DH1)):
            MG_lt = lifetime(tauN[ctau]) # set the mean lifetime
            MG_Lx_DH1.append((MG_px_DH1[i]/E_DH1[i])*c**2 * MG_lt * MG_gamma_DH1[i]) # compute the decay lenght in x,y,z
            MG_Ly_DH1.append((MG_py_DH1[i]/E_DH1[i])*c**2 * MG_lt * MG_gamma_DH1[i])
            MG_Lz_DH1.append((abs(MG_pz_DH1[i])/E_DH1[i])*c**2 * MG_lt  * MG_gamma_DH1[i] )
            MG_Lxy_DH1.append(np.sqrt((MG_Lx_DH1[i])**2 + (MG_Ly_DH1[i])**2)) # compute the transverse decay lenght

        MG_Lx_tot_DH1.append(MG_Lx_DH1) # convertion into arrays
        MG_Ly_tot_DH1.append(MG_Ly_DH1)
        MG_Lz_tot_DH1.append(MG_Lz_DH1)
        MG_Lxy_tot_DH1.append(MG_Lxy_DH1)

    return MG_Lxy_tot_DH1, MG_Lz_tot_DH1

#########################################################################################
# Decay lenght computation for LLP2.
#########################################################################################

def decaylenght_MG_DH2(MG_px_DH2, MG_py_DH2, MG_pz_DH2, E_DH2, MG_gamma_DH2, tauN):

    MG_Lx_tot_DH2 = []
    MG_Ly_tot_DH2 = []
    MG_Lz_tot_DH2 = []
    MG_Lxy_tot_DH2 = []

    for ctau in range(len(tauN)):

        MG_Lx_DH2 = []
        MG_Ly_DH2 = []
        MG_Lz_DH2 = []
        MG_Lxy_DH2 = []

        for i in range(len(MG_gamma_DH2)):
            MG_lt = lifetime(tauN[ctau]) # set the mean lifetime
            MG_Lx_DH2.append((MG_px_DH2[i]/E_DH2[i])*c**2 * MG_lt * MG_gamma_DH2[i]) # compute the decay lenght in x,y,z
            MG_Ly_DH2.append((MG_py_DH2[i]/E_DH2[i])*c**2 * MG_lt * MG_gamma_DH2[i])
            MG_Lz_DH2.append((abs(MG_pz_DH2[i])/E_DH2[i])*c**2 * MG_lt  * MG_gamma_DH2[i] )
            MG_Lxy_DH2.append(np.sqrt((MG_Lx_DH2[i])**2 + (MG_Ly_DH2[i])**2)) # compute the transverse decay lenght

        MG_Lx_tot_DH2.append(MG_Lx_DH2)
        MG_Ly_tot_DH2.append(MG_Ly_DH2)
        MG_Lz_tot_DH2.append(MG_Lz_DH2)
        MG_Lxy_tot_DH2.append(MG_Lxy_DH2)

    return MG_Lxy_tot_DH2, MG_Lz_tot_DH2

#########################################################################################
# Computation of the efficiency with the map from the data obtained with MG for the high-ET samples (mH <= 400GeV).
#########################################################################################

def eff_map_MG_high(MG_pT_DH1, MG_eta_DH1,MG_Lxy_tot_DH1, MG_Lz_tot_DH1, MG_pdg_DH1_1, MG_pT_DH2, MG_eta_DH2, MG_Lxy_tot_DH2, MG_Lz_tot_DH2, MG_pdg_DH2_1, tauN, nevent, mass_phi, mass_s):

    MG_eff_highETX = []

    for index in tqdm.tqdm(range(len(tauN))):
        MG_queryMapResult = []
        for iEvent in range(len(MG_pT_DH1)):
            MG_queryMapResult.append(rmN.queryMapFromKinematics(MG_pT_DH1[iEvent],
                                                            MG_eta_DH1[iEvent],
                                                            MG_Lxy_tot_DH1[index][iEvent],
                                                            MG_Lz_tot_DH1[index][iEvent],
                                                            abs(MG_pdg_DH1_1[iEvent]),
                                                            MG_pT_DH2[iEvent],
                                                            MG_eta_DH2[iEvent],
                                                            MG_Lxy_tot_DH2[index][iEvent],
                                                            MG_Lz_tot_DH2[index][iEvent],
                                                            abs(MG_pdg_DH2_1[iEvent]),
                                                            selection = "high-ET"))
        MG_eff_highETX.append(sum(MG_queryMapResult))
    MG_queryMapResult = np.array(MG_queryMapResult) # convertion into arrays
    MG_eff_highETX = np.array(MG_eff_highETX) # convertion into arrays
    MG_eff_highETX = MG_eff_highETX/nevent #eff/nbrevent

    MG_Data_Eff_High = np.column_stack(MG_eff_highETX)
    np.savetxt(f'./Plots_High/Efficiencies_Text_{mass_phi}_{mass_s}.txt', MG_Data_Eff_High)

    return MG_eff_highETX

#########################################################################################
# Computation of the efficiency with the map from the data obtained with MG for the low-ET samples (mH <= 400GeV).
#########################################################################################

def eff_map_MG_low(MG_pT_DH1, MG_eta_DH1,MG_Lxy_tot_DH1, MG_Lz_tot_DH1, MG_pdg_DH1_1, MG_pT_DH2, MG_eta_DH2, MG_Lxy_tot_DH2, MG_Lz_tot_DH2, MG_pdg_DH2_1, tauN, nevent, mass_phi, mass_s):

    MG_eff_lowETX = []

    for index in tqdm.tqdm(range(len(tauN))):
        MG_queryMapResult = []
        for iEvent in range(len(MG_pT_DH1)):
            MG_queryMapResult.append(rmN.queryMapFromKinematics(MG_pT_DH1[iEvent],
                                                            MG_eta_DH1[iEvent],
                                                            MG_Lxy_tot_DH1[index][iEvent],
                                                            MG_Lz_tot_DH1[index][iEvent],
                                                            abs(MG_pdg_DH1_1[iEvent]),
                                                            MG_pT_DH2[iEvent],
                                                            MG_eta_DH2[iEvent],
                                                            MG_Lxy_tot_DH2[index][iEvent],
                                                            MG_Lz_tot_DH2[index][iEvent],
                                                            abs(MG_pdg_DH2_1[iEvent]),
                                                            selection = "low-ET"))
        MG_eff_lowETX.append(sum(MG_queryMapResult))
    MG_queryMapResult = np.array(MG_queryMapResult) # convertion into arrays
    MG_eff_lowETX = np.array(MG_eff_lowETX) # convertion into arrays
    MG_eff_lowETX = MG_eff_lowETX/nevent #eff/nbrevent

    MG_Data_Eff_Low = np.column_stack(MG_eff_lowETX)
    np.savetxt(f'./Plots_Low/Efficiencies_Text_{mass_phi}_{mass_s}.txt', MG_Data_Eff_Low)

    return MG_eff_lowETX


#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

#########################################################################################
# Computing HEP data
#########################################################################################

def elem_list(HEP, File_HEP_limit) :

    file_HEP = uproot.open(HEP) # open the file from HEP data for the efficiency
    data_HEP = file_HEP[file_HEP.keys()[1]] # open the branch

    file_HEP_limit = uproot.open(File_HEP_limit) # open the file from HEP data for the limits
    branch_HEP_limit = file_HEP_limit[file_HEP_limit.keys()[2]] # open the branch

    return data_HEP, branch_HEP_limit

#########################################################################################
# Plots to compare the results of efficiency obtained with MG, MG+Pythia8 (High-ET).
#########################################################################################

def plt_eff(MG_eff_highETX, eff_highETX,tauN, data_HEP,  mass_phi , mass_s, model="HSS"):
    
    MG_eff_highETX, eff_highETX,tauN = np.array(MG_eff_highETX), np.array(eff_highETX), np.array(tauN)
    # only plot within valid range
    mask =  eff_highETX > 0
    MG_eff_highETX=MG_eff_highETX[mask]
    eff_highETX = eff_highETX[mask]
    tauN = tauN[mask]

    ################## PLOT EFFICIENCY ##################
    fig, ax = plt.subplots()

    ################## Plot efficiency from MG ##################
    plt.plot(tauN,MG_eff_highETX, 'k--', linewidth=2, label = 'Map')

    ################## Plot efficiency from MG+Pythia8 ##################
    plt.plot(tauN,eff_highETX, 'r', linewidth=2, label = 'BDT ')
    
    if data_HEP is not None:
       ################## Plot efficiency from HEP data ##################
       plt.plot(data_HEP.values(axis='both')[0],data_HEP.values(axis='both')[1], 'b')

       ################ Uncertainties from HEP ##################
       plt.fill_between(data_HEP.values(axis='both')[0], data_HEP.values(axis='both')[1] +  data_HEP.errors('high')[1] , data_HEP.values(axis='both')[1] - data_HEP.errors('high')[1] , color = 'blue', label = r'ATLAS, with $\pm$ 1 $\sigma$ error bands',alpha=.7)
 
    ################## Uncertainties from Map ##################
    plt.fill_between(tauN, np.array(eff_highETX) + 0.25* np.array(eff_highETX), np.array(eff_highETX) - 0.25 * np.array(eff_highETX), label='25\% error bands ', color='r', alpha=.7)

    ################## Limits of validity ##################
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    if model=="HSS": ax.text(0.05, 0.95, f" $ m_Φ $ = {mass_phi} GeV, $m_S$ = {mass_s} GeV" , transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    if model=="ALP": ax.text(0.05, 0.95, f"  $m_a$ = {mass_s} GeV" , transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    #x = np.linspace(0.5,10)
    #ax.fill_between(x, max(eff_highETX), color='orange', alpha=.2, hatch="/", edgecolor="black", linewidth=1.0, label = 'validity window') # adding hatch

    plt.xscale('log')
    plt.ylim([0, max(eff_highETX)*2]) # start at 0
    #plt.ylim([1e-4,1]) # start at 0
    #plt.yscale('log')
    plt.xlabel(r'c$\tau$ [m]', fontsize=20)
    plt.ylabel('Efficiency', fontsize=20 )
    plt.legend(fontsize = 11, locolor=1) # set the legend in the upper right corner
    slug = f"{model}_mH{mass_phi}_mS{mass_s}.png"
    if model=="ALP": slug = f"{model}_mALP{mass_s}"
    plt.savefig(f"./Plots/Efficiency_comparison_{slug}.png")
    plt.savefig(f"./Plots/Efficiency_comparison_{slug}.pdf")
    print(f"./Plots/Efficiency_comparison_{slug}.png")
    plt.close()
    np.save(f"./Plots/values_{slug}_ctau.npy", tauN)
    np.save(f"./Plots/values_{slug}_eff.npy", eff_highETX)

def plt_multi_eff(eff_ctau_pairs_dict, ref=None,  model="HSS"):
    

    ################## PLOT EFFICIENCY ##################
    fig, ax = plt.subplots()
    
    maxVal = -1 
    
    plt.plot(-1, -1, 'k-', label ="$\mathbfit{ATLAS}$ full analysis")
    plt.plot(-1, -1, 'k--', label ="MG5 + Py8 + BDT")
    
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    cIndex = -1
    for name, values in eff_ctau_pairs_dict.items():
      cIndex +=1
      for refName in ref.keys():
        if name in refName:
          plt.plot(ref[refName][1],ref[refName][0], '-', color=colors[cIndex], linewidth=2)

      eff, ctau = values
      if maxVal < max(eff): maxVal=max(eff)
      plt.plot(ctau,eff, '--', linewidth=2, color=colors[cIndex], label = name)
      plt.fill_between(ctau, np.array(eff) * 1.25, np.array(eff) * 0.75, color=colors[cIndex], alpha=.7)
    

    ax.text(0.95, 0.95, "$\mathbfit{ATLAS}$ $\mathit{Simulation}$ $\mathit{Internal}$" , transform=ax.transAxes, fontsize=17, verticalalignment='top', horizontalalignment='right')
    ax.text(0.95, 0.88, f"{model} model" , transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right')

    plt.xscale('log')
    plt.ylim([0, maxVal*1.5]) # start at 0
    plt.xlabel(r'c$\tau$ [m]', fontsize=20)
    plt.ylabel('Efficiency', fontsize=20 )
    plt.legend(fontsize = 14, loc=0) # set the legend in the upper right corner
    plt.savefig(f"./Plots/Efficiency_summary_{model}.png")
    plt.savefig(f"./Plots/Efficiency_summary_{model}.pdf")
    print(f"./Plots/Efficiency_summary_{model}.png")


#########################################################################################
# Plot limits obtained with the map, to compare with those obtain by ATLAS (High-ET).
#########################################################################################

def plt_cross(eff_highETX, tauN, mass_phi, mass_s, branch_HEP_limit, factor, hepdata_eff=None):

    fig, ax = plt.subplots()

    Nsobs = 0.5630 * 26 * factor # nbr of observed events = 26 ( factor )

    Crr_Sec_obs = (Nsobs)/((np.array(eff_highETX)) * 139e3 ) # Luminosity = 139e3 fb**(-1)

    plt.plot(tauN, Crr_Sec_obs, 'r', label ='Map results', linewidth = 2)
    plt.fill_between(tauN,  1.25* np.array(Crr_Sec_obs), 0.75 * np.array(Crr_Sec_obs), label='25\% error bands ', color='r', alpha=.7)
    if hepdata_eff is not None: 
       Crr_Sec_obs_hepdata_eff = (Nsobs)/((np.array(hepdata_eff.values(axis='both')[1])) * 139e3 ) # Luminosity = 139e3 fb**(-1)
       plt.plot(hepdata_eff.values(axis='both')[0], Crr_Sec_obs_hepdata_eff, 'g', label ='Observed', linewidth = 2)
    if branch_HEP_limit is not None:
      plt.plot(np.array(branch_HEP_limit.values(axis='both')[0]), np.array(branch_HEP_limit.values(axis='both')[1]), 'b', label ='Observed', linewidth = 2)

    x = np.linspace(0.5,10)
    ax.fill_between(x, max(Crr_Sec_obs)*1.1, color='orange', alpha=.2, hatch="/", edgecolor="black", linewidth=1.0, label = 'validity window') # adding hatch
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-3, 1e3])
    plt.xlabel(r'c$\tau$ [m]')
    plt.ylabel(r'95% CL limit on $\sigma \times B$ [pb]')

    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.05, 0.95, f" $ m_Φ $ = {mass_phi} GeV, $m_S$ = {mass_s} GeV" , transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.legend( fontsize = 10, loc=3)
    plt.savefig(f"./Plots/Cross_section_mH{mass_phi}_mS{mass_s}.png") #create a new fodlder ' Plots ' and save the fig in it
    plt.savefig(f"./Plots/Cross_section_mH{mass_phi}_mS{mass_s}.pdf") #create a new fodlder ' Plots ' and save the fig in it
    print(f"./Plots/Cross_section_mH{mass_phi}_mS{mass_s}.png") #create a new fodlder ' Plots ' and save the fig in it
    plt.close()

