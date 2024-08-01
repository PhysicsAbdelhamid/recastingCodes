import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import scipy
import time
import readMapNew as rmN
import tqdm
import mplhep as hep
import hepmc_parser as hepmc
import lhe_parser as lhe
import uproot
import random
import os
import glob
import pandas as pd
random.seed(123)


# set cst
c = 3e8 # Light velocity in m/s

def parsing_hepmc_generic(events, mediator=25, LLP2=36, LLP1=35, verbose=0):

    variables = ['px', 'py', 'pz', 'E', 'mass', 'decay']
    result={
    'Phi':pd.DataFrame(columns=variables),
    'LLP2':pd.DataFrame(columns=variables),
    'llp1':pd.DataFrame(columns=variables),
    'llp1_extra':pd.DataFrame(columns=variables),
    }
    print (f"using mediator={mediator}, LLP1={LLP1}, V={LLP2}")
    for ie , event in enumerate(events):
        if verbose: print("============NEW EVENT================")
        llpProdCounter=1
        llpDecayCounter=1
        LLP2Prod, LLP2Decay, mediatorProd, mediatorDecay, LLP1Prod, LLP1Decay, LLP1extraProd, LLP1extraDecay =  None, None, None, None, None, None, None, None
        for id, vertex in event.vertex.items():
            particlesOut = sorted([p for p in vertex.outcoming], key = lambda k : abs(k.pdg))
            particlesIn = sorted([p for p in vertex.incoming], key = lambda k : abs(k.pdg))
            pdgOut = [abs(p.pdg) for p in particlesOut]
            pdgIn = [abs(p.pdg) for p in particlesIn]
            pxIn = [p.px for p in particlesIn]
            pxOut = [p.px for p in particlesOut]
            # skip the cases where we have a particle both in the ingoing and outgoing particles
            # this is going to be simply a radiation of a gluon or photon...
            if LLP1 in pdgOut and LLP1 in pdgIn: continue
            if LLP2 in pdgOut and LLP2 in pdgIn: continue
            if mediator in pdgOut and mediator in pdgIn: continue
            # also skip vertices where none of the particles we are interested in are involved
            LLP1_inNeither = LLP1 not in pdgOut and LLP1 not in pdgIn
            V_inNeither = LLP2 not in pdgOut and LLP2 not in pdgIn
            med_inNeither = mediator not in pdgOut and mediator not in pdgIn
            if (LLP1_inNeither and V_inNeither and med_inNeither): continue
            if verbose: print (id, pdgIn, pxIn, "--->", pdgOut, pxOut)

            if LLP2 in pdgOut: # production of vector boson
               if verbose: print("--> it's the vector boson production vertex!")
               LLP2Prod  = particlesOut[pdgOut.index(LLP2)]
            if LLP2 in pdgIn: # decay of vector boson
               if verbose: print("--> it's the vector boson decay vertex!")
               LLP2Decay  = particlesOut[0] # always take the first of the two decay particles, should always get the lepton if l,v

            if  mediator in pdgOut: # production of mediator
               if verbose: print("--> it's the mediator production vertex!")
               mediatorProd  = particlesOut[pdgOut.index(mediator)]
            if  mediator in pdgIn: # decay of mediator
               if verbose: print("--> it's the mediator decay vertex!")
               mediatorDecay  = particlesOut[0]

            if  LLP1 in pdgOut: # production of LLP1
               if verbose: print("--> it's the LLP1 production vertex!")
               if pdgOut == [LLP1, LLP1] : # pair production of LLP1
                 LLP1Prod  = particlesOut[0]
                 LLP1extraProd  = particlesOut[1]
               else:
                 if llpProdCounter==1: LLP1Prod  = particlesOut[pdgOut.index(LLP1)]
                 if llpProdCounter==2: LLP1extraProd  = particlesOut[pdgOut.index(LLP1)]
                 llpProdCounter+=1

            if  LLP1 in pdgIn: # decay of LLP
               if verbose: print(f"--> it's the {llpDecayCounter}^th LLP1 decay vertex!",  particlesOut[0].pdg)
               if llpDecayCounter==1: LLP1Decay  = particlesOut[0]
               if llpDecayCounter==2: LLP1extraDecay  = particlesOut[0]
               llpDecayCounter+=1

        if LLP2Prod is not None and LLP2Decay is not None:
               result['LLP2'].loc[len(result['LLP2'])] = [LLP2Prod.px, LLP2Prod.py, LLP2Prod.pz, LLP2Prod.E, LLP2Prod.mass, LLP2Decay.pdg]
        if mediatorProd is not None and mediatorDecay is not None :
               result['Phi'].loc[len(result['Phi'])] = [mediatorProd.px, mediatorProd.py, mediatorProd.pz, mediatorProd.E, mediatorProd.mass, mediatorDecay.pdg]

        if LLP1Prod is not None and LLP1Decay is not None:
               result['llp1'].loc[len(result['llp1'])] = [LLP1Prod.px, LLP1Prod.py, LLP1Prod.pz, LLP1Prod.E, LLP1Prod.mass, LLP1Decay.pdg]
        
        if LLP1extraProd is not None and LLP1extraDecay is not None:
               result['llp1_extra'].loc[len(result['llp1_extra'])] = [LLP1extraProd.px, LLP1extraProd.py, LLP1extraProd.pz, LLP1extraProd.E, LLP1extraProd.mass, LLP1extraDecay.pdg]

    if len(result['LLP2'])==0:
        result['LLP2']=result['llp1_extra']
    return result

def kinematics(df):
    px = df.px
    py = df.py
    pz = df.pz
    E = df.E
    vx = (px*c)/E #compute the velocities in each direction
    vy = (py*c)/E
    vz = (pz*c)/E
    beta = np.sqrt(vx**2 + vy**2 + vz**2)/c # compute beta
    gamma = 1/(np.sqrt(1-beta**2)) # compute gamma

    pT = np.sqrt(px**2 + py**2) # compute the transverse momenta
    eta = np.arctanh(pz/(np.sqrt(px**2 + py**2 + pz**2))) # compute the pseudorapidity
    df['beta'] = beta
    df['gamma'] = gamma
    df['pT'] = pT
    df['eta'] = eta
    return df


def lifetime(avgtau = 4.3):
    import math
    avgtau = avgtau #/ c
    ct = random.random()
    print(avgtau, type(avgtau))
    return -1.0 * avgtau * math.log(ct)


def decayLength(df, tauN):

    px = df.px
    py = df.py
    pz = df.pz
    E = df.E
    gamma = df.gamma

    ctLx = []
    ctLy = []
    ctLz = []
    ctLxy = []

    for ctau in range(len(tauN)):

        Lx = []
        Ly = []
        Lz = []
        Lxy = []

        for i in range(len(gamma)):
            lt = lifetime(tauN[ctau]) # set mean lifetime
            Lx.append((px[i]/E[i]) * lt * gamma[i]) # compute the decay lenght in x,y,z
            Ly.append((py[i]/E[i]) * lt * gamma[i])
            Lz.append((abs(pz[i])/E[i]) * lt  * gamma[i] )
            Lxy.append(np.sqrt((Lx[i])**2 + (Ly[i])**2)) # compte the transverse decay lenght
        ctLx.append(Lx)
        ctLy.append(Ly)
        ctLz.append(Lz)
        ctLxy.append(Lxy)
    df['Lxy']=Lxy 
    df['Lz']=Lz 
    #print(' dataframe :  \n',df)
    print('longeur de df   : ' ,len(df))
    
    return [np.array(ctLxy), np.array(ctLz)]



def eff_map(values, selection):
    prob_results =[]

    for iEvent in range(len(values['llp1'])):
            prob_results.append(rmN.queryMapFromKinematics(values['llp1']['pT'][iEvent],
                                                        values['llp1']['eta'][iEvent],
                                                        values['llp1']['Lxy'][iEvent],
                                                        values['llp1']['Lz'][iEvent],
                                                        values['llp1']['decay'][iEvent],
                                                        values['LLP2']['pT'][iEvent],
                                                        values['LLP2']['eta'][iEvent],
                                                        values['LLP2']['Lxy'][iEvent],
                                                        values['LLP2']['Lz'][iEvent],
                                                        values['LLP2']['decay'][iEvent],
                                                        selection = selection))
    
    #print(values['llp1']['decay'][0])
    #print('prob_result : ',prob_results)
    #print('len(prob_result) :', len(prob_results))
    total_prob =  sum(prob_results)
    total_eff = total_prob / len(values['llp1'])
    print('total prob : ', total_prob)
    print('total eff :', total_eff)
    
    return total_eff



