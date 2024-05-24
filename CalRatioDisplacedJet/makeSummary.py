#!/usr/bin/env python3
import Computation_Functions as cf
import glob
import numpy as np

model="ALP"

filenames = glob.glob(f"Plots/values_*{model}*_ctau.npy")
plotDict = {}

for fn in filenames:
  name = fn.split(f"_{model}_")[1].split("_ctau")[0]
  print(fn, name)
  eff = np.load(fn.replace("ctau", "eff"))
  ctau = np.load(fn)
  plotDict[name]= [eff, ctau]

ref = {} 
filenames = glob.glob(f"ReferenceData/*{model}*txt")
for fn in filenames:
  print(fn)
  name = "m"+fn.split(f"_m")[-1].split(".txt")[0]
  ctau, eff = np.loadtxt(fn, delimiter=',').T
  ref[name]= [eff, ctau]

cf.plt_multi_eff(plotDict, ref,  model=model)
