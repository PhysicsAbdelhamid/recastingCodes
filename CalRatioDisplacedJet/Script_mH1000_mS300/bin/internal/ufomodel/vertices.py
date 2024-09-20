# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 12.0.0 for Linux x86 (64-bit) (April 7, 2019)
# Date: Mon 29 Apr 2024 14:51:34


from object_library import all_vertices, Vertex
import particles as P
import couplings as C
import lorentz as L


V_1 = Vertex(name = 'V_1',
             particles = [ P.h, P.h, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_12})

V_2 = Vertex(name = 'V_2',
             particles = [ P.h, P.h, P.h, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_11})

V_3 = Vertex(name = 'V_3',
             particles = [ P.h, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_10})

V_4 = Vertex(name = 'V_4',
             particles = [ P.h, P.h, P.h, P.h ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_14})

V_5 = Vertex(name = 'V_5',
             particles = [ P.H, P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSSS1 ],
             couplings = {(0,0):C.GC_13})

V_6 = Vertex(name = 'V_6',
             particles = [ P.h, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_17})

V_7 = Vertex(name = 'V_7',
             particles = [ P.h, P.h, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_16})

V_8 = Vertex(name = 'V_8',
             particles = [ P.H, P.H, P.H ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_15})

V_9 = Vertex(name = 'V_9',
             particles = [ P.h, P.h, P.h ],
             color = [ '1' ],
             lorentz = [ L.SSS1 ],
             couplings = {(0,0):C.GC_18})

V_10 = Vertex(name = 'V_10',
              particles = [ P.A, P.A, P.h ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_1})

V_11 = Vertex(name = 'V_11',
              particles = [ P.A, P.A, P.H ],
              color = [ '1' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_6})

V_12 = Vertex(name = 'V_12',
              particles = [ P.G, P.G, P.h ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_3})

V_13 = Vertex(name = 'V_13',
              particles = [ P.G, P.G, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.VVS1 ],
              couplings = {(0,0):C.GC_7})

V_14 = Vertex(name = 'V_14',
              particles = [ P.G, P.G, P.G, P.h ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVS1 ],
              couplings = {(0,0):C.GC_4})

V_15 = Vertex(name = 'V_15',
              particles = [ P.G, P.G, P.G, P.H ],
              color = [ 'f(1,2,3)' ],
              lorentz = [ L.VVVS1 ],
              couplings = {(0,0):C.GC_8})

V_16 = Vertex(name = 'V_16',
              particles = [ P.G, P.G, P.G, P.G, P.h ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVS1, L.VVVVS2, L.VVVVS3 ],
              couplings = {(1,1):C.GC_5,(0,0):C.GC_5,(2,2):C.GC_5})

V_17 = Vertex(name = 'V_17',
              particles = [ P.G, P.G, P.G, P.G, P.H ],
              color = [ 'f(-1,1,2)*f(3,4,-1)', 'f(-1,1,3)*f(2,4,-1)', 'f(-1,1,4)*f(2,3,-1)' ],
              lorentz = [ L.VVVVS1, L.VVVVS2, L.VVVVS3 ],
              couplings = {(1,1):C.GC_9,(0,0):C.GC_9,(2,2):C.GC_9})

V_18 = Vertex(name = 'V_18',
              particles = [ P.d__tilde__, P.d, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_19 = Vertex(name = 'V_19',
              particles = [ P.s__tilde__, P.s, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_20 = Vertex(name = 'V_20',
              particles = [ P.b__tilde__, P.b, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_21 = Vertex(name = 'V_21',
              particles = [ P.u__tilde__, P.u, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_22 = Vertex(name = 'V_22',
              particles = [ P.c__tilde__, P.c, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_23 = Vertex(name = 'V_23',
              particles = [ P.t__tilde__, P.t, P.G ],
              color = [ 'T(3,2,1)' ],
              lorentz = [ L.FFV1 ],
              couplings = {(0,0):C.GC_2})

V_24 = Vertex(name = 'V_24',
              particles = [ P.d__tilde__, P.d, P.h ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_23})

V_25 = Vertex(name = 'V_25',
              particles = [ P.s__tilde__, P.s, P.h ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_29})

V_26 = Vertex(name = 'V_26',
              particles = [ P.b__tilde__, P.b, P.h ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_19})

V_27 = Vertex(name = 'V_27',
              particles = [ P.d__tilde__, P.d, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_24})

V_28 = Vertex(name = 'V_28',
              particles = [ P.s__tilde__, P.s, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_30})

V_29 = Vertex(name = 'V_29',
              particles = [ P.b__tilde__, P.b, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_20})

V_30 = Vertex(name = 'V_30',
              particles = [ P.e__plus__, P.e__minus__, P.h ],
              color = [ '1' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_25})

V_31 = Vertex(name = 'V_31',
              particles = [ P.m__plus__, P.m__minus__, P.h ],
              color = [ '1' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_27})

V_32 = Vertex(name = 'V_32',
              particles = [ P.tt__plus__, P.tt__minus__, P.h ],
              color = [ '1' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_33})

V_33 = Vertex(name = 'V_33',
              particles = [ P.e__plus__, P.e__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_26})

V_34 = Vertex(name = 'V_34',
              particles = [ P.m__plus__, P.m__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_28})

V_35 = Vertex(name = 'V_35',
              particles = [ P.tt__plus__, P.tt__minus__, P.H ],
              color = [ '1' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_34})

V_36 = Vertex(name = 'V_36',
              particles = [ P.u__tilde__, P.u, P.h ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_35})

V_37 = Vertex(name = 'V_37',
              particles = [ P.c__tilde__, P.c, P.h ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_21})

V_38 = Vertex(name = 'V_38',
              particles = [ P.t__tilde__, P.t, P.h ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_31})

V_39 = Vertex(name = 'V_39',
              particles = [ P.u__tilde__, P.u, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_36})

V_40 = Vertex(name = 'V_40',
              particles = [ P.c__tilde__, P.c, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_22})

V_41 = Vertex(name = 'V_41',
              particles = [ P.t__tilde__, P.t, P.H ],
              color = [ 'Identity(1,2)' ],
              lorentz = [ L.FFS1 ],
              couplings = {(0,0):C.GC_32})

