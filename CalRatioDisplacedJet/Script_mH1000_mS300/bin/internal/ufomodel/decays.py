# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 12.0.0 for Linux x86 (64-bit) (April 7, 2019)
# Date: Mon 29 Apr 2024 14:51:34


from object_library import all_decays, Decay
import particles as P


Decay_h = Decay(name = 'Decay_h',
                particle = P.h,
                partial_widths = {(P.H,P.H):'((ch**6*kap**2*v**2 - 4*ch**4*kap**2*sh**2*v**2 + 12*ch**4*kap*lam*sh**2*v**2 + 4*ch**2*kap**2*sh**4*v**2 - 24*ch**2*kap*lam*sh**4*v**2 + 36*ch**2*lam**2*sh**4*v**2 + 4*ch**5*kap**2*sh*v*xi - 12*ch**5*kap*rho*sh*v*xi - 10*ch**3*kap**2*sh**3*v*xi + 24*ch**3*kap*lam*sh**3*v*xi + 24*ch**3*kap*rho*sh**3*v*xi - 72*ch**3*lam*rho*sh**3*v*xi + 4*ch*kap**2*sh**5*v*xi - 12*ch*kap*lam*sh**5*v*xi + 4*ch**4*kap**2*sh**2*xi**2 - 24*ch**4*kap*rho*sh**2*xi**2 + 36*ch**4*rho**2*sh**2*xi**2 - 4*ch**2*kap**2*sh**4*xi**2 + 12*ch**2*kap*rho*sh**4*xi**2 + kap**2*sh**6*xi**2)*cmath.sqrt(MHinput**4 - 4*MHinput**2*MHSinput**2))/(32.*cmath.pi*abs(MHinput)**3)',
                                  (P.A,P.A):'(AH**2*ch**2*MHinput**6)/(64.*cmath.pi*abs(MHinput)**3)',
                                  (P.G,P.G):'(ch**2*GH**2*MHinput**6)/(8.*cmath.pi*abs(MHinput)**3)',
                                  (P.d,P.d__tilde__):'((-12*ch**2*MD**2*yd**2 + 3*ch**2*MHinput**2*yd**2)*cmath.sqrt(-4*MD**2*MHinput**2 + MHinput**4))/(16.*cmath.pi*abs(MHinput)**3)',
                                  (P.s,P.s__tilde__):'((3*ch**2*MHinput**2*ys**2 - 12*ch**2*MS**2*ys**2)*cmath.sqrt(MHinput**4 - 4*MHinput**2*MS**2))/(16.*cmath.pi*abs(MHinput)**3)',
                                  (P.b,P.b__tilde__):'((-12*ch**2*MB**2*yb**2 + 3*ch**2*MHinput**2*yb**2)*cmath.sqrt(-4*MB**2*MHinput**2 + MHinput**4))/(16.*cmath.pi*abs(MHinput)**3)',
                                  (P.e__minus__,P.e__plus__):'((-4*ch**2*ME**2*ye**2 + ch**2*MHinput**2*ye**2)*cmath.sqrt(-4*ME**2*MHinput**2 + MHinput**4))/(16.*cmath.pi*abs(MHinput)**3)',
                                  (P.m__minus__,P.m__plus__):'((ch**2*MHinput**2*ym**2 - 4*ch**2*MM**2*ym**2)*cmath.sqrt(MHinput**4 - 4*MHinput**2*MM**2))/(16.*cmath.pi*abs(MHinput)**3)',
                                  (P.tt__minus__,P.tt__plus__):'((ch**2*MHinput**2*ytau**2 - 4*ch**2*MTA**2*ytau**2)*cmath.sqrt(MHinput**4 - 4*MHinput**2*MTA**2))/(16.*cmath.pi*abs(MHinput)**3)',
                                  (P.u,P.u__tilde__):'((3*ch**2*MHinput**2*yu**2 - 12*ch**2*MU**2*yu**2)*cmath.sqrt(MHinput**4 - 4*MHinput**2*MU**2))/(16.*cmath.pi*abs(MHinput)**3)',
                                  (P.c,P.c__tilde__):'((-12*ch**2*MC**2*yc**2 + 3*ch**2*MHinput**2*yc**2)*cmath.sqrt(-4*MC**2*MHinput**2 + MHinput**4))/(16.*cmath.pi*abs(MHinput)**3)',
                                  (P.t,P.t__tilde__):'((3*ch**2*MHinput**2*yt**2 - 12*ch**2*MT**2*yt**2)*cmath.sqrt(MHinput**4 - 4*MHinput**2*MT**2))/(16.*cmath.pi*abs(MHinput)**3)'})

Decay_H = Decay(name = 'Decay_H',
                particle = P.H,
                partial_widths = {(P.h,P.h):'((4*ch**4*kap**2*sh**2*v**2 - 24*ch**4*kap*lam*sh**2*v**2 + 36*ch**4*lam**2*sh**2*v**2 - 4*ch**2*kap**2*sh**4*v**2 + 12*ch**2*kap*lam*sh**4*v**2 + kap**2*sh**6*v**2 - 4*ch**5*kap**2*sh*v*xi + 12*ch**5*kap*lam*sh*v*xi + 10*ch**3*kap**2*sh**3*v*xi - 24*ch**3*kap*lam*sh**3*v*xi - 24*ch**3*kap*rho*sh**3*v*xi + 72*ch**3*lam*rho*sh**3*v*xi - 4*ch*kap**2*sh**5*v*xi + 12*ch*kap*rho*sh**5*v*xi + ch**6*kap**2*xi**2 - 4*ch**4*kap**2*sh**2*xi**2 + 12*ch**4*kap*rho*sh**2*xi**2 + 4*ch**2*kap**2*sh**4*xi**2 - 24*ch**2*kap*rho*sh**4*xi**2 + 36*ch**2*rho**2*sh**4*xi**2)*cmath.sqrt(-4*MHinput**2*MHSinput**2 + MHSinput**4))/(32.*cmath.pi*abs(MHSinput)**3)',
                                  (P.A,P.A):'(AH**2*MHSinput**6*sh**2)/(64.*cmath.pi*abs(MHSinput)**3)',
                                  (P.G,P.G):'(GH**2*MHSinput**6*sh**2)/(8.*cmath.pi*abs(MHSinput)**3)',
                                  (P.d,P.d__tilde__):'((-12*MD**2*sh**2*yd**2 + 3*MHSinput**2*sh**2*yd**2)*cmath.sqrt(-4*MD**2*MHSinput**2 + MHSinput**4))/(16.*cmath.pi*abs(MHSinput)**3)',
                                  (P.s,P.s__tilde__):'((3*MHSinput**2*sh**2*ys**2 - 12*MS**2*sh**2*ys**2)*cmath.sqrt(MHSinput**4 - 4*MHSinput**2*MS**2))/(16.*cmath.pi*abs(MHSinput)**3)',
                                  (P.b,P.b__tilde__):'((-12*MB**2*sh**2*yb**2 + 3*MHSinput**2*sh**2*yb**2)*cmath.sqrt(-4*MB**2*MHSinput**2 + MHSinput**4))/(16.*cmath.pi*abs(MHSinput)**3)',
                                  (P.e__minus__,P.e__plus__):'((-4*ME**2*sh**2*ye**2 + MHSinput**2*sh**2*ye**2)*cmath.sqrt(-4*ME**2*MHSinput**2 + MHSinput**4))/(16.*cmath.pi*abs(MHSinput)**3)',
                                  (P.m__minus__,P.m__plus__):'((MHSinput**2*sh**2*ym**2 - 4*MM**2*sh**2*ym**2)*cmath.sqrt(MHSinput**4 - 4*MHSinput**2*MM**2))/(16.*cmath.pi*abs(MHSinput)**3)',
                                  (P.tt__minus__,P.tt__plus__):'((MHSinput**2*sh**2*ytau**2 - 4*MTA**2*sh**2*ytau**2)*cmath.sqrt(MHSinput**4 - 4*MHSinput**2*MTA**2))/(16.*cmath.pi*abs(MHSinput)**3)',
                                  (P.u,P.u__tilde__):'((3*MHSinput**2*sh**2*yu**2 - 12*MU**2*sh**2*yu**2)*cmath.sqrt(MHSinput**4 - 4*MHSinput**2*MU**2))/(16.*cmath.pi*abs(MHSinput)**3)',
                                  (P.c,P.c__tilde__):'((-12*MC**2*sh**2*yc**2 + 3*MHSinput**2*sh**2*yc**2)*cmath.sqrt(-4*MC**2*MHSinput**2 + MHSinput**4))/(16.*cmath.pi*abs(MHSinput)**3)',
                                  (P.t,P.t__tilde__):'((3*MHSinput**2*sh**2*yt**2 - 12*MT**2*sh**2*yt**2)*cmath.sqrt(MHSinput**4 - 4*MHSinput**2*MT**2))/(16.*cmath.pi*abs(MHSinput)**3)'})

