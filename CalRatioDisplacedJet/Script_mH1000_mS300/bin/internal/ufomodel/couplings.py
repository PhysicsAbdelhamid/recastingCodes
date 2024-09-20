# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 12.0.0 for Linux x86 (64-bit) (April 7, 2019)
# Date: Mon 29 Apr 2024 14:51:34


from object_library import all_couplings, Coupling

from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot



GC_1 = Coupling(name = 'GC_1',
                value = '-(AH*ch*complex(0,1))',
                order = {'HIW':1})

GC_2 = Coupling(name = 'GC_2',
                value = 'complex(0,1)*G',
                order = {'QCD':1})

GC_3 = Coupling(name = 'GC_3',
                value = '-(ch*complex(0,1)*GH)',
                order = {'HIG':1})

GC_4 = Coupling(name = 'GC_4',
                value = '-(ch*G*GH)',
                order = {'HIG':1,'QCD':1})

GC_5 = Coupling(name = 'GC_5',
                value = 'ch*complex(0,1)*G**2*GH',
                order = {'HIG':1,'QCD':2})

GC_6 = Coupling(name = 'GC_6',
                value = '-(AH*complex(0,1)*sh)',
                order = {'HIW':1})

GC_7 = Coupling(name = 'GC_7',
                value = '-(complex(0,1)*GH*sh)',
                order = {'HIG':1})

GC_8 = Coupling(name = 'GC_8',
                value = '-(G*GH*sh)',
                order = {'HIG':1,'QCD':1})

GC_9 = Coupling(name = 'GC_9',
                value = 'complex(0,1)*G**2*GH*sh',
                order = {'HIG':1,'QCD':2})

GC_10 = Coupling(name = 'GC_10',
                 value = '-3*ch**3*complex(0,1)*kap*sh + 6*ch**3*complex(0,1)*rho*sh + 3*ch*complex(0,1)*kap*sh**3 - 6*ch*complex(0,1)*lam*sh**3',
                 order = {'QED':2})

GC_11 = Coupling(name = 'GC_11',
                 value = '3*ch**3*complex(0,1)*kap*sh - 6*ch**3*complex(0,1)*lam*sh - 3*ch*complex(0,1)*kap*sh**3 + 6*ch*complex(0,1)*rho*sh**3',
                 order = {'QED':2})

GC_12 = Coupling(name = 'GC_12',
                 value = '-(ch**4*complex(0,1)*kap) + 4*ch**2*complex(0,1)*kap*sh**2 - 6*ch**2*complex(0,1)*lam*sh**2 - 6*ch**2*complex(0,1)*rho*sh**2 - complex(0,1)*kap*sh**4',
                 order = {'QED':2})

GC_13 = Coupling(name = 'GC_13',
                 value = '-6*ch**4*complex(0,1)*rho - 6*ch**2*complex(0,1)*kap*sh**2 - 6*complex(0,1)*lam*sh**4',
                 order = {'QED':2})

GC_14 = Coupling(name = 'GC_14',
                 value = '-6*ch**4*complex(0,1)*lam - 6*ch**2*complex(0,1)*kap*sh**2 - 6*complex(0,1)*rho*sh**4',
                 order = {'QED':2})

GC_15 = Coupling(name = 'GC_15',
                 value = '-3*ch**2*complex(0,1)*kap*sh*v - 6*complex(0,1)*lam*sh**3*v - 6*ch**3*complex(0,1)*rho*xi - 3*ch*complex(0,1)*kap*sh**2*xi',
                 order = {'QED':1})

GC_16 = Coupling(name = 'GC_16',
                 value = '2*ch**2*complex(0,1)*kap*sh*v - 6*ch**2*complex(0,1)*lam*sh*v - complex(0,1)*kap*sh**3*v - ch**3*complex(0,1)*kap*xi + 2*ch*complex(0,1)*kap*sh**2*xi - 6*ch*complex(0,1)*rho*sh**2*xi',
                 order = {'QED':1})

GC_17 = Coupling(name = 'GC_17',
                 value = '-(ch**3*complex(0,1)*kap*v) + 2*ch*complex(0,1)*kap*sh**2*v - 6*ch*complex(0,1)*lam*sh**2*v - 2*ch**2*complex(0,1)*kap*sh*xi + 6*ch**2*complex(0,1)*rho*sh*xi + complex(0,1)*kap*sh**3*xi',
                 order = {'QED':1})

GC_18 = Coupling(name = 'GC_18',
                 value = '-6*ch**3*complex(0,1)*lam*v - 3*ch*complex(0,1)*kap*sh**2*v + 3*ch**2*complex(0,1)*kap*sh*xi + 6*complex(0,1)*rho*sh**3*xi',
                 order = {'QED':1})

GC_19 = Coupling(name = 'GC_19',
                 value = '-((ch*complex(0,1)*yb)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_20 = Coupling(name = 'GC_20',
                 value = '-((complex(0,1)*sh*yb)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_21 = Coupling(name = 'GC_21',
                 value = '-((ch*complex(0,1)*yc)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_22 = Coupling(name = 'GC_22',
                 value = '-((complex(0,1)*sh*yc)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_23 = Coupling(name = 'GC_23',
                 value = '-((ch*complex(0,1)*yd)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_24 = Coupling(name = 'GC_24',
                 value = '-((complex(0,1)*sh*yd)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_25 = Coupling(name = 'GC_25',
                 value = '-((ch*complex(0,1)*ye)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_26 = Coupling(name = 'GC_26',
                 value = '-((complex(0,1)*sh*ye)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_27 = Coupling(name = 'GC_27',
                 value = '-((ch*complex(0,1)*ym)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_28 = Coupling(name = 'GC_28',
                 value = '-((complex(0,1)*sh*ym)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_29 = Coupling(name = 'GC_29',
                 value = '-((ch*complex(0,1)*ys)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_30 = Coupling(name = 'GC_30',
                 value = '-((complex(0,1)*sh*ys)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_31 = Coupling(name = 'GC_31',
                 value = '-((ch*complex(0,1)*yt)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_32 = Coupling(name = 'GC_32',
                 value = '-((complex(0,1)*sh*yt)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_33 = Coupling(name = 'GC_33',
                 value = '-((ch*complex(0,1)*ytau)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_34 = Coupling(name = 'GC_34',
                 value = '-((complex(0,1)*sh*ytau)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_35 = Coupling(name = 'GC_35',
                 value = '-((ch*complex(0,1)*yu)/cmath.sqrt(2))',
                 order = {'YUK':1})

GC_36 = Coupling(name = 'GC_36',
                 value = '-((complex(0,1)*sh*yu)/cmath.sqrt(2))',
                 order = {'YUK':1})

