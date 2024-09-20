# This file was automatically created by FeynRules 2.3.49
# Mathematica version: 12.0.0 for Linux x86 (64-bit) (April 7, 2019)
# Date: Mon 29 Apr 2024 14:51:34



from object_library import all_parameters, Parameter


from function_library import complexconjugate, re, im, csc, sec, acsc, asec, cot

# This is a default parameter object representing 0.
ZERO = Parameter(name = 'ZERO',
                 nature = 'internal',
                 type = 'real',
                 value = '0.0',
                 texname = '0')

# User-defined parameters.
kap = Parameter(name = 'kap',
                nature = 'external',
                type = 'real',
                value = 1.e-9,
                texname = '\\text{kap}',
                lhablock = 'HIDDEN',
                lhacode = [ 1 ])

aEWM1 = Parameter(name = 'aEWM1',
                  nature = 'external',
                  type = 'real',
                  value = 127.9,
                  texname = '\\text{aEWM1}',
                  lhablock = 'SMINPUTS',
                  lhacode = [ 1 ])

Gf = Parameter(name = 'Gf',
               nature = 'external',
               type = 'real',
               value = 0.000011663900000000002,
               texname = '\\text{Gf}',
               lhablock = 'SMINPUTS',
               lhacode = [ 2 ])

aS = Parameter(name = 'aS',
               nature = 'external',
               type = 'real',
               value = 0.118,
               texname = '\\text{aS}',
               lhablock = 'SMINPUTS',
               lhacode = [ 3 ])

ymd = Parameter(name = 'ymd',
                nature = 'external',
                type = 'real',
                value = 0.00467,
                texname = '\\text{ymd}',
                lhablock = 'YUKAWA',
                lhacode = [ 1 ])

ymu = Parameter(name = 'ymu',
                nature = 'external',
                type = 'real',
                value = 0.0026,
                texname = '\\text{ymu}',
                lhablock = 'YUKAWA',
                lhacode = [ 2 ])

yms = Parameter(name = 'yms',
                nature = 'external',
                type = 'real',
                value = 0.093,
                texname = '\\text{yms}',
                lhablock = 'YUKAWA',
                lhacode = [ 3 ])

ymc = Parameter(name = 'ymc',
                nature = 'external',
                type = 'real',
                value = 1.42,
                texname = '\\text{ymc}',
                lhablock = 'YUKAWA',
                lhacode = [ 4 ])

ymb = Parameter(name = 'ymb',
                nature = 'external',
                type = 'real',
                value = 4.7,
                texname = '\\text{ymb}',
                lhablock = 'YUKAWA',
                lhacode = [ 5 ])

ymt = Parameter(name = 'ymt',
                nature = 'external',
                type = 'real',
                value = 174.3,
                texname = '\\text{ymt}',
                lhablock = 'YUKAWA',
                lhacode = [ 6 ])

ymel = Parameter(name = 'ymel',
                 nature = 'external',
                 type = 'real',
                 value = 0.000511,
                 texname = '\\text{ymel}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 11 ])

ymmu = Parameter(name = 'ymmu',
                 nature = 'external',
                 type = 'real',
                 value = 0.1057,
                 texname = '\\text{ymmu}',
                 lhablock = 'YUKAWA',
                 lhacode = [ 13 ])

ymtau = Parameter(name = 'ymtau',
                  nature = 'external',
                  type = 'real',
                  value = 1.777,
                  texname = '\\text{ymtau}',
                  lhablock = 'YUKAWA',
                  lhacode = [ 15 ])

MW = Parameter(name = 'MW',
               nature = 'external',
               type = 'real',
               value = 80.377,
               texname = 'M_W',
               lhablock = 'FRBlock',
               lhacode = [ 1 ])

xi = Parameter(name = 'xi',
               nature = 'external',
               type = 'real',
               value = 1.e-7,
               texname = '\\xi',
               lhablock = 'FRBlock',
               lhacode = [ 2 ])

ME = Parameter(name = 'ME',
               nature = 'external',
               type = 'real',
               value = 0.000511,
               texname = '\\text{ME}',
               lhablock = 'MASS',
               lhacode = [ 11 ])

MM = Parameter(name = 'MM',
               nature = 'external',
               type = 'real',
               value = 0.1057,
               texname = '\\text{MM}',
               lhablock = 'MASS',
               lhacode = [ 13 ])

MTA = Parameter(name = 'MTA',
                nature = 'external',
                type = 'real',
                value = 1.777,
                texname = '\\text{MTA}',
                lhablock = 'MASS',
                lhacode = [ 15 ])

MU = Parameter(name = 'MU',
               nature = 'external',
               type = 'real',
               value = 0.0026,
               texname = 'M',
               lhablock = 'MASS',
               lhacode = [ 2 ])

MC = Parameter(name = 'MC',
               nature = 'external',
               type = 'real',
               value = 1.42,
               texname = '\\text{MC}',
               lhablock = 'MASS',
               lhacode = [ 4 ])

MT = Parameter(name = 'MT',
               nature = 'external',
               type = 'real',
               value = 174.3,
               texname = '\\text{MT}',
               lhablock = 'MASS',
               lhacode = [ 6 ])

MD = Parameter(name = 'MD',
               nature = 'external',
               type = 'real',
               value = 0.00467,
               texname = '\\text{MD}',
               lhablock = 'MASS',
               lhacode = [ 1 ])

MS = Parameter(name = 'MS',
               nature = 'external',
               type = 'real',
               value = 0.093,
               texname = '\\text{MS}',
               lhablock = 'MASS',
               lhacode = [ 3 ])

MB = Parameter(name = 'MB',
               nature = 'external',
               type = 'real',
               value = 4.7,
               texname = '\\text{MB}',
               lhablock = 'MASS',
               lhacode = [ 5 ])

MHinput = Parameter(name = 'MHinput',
                    nature = 'external',
                    type = 'real',
                    value = 125,
                    texname = '\\text{MHinput}',
                    lhablock = 'MASS',
                    lhacode = [ 25 ])

MHSinput = Parameter(name = 'MHSinput',
                     nature = 'external',
                     type = 'real',
                     value = 100,
                     texname = '\\text{MHSinput}',
                     lhablock = 'MASS',
                     lhacode = [ 35 ])

WT = Parameter(name = 'WT',
               nature = 'external',
               type = 'real',
               value = 1.50833649,
               texname = '\\text{WT}',
               lhablock = 'DECAY',
               lhacode = [ 6 ])

WH = Parameter(name = 'WH',
               nature = 'external',
               type = 'real',
               value = 0.00282299,
               texname = '\\text{WH}',
               lhablock = 'DECAY',
               lhacode = [ 25 ])

WHS = Parameter(name = 'WHS',
                nature = 'external',
                type = 'real',
                value = 5.23795,
                texname = '\\text{WHS}',
                lhablock = 'DECAY',
                lhacode = [ 35 ])

aEW = Parameter(name = 'aEW',
                nature = 'internal',
                type = 'real',
                value = '1/aEWM1',
                texname = '\\text{aEW}')

G = Parameter(name = 'G',
              nature = 'internal',
              type = 'real',
              value = '2*cmath.sqrt(aS)*cmath.sqrt(cmath.pi)',
              texname = 'G')

v = Parameter(name = 'v',
              nature = 'internal',
              type = 'real',
              value = '1/(2**0.25*cmath.sqrt(Gf))',
              texname = 'v')

th = Parameter(name = 'th',
               nature = 'internal',
               type = 'real',
               value = '(-MHinput**2 + MHSinput**2 + (2*cmath.atan(10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000*(MHinput - MHSinput))*cmath.sqrt((MHinput**2 - MHSinput**2)**2 - 4*kap**2*v**2*xi**2))/cmath.pi)/(2.*kap*v*xi)',
               texname = 't_h')

ee = Parameter(name = 'ee',
               nature = 'internal',
               type = 'real',
               value = '2*cmath.sqrt(aEW)*cmath.sqrt(cmath.pi)',
               texname = 'e')

GH = Parameter(name = 'GH',
               nature = 'internal',
               type = 'real',
               value = '-(G**2*(1 + (13*MHinput**6)/(16800.*MT**6) + MHinput**4/(168.*MT**4) + (7*MHinput**2)/(120.*MT**2)))/(12.*cmath.pi**2*v)',
               texname = 'G_H')

lam = Parameter(name = 'lam',
                nature = 'internal',
                type = 'real',
                value = '(MHinput**2 + MHSinput**2 + (2*cmath.atan(10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000*(MHinput - MHSinput))*cmath.sqrt((MHinput**2 - MHSinput**2)**2 - 4*kap**2*v**2*xi**2))/cmath.pi)/(4.*v**2)',
                texname = '\\text{lam}')

rho = Parameter(name = 'rho',
                nature = 'internal',
                type = 'real',
                value = '(MHinput**2 + MHSinput**2 - (2*cmath.atan(10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000*(MHinput - MHSinput))*cmath.sqrt((MHinput**2 - MHSinput**2)**2 - 4*kap**2*v**2*xi**2))/cmath.pi)/(4.*xi**2)',
                texname = '\\rho')

yb = Parameter(name = 'yb',
               nature = 'internal',
               type = 'real',
               value = '(ymb*cmath.sqrt(2))/v',
               texname = '\\text{yb}')

yc = Parameter(name = 'yc',
               nature = 'internal',
               type = 'real',
               value = '(ymc*cmath.sqrt(2))/v',
               texname = '\\text{yc}')

yd = Parameter(name = 'yd',
               nature = 'internal',
               type = 'real',
               value = '(ymd*cmath.sqrt(2))/v',
               texname = '\\text{yd}')

ye = Parameter(name = 'ye',
               nature = 'internal',
               type = 'real',
               value = '(ymel*cmath.sqrt(2))/v',
               texname = '\\text{ye}')

ym = Parameter(name = 'ym',
               nature = 'internal',
               type = 'real',
               value = '(ymmu*cmath.sqrt(2))/v',
               texname = '\\text{ym}')

ys = Parameter(name = 'ys',
               nature = 'internal',
               type = 'real',
               value = '(yms*cmath.sqrt(2))/v',
               texname = '\\text{ys}')

yt = Parameter(name = 'yt',
               nature = 'internal',
               type = 'real',
               value = '(ymt*cmath.sqrt(2))/v',
               texname = '\\text{yt}')

ytau = Parameter(name = 'ytau',
                 nature = 'internal',
                 type = 'real',
                 value = '(ymtau*cmath.sqrt(2))/v',
                 texname = '\\text{ytau}')

yu = Parameter(name = 'yu',
               nature = 'internal',
               type = 'real',
               value = '(ymu*cmath.sqrt(2))/v',
               texname = '\\text{yu}')

ch = Parameter(name = 'ch',
               nature = 'internal',
               type = 'real',
               value = '1/cmath.sqrt(1 + th**2)',
               texname = 'c_h')

muH2 = Parameter(name = 'muH2',
                 nature = 'internal',
                 type = 'real',
                 value = '(kap*v**2)/2. + rho*xi**2',
                 texname = '\\text{muH2}')

muSM2 = Parameter(name = 'muSM2',
                  nature = 'internal',
                  type = 'real',
                  value = 'lam*v**2 + (kap*xi**2)/2.',
                  texname = '\\text{muSM2}')

sh = Parameter(name = 'sh',
               nature = 'internal',
               type = 'real',
               value = 'th/cmath.sqrt(1 + th**2)',
               texname = 's_h')

AH = Parameter(name = 'AH',
               nature = 'internal',
               type = 'real',
               value = '(47*ee**2*(1 - (2*MHinput**4)/(987.*MT**4) - (14*MHinput**2)/(705.*MT**2) + (213*MHinput**12)/(2.634632e7*MW**12) + (5*MHinput**10)/(119756.*MW**10) + (41*MHinput**8)/(180950.*MW**8) + (87*MHinput**6)/(65800.*MW**6) + (57*MHinput**4)/(6580.*MW**4) + (33*MHinput**2)/(470.*MW**2)))/(72.*cmath.pi**2*v)',
               texname = 'A_H')

