ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

C     Parameters that should not be recomputed event by event.
C     
      IF(READLHA) THEN

        G = 2 * DSQRT(AS*PI)  ! for the first init

        MDL_NB__2__EXP__0_25 = 2.000000D+00**2.500000D-01

        MDL_SQRT__GF = SQRT(DCMPLX(MDL_GF))

        MDL_V = 1.000000D+00/(MDL_NB__2__EXP__0_25*MDL_SQRT__GF)

        MDL_MS1__EXP__2 = MDL_MS1**2

        MDL_V__EXP__2 = MDL_V**2

        MDL_MB__EXP__2 = MDL_MB**2

        MDL_MC__EXP__2 = MDL_MC**2

        MDL_MD__EXP__2 = MDL_MD**2

        MDL_MS__EXP__2 = MDL_MS**2

        MDL_MTA__EXP__2 = MDL_MTA**2

        MDL_MU__EXP__2 = MDL_MU**2

        MDL_Y1 = (PI*MDL_HBAR*MDL_MS1__EXP__2*MDL_V__EXP__2)
     $   /(MDL_MB__EXP__2*(-4.000000D+00*MDL_MB__EXP__2
     $   +MDL_MS1__EXP__2)**1.500000D+00+MDL_MC__EXP__2*(-4.000000D+00
     $   *MDL_MC__EXP__2+MDL_MS1__EXP__2)**1.500000D+00+MDL_MD__EXP__2
     $   *(-4.000000D+00*MDL_MD__EXP__2+MDL_MS1__EXP__2)**1.500000D+00
     $   +MDL_MS__EXP__2*(-4.000000D+00*MDL_MS__EXP__2+MDL_MS1__EXP__2)
     $   **1.500000D+00+MDL_MTA__EXP__2*(MDL_MS1__EXP__2-4.000000D+00
     $   *MDL_MTA__EXP__2)**1.500000D+00+(3.000000D+00*MDL_CT1
     $   *MDL_MU__EXP__2*(MDL_MS1__EXP__2-4.000000D+00*MDL_MU__EXP__2)
     $   **1.500000D+00)/MDL_C)

        MDL_MS2__EXP__2 = MDL_MS2**2

        MDL_Y2 = (PI*MDL_HBAR*MDL_MS2__EXP__2*MDL_V__EXP__2)
     $   /(MDL_MB__EXP__2*(-4.000000D+00*MDL_MB__EXP__2
     $   +MDL_MS2__EXP__2)**1.500000D+00+MDL_MC__EXP__2*(-4.000000D+00
     $   *MDL_MC__EXP__2+MDL_MS2__EXP__2)**1.500000D+00+MDL_MD__EXP__2
     $   *(-4.000000D+00*MDL_MD__EXP__2+MDL_MS2__EXP__2)**1.500000D+00
     $   +MDL_MS__EXP__2*(-4.000000D+00*MDL_MS__EXP__2+MDL_MS2__EXP__2)
     $   **1.500000D+00+MDL_MTA__EXP__2*(MDL_MS2__EXP__2-4.000000D+00
     $   *MDL_MTA__EXP__2)**1.500000D+00+(3.000000D+00*MDL_CT2
     $   *MDL_MU__EXP__2*(MDL_MS2__EXP__2-4.000000D+00*MDL_MU__EXP__2)
     $   **1.500000D+00)/MDL_C)

        MDL_COMPLEXI = DCMPLX(0.000000D+00,1.000000D+00)

        MDL_SQRT__2 = SQRT(DCMPLX(2.000000D+00))

        MDL_AEW = 1.000000D+00/AEWM1

        MDL_SQRT__AEW = SQRT(DCMPLX(MDL_AEW))

        MDL_EE = 2.000000D+00*MDL_SQRT__AEW*SQRT(DCMPLX(PI))

      ENDIF
C     
C     Parameters that should be recomputed at an event by even basis.
C     
      AS = G**2/4/PI

      MDL_SQRT__AS = SQRT(DCMPLX(AS))

      MDL_G__EXP__2 = G**2

C     
C     Parameters that should be updated for the loops.
C     
C     
C     Definition of the EW coupling used in the write out of aqed
C     
      GAL(1) = 3.5449077018110318D0 / DSQRT(ABS(AEWM1))
      GAL(2) = 1D0

