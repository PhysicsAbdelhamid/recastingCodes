ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c      written by the UFO converter
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

      SUBROUTINE COUP1()

      IMPLICIT NONE
      INCLUDE 'model_functions.inc'

      DOUBLE PRECISION PI, ZERO
      PARAMETER  (PI=3.141592653589793D0)
      PARAMETER  (ZERO=0D0)
      INCLUDE 'input.inc'
      INCLUDE 'coupl.inc'
      GC_4 = (4.000000D+00*MDL_CL*MDL_COMPLEXI)/MDL_LAMBDA
      GC_7 = MDL_COMPLEXI*MDL_K*MDL_V
      GC_8 = (2.000000D+00*MDL_COMPLEXI*MDL_MB*MDL_Y1*MDL_SQRT__2)
     $ /MDL_V
      GC_9 = (2.000000D+00*MDL_COMPLEXI*MDL_MC*MDL_Y1*MDL_SQRT__2)
     $ /MDL_V
      GC_10 = (2.000000D+00*MDL_COMPLEXI*MDL_MD*MDL_Y1*MDL_SQRT__2)
     $ /MDL_V
      GC_11 = (2.000000D+00*MDL_COMPLEXI*MDL_MS*MDL_Y1*MDL_SQRT__2)
     $ /MDL_V
      GC_13 = (2.000000D+00*MDL_COMPLEXI*MDL_MTA*MDL_Y1*MDL_SQRT__2)
     $ /MDL_V
      GC_14 = (2.000000D+00*MDL_COMPLEXI*MDL_MU*MDL_Y1*MDL_SQRT__2)
     $ /MDL_V
      GC_15 = (2.000000D+00*MDL_COMPLEXI*MDL_MB*MDL_Y2*MDL_SQRT__2)
     $ /MDL_V
      GC_16 = (2.000000D+00*MDL_COMPLEXI*MDL_MC*MDL_Y2*MDL_SQRT__2)
     $ /MDL_V
      GC_17 = (2.000000D+00*MDL_COMPLEXI*MDL_MD*MDL_Y2*MDL_SQRT__2)
     $ /MDL_V
      GC_18 = (2.000000D+00*MDL_COMPLEXI*MDL_MS*MDL_Y2*MDL_SQRT__2)
     $ /MDL_V
      GC_20 = (2.000000D+00*MDL_COMPLEXI*MDL_MTA*MDL_Y2*MDL_SQRT__2)
     $ /MDL_V
      GC_21 = (2.000000D+00*MDL_COMPLEXI*MDL_MU*MDL_Y2*MDL_SQRT__2)
     $ /MDL_V
      END
