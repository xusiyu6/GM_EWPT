# The name convention:
# GM: Georgi-Machacek Model
# Z2: The Z2 symmetric version, i.e. w/o trilinear terms
# 15: Only consider v1 and v8=\sqrt{2}*v5 (see notation in arXiv:2012.07758)

import numpy as np
from CosmoTransitions.cosmoTransitions import generic_potential

C_VEV = 246.22
C_MW = 80.385
C_MZ = 91.1776
C_MT = 173.5
C_MB = 4.18
C_MH = 125.0
C_PI = 3.1415927

class GM_Z2_15(generic_potential.generic_potential):
    """
    The effective potential for Z2 symmetric GM model with v1 and v8=\sqrt{2}v5 (neglecting other background fields)
    """
    def init(self):
        pass

    def set_potential_parameters(self, mu22, mu32, lam1, lam2)
