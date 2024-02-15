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
    def init(self, sH, sa, MHH, MH3, MH5):
        self.set_physical_parameters(sH,sa,MHH,MH3,MH5)

    def set_physical_parameters(self, sH, sa, MHH, MH3, MH5):
        if sH > 1:
            sH = 1
        if sH < -1:
            sH = -1
        self.sH = sH
        self.cH = np.sqrt(1-sH*sH)
        self.v1 = C_VEV*self.cH
        self.v5 = C_VEV*self.sH/2.0/np.sqrt(2.0)

        if sa > 1:
            sa = 1
        if sa < -1:
            sa = -1
        self.sa = sa
        self.ca = np.sqrt(1-sa*sa)
        self.MHL = C_MH
        self.MHH = MHH
        self.MH3 = MH3
        self.MH5 = MH5
        self.lam1 = (self.MHL**2*self.ca**2+self.MHH**2*self.sa**2)/8.0/self.v1**2
        self.lam2 = self.MH3**3/C_VEV**2 + self.sa*self.ca/4.0/np.sqrt(3)*(self.MHH**2-self.MHL**2)/self.v1/self.v5
        self.lam3 = self.MH5**2/8.0/self.v5**2 - 3.0*self.v1**2/8.0/self.v5**2*self.MH3**2/C_VEV**2
        self.lam4 = self.v1**2/8.0/self.v5**2*self.MH3**2/C_VEV**2 + (self.MHL**2*self.sa**2+self.MHH**2*self.ca**2-self.MH5**2)/24.0/self.v5**2
        self.lam5 = 2.0*self.MH3**2/C_VEV**2

        self.mu22 = -4.0*self.lam1*self.v1**2 + 3.0*(self.lam5-2.0*self.lam2)*self.v5**2
        self.mu32 = (self.lam5-2.0*self.lam2)*self.v1**2 - 4.0*(self.lam3+3.0*self.lam4)*self.v5**2
