import numpy as np
import random
import time
import sys
from os import path
curdir=path.dirname(path.abspath(__file__))
sys.path.append(curdir+'/..')
from CosmoTransitions.cosmoTransitions import generic_potential


C_VEV = 246.22
C_MW = 80.385
C_MZ = 91.1776
C_MT = 173.5
C_MB = 4.18
C_MH = 125.0
C_PI = 3.1415927

class GM_Z2_15(generic_potential.generic_potential):

    def init(self, M1, M5, lam2, lam4):  # 重写generic_potential里的方法init
        self.Ndim = 2
        self.renormScaleSq = C_VEV ** 2
        self.g = 2.0 * C_MW / C_VEV
        self.gp = np.sqrt(4.0 * (C_MZ ** 2 - C_MW ** 2) / C_VEV ** 2)
        self.CW = C_MW / C_MZ
        self.SW = np.sqrt(1.0 - self.CW ** 2)
        self.TW = self.SW / self.CW
        self.yt = np.sqrt(2) * C_MT / C_VEV
        self.yb = np.sqrt(2) * C_MB / C_VEV
        self.set_physical_parameters(M1, M5, lam2, lam4)

    def set_physical_parameters(self, M1, M5, lam2, lam4):
        self.M1 = M1
        self.M5 = M5
        self.lam2 = lam2
        self.lam4 = lam4
        self.v = C_VEV
        self.MH = C_MH
        self.M3 =np.sqrt((self.M5 ** 2 + 2 * self.M1 ** 2) / 3)
        self.lam1 = self.MH ** 2 / (8*self.v ** 2)
        self.lam5 =  2*(self.M5 ** 2 - self.M1 ** 2) / (3*self.v** 2)
        self.mu22 = -4.0 * self.lam1 * self.v ** 2
        self.mu32 = (self.M1 ** 2 + 2 * self.M5 ** 2 - 6 * self.lam2 * self.v ** 2) / 3
        self.c2 = (3.0*self.g**2+self.gp**2)/16.0 + self.yt**2/4.0 + 2.0*self.lam1 + 1.5*self.lam2
        self.c3 = (3.0*self.g**2+self.gp**2)/2.0 + 2.0*self.lam2 + 11.0*self.lam4

    def check_UNI(self, method = 0): # Checking whether the Unitarity conditions are satisfied
        # Method 1: |a0|<1;
        # Method 0: |Rea0|<1/2;
        factor=1
        if method == 1:
            factor=2
        if np.sqrt((6.0*self.lam1 - 11.0*self.lam4)**2+36.0*self.lam2**2) + np.abs(6.0*self.lam1 + 11.0*self.lam4) >= 4.0*factor*np.pi:
            return False
        if np.sqrt((2.0*self.lam1-2.0*self.lam4)**2+self.lam5**2)+np.abs(2.0*self.lam1+2.0*self.lam4) >= 4.0*factor*np.pi:
            return False
        if np.abs(self.lam4) >= factor*np.pi:
            return False
        if np.abs(self.lam2-self.lam5) >= factor*2*np.pi:
            return False
        return True

    def check_BFB(self): # Checking whether the BFB conditions are satisfied
        # Works only for lam3 = 0
        if self.lam1 <= 0:
            return False
        if self.lam4 <= 0:
            return False
        if self.lam5>=0 and self.lam2 <= self.lam5/2.0 - 2.0*np.sqrt(self.lam1*self.lam4):
            return False
        if self.lam5<0 and self.lam2 <= -self.lam5/4.0 - 2.0*np.sqrt(self.lam1*self.lam4):
            # This only works for lam3 = 0
            # where we only need to check for zeta = 1/2 where omega_minus obtain its minimum negative value
            return False
        return True


def generate_random_number(range_start, range_end):

    return random.uniform(range_start, range_end)


def GM_EWPT_Scan(filename,n_required=500):
    with open(filename,'w') as f:
        n_obtained = 0
        while n_obtained < n_required:
            M1_rnd = generate_random_number(0,2000)
            M5_rnd = generate_random_number(0,2000)
            lam2_rnd = generate_random_number(-3,3)
            lam4_rnd = generate_random_number(0,3.2)

            mod = GM_Z2_15(M1_rnd,M5_rnd,lam2_rnd,lam4_rnd)
            if not mod.check_UNI() or not mod.check_BFB() :
                continue
           
            f.write("%f %f %f %f %f %f %f %f %f\n"%(M1_rnd,M5_rnd,lam2_rnd,lam4_rnd))
            f.flush()


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d%H%M%S",time.localtime())
    GM_EWPT_Scan("1_GM_EWPT_Scan_%s.dat"%timestamp)
