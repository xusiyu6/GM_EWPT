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

    def check_EWPT_critical(self):
        if self.c3/self.c2 >= 3.0*self.mu32/self.mu22:
            return False
        if 3.0*self.mu32/self.mu22 >= 3.0*(2.0*self.lam2 - self.lam5)/4.0/self.lam1:
            return False
        return True

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

    def Tc_Analytical(self):
        return np.sqrt((3.0*self.mu22*np.sqrt(self.lam4)-3.0*self.mu32*np.sqrt(self.lam1))/(self.c3*np.sqrt(self.lam1)-3.0*self.c2*np.sqrt(self.lam4)))

    def V0(self, X):  # 重写generic_potential里的方法V0
        X = np.asanyarray(X)
        omg1 = X[..., 0]
        omg5 = X[..., 1]

        r2 = self.mu22 * omg1 ** 2 / 2.0 + 3.0 * self.mu32 * omg5 ** 2 / 2.0
        r4 = self.lam1 * omg1 ** 4 + 3.0 * (
                2.0 * self.lam2 - self.lam5) * omg1 ** 2 * omg5 ** 2 / 2.0 + 9 * self.lam4 * omg5 ** 4

        return r2 + r4

    def Vtot(self, X, T, include_radiation=True):  # 重写generic_potential里的方法Vtot
        T = np.asanyarray(T)
        X = np.asanyarray(X)
        omg1 = X[..., 0]
        omg5 = X[..., 1]

        y = self.V0(X)
        y += self.c2*omg1**2*T**2/2.0 + self.c3*omg5**2*T**2/2.0
        # y += 1 / 6 * self.mu22 * T ** 2 + 3 / 8 * self.mu32 * T ** 2 + self.lam1 * omg1 ** 2 * T ** 2 + 3 / 4 * self.lam2 * omg1 ** 2 * T ** 2
        # y += self.lam2 * omg5 ** 2 * T ** 2 + 11 / 2 * self.lam4 * omg5 ** 2 * T ** 2
        # y += 1 / 16 * self.g ** 2 * (omg1 ** 2 + 8 * omg5 ** 2) * T ** 2 + 1 / 32 * (self.g ** 2 + self.gp ** 2) * (
        #         omg1 ** 2 + 8 * omg5 ** 2) * T ** 2
        # y += 1 / 8 * self.yt ** 2 * omg1 ** 2 * T ** 2
        return y

    def approxZeroTMin(self):  # 重写generic_potential里的方法approxZeroTMin
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        v = C_VEV
        return [np.array([v, 0])]

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
            if not mod.check_UNI() or not mod.check_BFB() or not mod.check_EWPT_critical():
                continue
            Tc_analytical = mod.Tc_Analytical()
            if M1_rnd / Tc_analytical<10 or M5_rnd / Tc_analytical<10 :
                continue
            nucl_res = []
            try:
                nucl_res = mod.findAllTransitions()
            except Exception as e:
                print("Error in numerical calculations: %s"%e)
            for nucl in nucl_res:
                nucl_type = nucl['trantype']
                if nucl_type != 1:
                    continue
                Tn = nucl['Tnuc']
                dRho = nucl['Delta_rho']
                beta = nucl['betaHn_GW']
                crit_res = nucl['crit_trans']
                Tc = -1
                if crit_res:
                    Tc = crit_res['Tcrit']
                n_obtained += 1
                f.write("%f %f %f %f %f %f %f %f %f\n"%(M1_rnd,M5_rnd,lam2_rnd,lam4_rnd,Tn,Tc,Tc_analytical,dRho,beta))
                f.flush()


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d%H%M%S",time.localtime())
    GM_EWPT_Scan("3_GM_EWPT_Scan_%s.dat"%timestamp)