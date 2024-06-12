import numpy as np
import random
import time
import sys
from os import path

curdir = path.dirname(path.abspath(__file__))
sys.path.append(curdir + '/..')
from CosmoTransitions.cosmoTransitions import generic_potential

C_VEV = 246.22
C_MW = 80.385
C_MZ = 91.1776
C_MT = 173.5
C_MB = 4.18
C_MH = 125.0
C_PI = 3.1415927

def omega(zeta, B):
    omega_plus = 1 / 6 * (1 - B) + np.sqrt(2) / 3 * np.sqrt((1 - B) * (1 / 2 + B))
    omega_minus = 1 / 6 * (1 - B) - np.sqrt(2) / 3 * np.sqrt((1 - B) * (1 / 2 + B))
    return omega_plus, omega_minus

class GM_Z2_15(generic_potential.generic_potential):
    def __init__(self, M1, M5, lam2, lam3, lam4):
        self.Ndim = 2
        self.renormScaleSq = C_VEV ** 2
        self.g = 2.0 * C_MW / C_VEV
        self.gp = np.sqrt(4.0 * (C_MZ ** 2 - C_MW ** 2) / C_VEV ** 2)
        self.CW = C_MW / C_MZ
        self.SW = np.sqrt(1.0 - self.CW ** 2)
        self.TW = self.SW / self.CW
        self.yt = np.sqrt(2) * C_MT / C_VEV
        self.yb = np.sqrt(2) * C_MB / C_VEV
        self.set_physical_parameters(M1, M5, lam2, lam3, lam4)

    def set_physical_parameters(self, M1, M5, lam2, lam3, lam4):
        self.M1 = M1
        self.M5 = M5
        self.lam2 = lam2
        self.lam3 = lam3
        self.lam4 = lam4
        self.v = C_VEV
        self.MH = C_MH
        self.M3 = np.sqrt((self.M5 ** 2 + 2 * self.M1 ** 2) / 3)
        self.lam1 = self.MH ** 2 / (8 * self.v ** 2)
        self.lam5 = 2 * (self.M5 ** 2 - self.M1 ** 2) / (3 * self.v ** 2)
        self.mu22 = -4.0 * self.lam1 * self.v ** 2
        self.mu32 = (self.M1 ** 2 + 2 * self.M5 ** 2 - 6 * self.lam2 * self.v ** 2) / 3

    def check_BFB(self):  # Checking whether the BFB conditions are satisfied
        if self.lam1 <= 0:
            return False
        if self.lam3 >= 0 and self.lam4 <= -1 / 3 * self.lam3:
            return False
        if self.lam3 < 0 and self.lam4 <= -self.lam3:
            return False

        if self.lam5 >= 0:
            if self.lam3 >= 0 and self.lam2 <= self.lam5 / 2.0 - 2.0 * np.sqrt(self.lam1 * (1 / 3 * self.lam3 + self.lam4)):
                return False
            if self.lam3 < 0:
                for zeta in np.arange(1 / 3, 1, 0.001):
                    B = np.sqrt(3 / 2 * (zeta - 1 / 3))
                    omega_plus, _ = omega(zeta, B)
                    if self.lam2 <= omega_plus * self.lam5 - 2.0 * np.sqrt(self.lam1 * (zeta * self.lam3 + self.lam4)):
                        return False

        if self.lam5 < 0:
            for zeta in np.arange(1 / 3, 1, 0.001):
                B = np.sqrt(3 / 2 * (zeta - 1 / 3))
                _, omega_minus = omega(zeta, B)
                if self.lam2 <= omega_minus * self.lam5 - 2.0 * np.sqrt(self.lam1 * (zeta * self.lam3 + self.lam4)):
                    return False
        return True

    def check_UNI(self, method=0):  # Checking whether the Unitarity conditions are satisfied
        factor = 1
        if method == 1:
            factor = 2
        if np.sqrt((6.0 * self.lam1 - 7.0 * self.lam3 - 11.0 * self.lam4) ** 2 + 36.0 * self.lam2 ** 2) + np.abs(6.0 * self.lam1 + 7.0 * self.lam3 + 11.0 * self.lam4) >= 4.0 * factor * np.pi:
            return False
        if np.sqrt((2.0 * self.lam1 + self.lam3 - 2.0 * self.lam4) ** 2 + self.lam5 ** 2) + np.abs(2.0 * self.lam1 - self.lam3 + 2.0 * self.lam4) >= 4.0 * factor * np.pi:
            return False
        if np.abs(2.0 * self.lam3 + self.lam4) >= factor * np.pi:
            return False
        if np.abs(self.lam2 - self.lam5) >= factor * 2 * np.pi:
            return False
        return True


def generate_random_number(range_start, range_end):
    return random.uniform(range_start, range_end)


def GM_EWPT_Scan(filename, n_required=500):
    with open(filename, 'w') as f:
        n_obtained = 0
        while n_obtained < n_required:
            M1_rnd = generate_random_number(40, 100)
            M5_rnd = generate_random_number(40, 800)
            lam2_rnd = generate_random_number(-5, 5)
            lam3_rnd = generate_random_number(-5, 5)
            lam4_rnd = generate_random_number(-5, 5)

            mod = GM_Z2_15(M1_rnd, M5_rnd, lam2_rnd, lam3_rnd, lam4_rnd)
            if not mod.check_UNI() or not mod.check_BFB():
                continue

            f.write("%f %f %f %f %f \n" % (M1_rnd, M5_rnd, lam2_rnd, lam3_rnd, lam4_rnd))
            f.flush()
            n_obtained += 1


if __name__ == "__main__":
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    GM_EWPT_Scan("1zetaGM_%s.dat" % timestamp)
