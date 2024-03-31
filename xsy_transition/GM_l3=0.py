import numpy as np
from cosmoTransitions import generic_potential

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
    def init(self, M1=250, M5=500,lam2=2,lam4=2):
        self.Ndim = 2
        self.renormScaleSq = C_VEV**2

        self.g = 2.0*C_MW/C_VEV
        self.gp = np.sqrt(4.0*(C_MZ**2-C_MW**2)/C_VEV**2)
        self.CW = C_MW/C_MZ
        self.SW = np.sqrt(1.0-self.CW**2)
        self.TW = self.SW/self.CW
        self.yt = np.sqrt(2)*C_MT/C_VEV
        self.yb = np.sqrt(2)*C_MB/C_VEV
        self.set_physical_parameters(M1, M5,lam2,lam4)

    def set_physical_parameters(self, M1, M5,lam2,lam4):
        self.M5=M5
        self.M1=M1
        self.lam2 = lam2
        self.lam4 = lam4
        self.v = C_VEV
        self.MH = C_MH
        self.M3 = (self.M5**2+2*self.M1**2)/3
        self.lam1 = self.MH**2/self.v**2
        self.lam5 =2*(self.M5**2-self.M1**2)/self.v**2

        self.mu22 = -4.0*self.lam1*self.v**2
        self.mu32 = (self.M1**2+2*self.M5**2-4*self.lam2*self.v**2)/2

    def V0(self, X):
        X = np.asanyarray(X)
        omg1 = X[...,0]
        omg5 = X[...,1]

        r2 = self.mu22*omg1**2/2.0 + 3.0*self.mu32*omg5**2/2.0
        r4 = self.lam1*omg1**4 + 3.0*(2.0*self.lam2-self.lam5)*omg1**2*omg5**2/2.0+9*self.lam4*omg5**4

        return r2 + r4

    def Vtot(self, X, T, include_radiation = True):

        T = np.asanyarray(T)
        X = np.asanyarray(X)
        omg1 = X[..., 0]
        omg5 = X[..., 1]

        y = self.V0(X)
        y += 1/6*self.mu22*T**2 + 3/8*self.mu32*T**2 + self.lam1*omg1**2*T**2 + 3/4*self.lam2*omg1**2*T**2
        y += self.lam2*omg5**2*T**2+ 11/2*self.lam4*omg5**2*T**2
        y += 1/16*self.g**2*(omg1**2+8*omg5**2)*T**2+ 1/32*(self.g**2+self.gp**2)*(omg1**2+8*omg5**2)*T**2
        y += 1/8*self.yt**2*omg1**2*T**2
        return y


    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        v = C_VEV
        return [np.array([v,0])]

def makePlots(m=None):
    if m is None:
        m = GM_Z2_15()
        m.calcTcTrans()

makePlots()
