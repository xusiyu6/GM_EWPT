import numpy as np
from CosmoTransitions.cosmoTransitions import generic_potential

C_VEV = 246.22
C_MW = 80.385
C_MZ = 91.1776
C_MT = 173.5
C_MB = 4.18
C_MH = 125.0
C_PI = 3.1415927

class SM(generic_potential.generic_potential):
    """
    SM model for CosmoTransitions
    We assume the Higgs mass has not been measured yet
    """
    def init(self, mh=C_MH):
        self.Ndim = 1
        self.renormScaleSq = C_VEV**2

        self.MH = mh
        self.mu2 = -mh**2/2.0
        self.lam = mh**2/2.0/C_VEV**2

        self.g = 2.0*C_MW/C_VEV
        self.gp = np.sqrt(4.0*(C_MZ**2-C_MW**2)/C_VEV**2)

        self.yt = np.sqrt(2)*C_MT/C_VEV
        self.yb = np.sqrt(2)*C_MB/C_VEV


    def V0(self, X):
        X = np.asanyarray(X)
        phi = X[..., 0]

        r2 = self.mu2/2.0*phi**2
        r4 = self.lam/4.0*phi**4

        return r2 + r4


    def boson_massSq(self, X, T):
        X = np.asanyarray(X)
        phi = X[...,0]

        mWpmSq, dofmWpm = self.g**2*phi**2/4.0, 2*3
        mZsq, dofZ = (self.g**2+self.gp**2)*phi**2/4.0, 3

        mHsq, dofH = self.mu2 + 3.0*self.lam*phi**2, 1
        mChisq, dofChi= self.mu2 + self.lam*phi**2, 3

        massSq = np.empty(mHsq.shape + (4,))
        massSq[...,0] = mWpmSq
        massSq[...,1] = mZsq
        massSq[...,2] = mHsq
        massSq[...,3] = mChisq
        dof = np.array([dofmWpm,dofZ,dofH,dofChi])
        c = np.array([5.0/6.0,5.0/6.0,1.5,1.5])
        return massSq, dof, c

    def boson_massSq_thermal(self, X, T):
        X = np.asanyarray(X)
        phi = X[...,0]

        mWpmSq = self.g**2*phi**2/4.0
        mWpmSqT = self.g**2*phi**2/4.0 + 11.0*self.g**2*T**2/6.0

        term1 = (self.g**2+self.gp**2)*(22.0*T**2+3.0*phi**2)
        term2 = np.sqrt(9.0*(self.g**2+self.gp**2)**2*phi**4+132.0*(self.g**2-self.gp**2)**2*T**2*phi**2+484.0*(self.g**2-self.gp**2)**2*T**4)

        mZsq = (self.g**2+self.gp**2)*phi**2/4.0
        mZsqT = (term1+term2)/24.0
        mAsq = 0.
        mAsqT = (term1-term2)/24.0

        termS = (6.0*(self.yt**2+self.yb**2)+12.0*self.lam+3.0*(3.0*self.g**2+self.gp**2)/2.0)*T**2/24.0

        mHsq = self.mu2 + 3.0*self.lam*phi**2
        mHsqT = self.mu2 + 3.0*self.lam*phi**2 + termS
        mChisq = self.mu2 + self.lam*phi**2
        mChisqT = self.mu2 + self.lam*phi**2 + termS

        massSq = np.empty(mWpmSqT.shape + (5,))
        massSqT = np.empty(mWpmSqT.shape + (5,))

        massSq[...,0] = (np.abs(mWpmSq) + mWpmSq)/2.0
        massSq[...,1] = (np.abs(mZsq) + mZsq)/2.0
        massSq[...,2] = (np.abs(mAsq) + mAsq)/2.0
        massSq[...,3] = (np.abs(mHsq) + mHsq)/2.0
        massSq[...,4] = (np.abs(mChisq) + mChisq)/2.0

        massSqT[...,0] = (np.abs(mWpmSqT) + mWpmSqT)/2.0
        massSqT[...,1] = (np.abs(mZsqT)+mZsqT)/2.0
        massSqT[...,2] = (np.abs(mAsqT)+mAsqT)/2.0
        massSqT[...,3] = (np.abs(mHsqT)+mHsqT)/2.0
        massSqT[...,4] = (np.abs(mChisqT)+mChisqT)/2.0

        dof = np.array([2,1,1,1,3])

        return massSq, massSqT, dof

    def fermion_massSq(self, X):
        X=np.asanyarray(X)
        phi = X[...,0]

        mTsq, dofT = self.yt**2*phi**2/2.0, 2*2*3
        mBsq, dofB = self.yb**2*phi**2/2.0, 2*2*3

        massSq = np.empty(mTsq.shape + (2,))
        massSq[...,0] = mTsq
        massSq[...,1] = mBsq

        dof = np.array([dofT,dofB])

        return massSq, dof

    def daisy(self, X, T):

        m2, m2T, dof = self.boson_massSq_thermal(X,T)

        y = np.sum(-T*(m2T**1.5-m2**1.5)*dof/12.0/C_PI,axis=-1)

        return y

    def Vtot(self, X, T, include_radiation = True):

        T = np.asanyarray(T)
        X = np.asanyarray(X)

        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.V1T_from_X(X, T, include_radiation)
        y += self.daisy(X,T)

        return y
