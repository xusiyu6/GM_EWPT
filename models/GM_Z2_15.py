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

def Eigenvalues_2x2(a,b,c):
    """
    Calculate the eigenvalues of a symmetric matrix of
    | a  b |
    | b  c |
    e1 = (a + c - sqrt((a-c)^2+4b^2))/2.0
    e2 = (a + c + sqrt((a-c)^2+4b^2))/2.0
    """
    Delta = (a-c)**2+4.0*b**2
    e1 = (a+c-np.sqrt(Delta))/2.0
    e2 = (a+c+np.sqrt(Delta))/2.0
    return e1,e2

class GM_Z2_15(generic_potential.generic_potential):
    """
    The effective potential for Z2 symmetric GM model with v1 and v8=\sqrt{2}v5 (neglecting other background fields)
    """
    def init(self, sH, sa, MHH, MH3, MH5):
        self.Ndim = 2
        self.renormScaleSq = C_VEV**2

        self.g = 2.0*C_MW/C_VEV
        self.gp = np.sqrt(54.0*(C_MZ**2-C_MW**2)/C_VEV**2)
        self.CW = C_MW/C_MZ
        self.SW = np.sqrt(1.0-self.CW**2)
        self.TW = self.SW/self.CW
        self.yt = np.sqrt(2)*C_MT/C_VEV
        self.yb = np.sqrt(2)*C_MB/C_VEV
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

    def V0(self, X):
        X = np.asanyarray(X)
        omg1 = X[...,0]
        omg5 = X[...,1]

        r2 = self.mu22*omg1**2/2.0 + 3.0*self.mu32*omg5**2/2.0
        r4 = self.lam1*omg1**4 + 3.0*(2.0*self.lam2-self.lam5)*omg1**2*omg5**2/2.0+3.0*(self.lam3+3.0*self.lam4)*omg5**4

        return r2 + r4

    def boson_massSq(self, X, T):
        X = np.asanyarray(X)
        omg1 = X[...,0]
        omg5 = X[...,1]

        mWpmSq, dofmWpm = self.g**2*(omg1**2+8.0*omg5**2)/4.0, 2*3
        mZsq, dofZ = (self.g**2+self.gp**2)*(omg1**2+8.0*omg5**2)/4.0, 3

        atmp = self.mu22 + 12.0*self.lam1*omg1**2+3.0*(2.0*self.lam2-self.lam5)*omg5**2
        btmp = 2.0*np.sqrt(3.0)*(2.0*self.lam2-self.lam5)*omg1*omg5
        ctmp = self.mu32 + (2.0*self.lam2-self.lam5)*omg1**2 + 12.0*(self.lam3+3.0*self.lam4)*omg5**2
        mS1sq, mS2sq = Eigenvalues_2x2(atmp,btmp,ctmp)
        dofS1, dofS2 = 1, 1

        atmp = self.mu22 + 4.0*self.lam1*omg1**2+(6.0*self.lam2+self.lam5)*omg5**2
        btmp = -np.sqrt(2.0)*self.lam5*omg1*omg5
        ctmp = self.mu32 + (4.0*self.lam2-self.lam5)*omg1**2/2.0 + 4.0*(self.lam3+3.0*self.lam4)*omg5**2
        mT1sq, mT2sq = Eigenvalues_2x2(atmp,btmp,ctmp)
        dofT1, dofT2 = 3, 3

        mFsq = self.mu32 + (4.0*self.lam2+self.lam5)*omg1**2/2.0 + 12.0*(self.lam3+self.lam4)*omg5**2
        dofF = 5

        massSq = np.empty(mFsq.shape + (7,))
        massSq[...,0] = mWpmSq
        massSq[...,1] = mZsq
        massSq[...,2] = mS1sq
        massSq[...,3] = mS2sq
        massSq[...,4] = mT1sq
        massSq[...,5] = mT2sq
        massSq[...,6] = mFsq

        dof = np.array([dofmWpm, dofZ, dofS1, dofS2, dofT1, dofT2, dofF])

        c = np.array([5.0/6.0,5.0/6.0,1.5,1.5,1.5,1.5,1.5])

        return massSq, dof, c

    def boson_massSq_thermal(self, X, T):
        X = np.asanyarray(X)
        omg1 = X[...,0]
        omg5 = X[...,1]

        mWpmSq = self.g**2*(omg1**2+8.0*omg5**2)/4.0
        mWpmSqT = mWpmSq + 371.0*self.g**2*T**2/78.0

        mZsq, mAsq = (self.g**2+self.gp**2)*(omg1**2+8.0*omg5**2)/4.0, 0

        atmp = (omg1**2+8.0*omg5**2)*self.g**2/4.0 + 371.0*self.g**2*T**2/78.0
        btmp = -(omg1**2+8.0*omg5**2)*self.g**2/4.0*self.TW
        ctmp = (omg1**2+8.0*omg5**2)*self.g**2*self.TW**2/4.0 + 371.0*self.g**2*T**2*self.TW**2/78.0
        mAsqT, mZsqT = Eigenvalues_2x2(atmp,btmp,ctmp)

        Delta12 = T**2*(9.0*self.g**2/2.0 + 12.0*(4.0*self.lam1 + 3.0*self.lam2)+6.0*(self.yt**2+self.yb**2))/24.0
        Delta22 = T**2*(12.0*self.g**2 + 8.0*(2.0*self.lam2+7.0*self.lam3+11.0*self.lam4))/24.0

        atmp = self.mu22 + 12.0*self.lam1*omg1**2+3.0*(2.0*self.lam2-self.lam5)*omg5**2
        btmp = 2.0*np.sqrt(3.0)*(2.0*self.lam2-self.lam5)*omg1*omg5
        ctmp = self.mu32 + (2.0*self.lam2-self.lam5)*omg1**2 + 12.0*(self.lam3+3.0*self.lam4)*omg5**2
        mS1sq, mS2sq = Eigenvalues_2x2(atmp,btmp,ctmp)
        atmp += Delta12
        ctmp += Delta22
        mS1sqT, mS2sqT = Eigenvalues_2x2(atmp,btmp,ctmp)

        atmp = self.mu22 + 4.0*self.lam1*omg1**2+(6.0*self.lam2+self.lam5)*omg5**2
        btmp = -np.sqrt(2.0)*self.lam5*omg1*omg5
        ctmp = self.mu32 + (4.0*self.lam2-self.lam5)*omg1**2/2.0 + 4.0*(self.lam3+3.0*self.lam4)*omg5**2
        mT1sq, mT2sq = Eigenvalues_2x2(atmp,btmp,ctmp)
        atmp += Delta12
        ctmp += Delta22
        mT1sqT, mT2sqT = Eigenvalues_2x2(atmp,btmp,ctmp)

        mFsq = self.mu32 + (4.0*self.lam2+self.lam5)*omg1**2/2.0 + 12.0*(self.lam3+self.lam4)*omg5**2
        mFsqT = mFsq + Delta22

        massSq = np.empty(mFsq.shape + (8,))
        massSqT = np.empty(mFsqT.shape + (8,))

        massSq[...,0] = (np.abs(mWpmSq) + mWpmSq)/2.0
        massSq[...,1] = (np.abs(mZsq) + mZsq)/2.0
        massSq[...,2] = (np.abs(mAsq) + mAsq)/2.0
        massSq[...,3] = (np.abs(mS1sq) + mS1sq)/2.0
        massSq[...,4] = (np.abs(mS2sq) + mS2sq)/2.0
        massSq[...,5] = (np.abs(mT1sq) + mT1sq)/2.0
        massSq[...,6] = (np.abs(mT2sq) + mT2sq)/2.0
        massSq[...,7] = (np.abs(mFsq) + mFsq)/2.0

        massSqT[...,0] = (np.abs(mWpmSqT) + mWpmSqT)/2.0
        massSqT[...,1] = (np.abs(mZsqT) + mZsqT)/2.0
        massSqT[...,2] = (np.abs(mAsqT) + mAsqT)/2.0
        massSqT[...,3] = (np.abs(mS1sqT) + mS1sqT)/2.0
        massSqT[...,4] = (np.abs(mS2sqT) + mS2sqT)/2.0
        massSqT[...,5] = (np.abs(mT1sqT) + mT1sqT)/2.0
        massSqT[...,6] = (np.abs(mT2sqT) + mT2sqT)/2.0
        massSqT[...,7] = (np.abs(mFsqT) + mFsqT)/2.0

        dof = np.array([2,1,1,1,1,3,3,5])

        return massSq, massSqT, dof

    def fermion_massSq(self, X):
        X=np.asanyarray(X)
        omg1 = X[...,0]

        mTsq, dofT = self.yt**2*omg1**2/2.0, 2*2*3
        mBsq, dofB = self.yb**2*omg1**2/2.0, 2*2*3

        massSq = np.empty(mTsq.shape + (2,))
        massSq[...,0] = mTsq
        massSq[...,1] = mBsq

        dof = np.array([dofT, dofB])

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
