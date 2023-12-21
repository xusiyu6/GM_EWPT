import numpy as np
from cosmoTransitions import generic_potential

from cosmoTransitions.finiteT import Jb_spline as Jb
from cosmoTransitions.finiteT import Jf_spline as Jf

class model2(generic_potential.generic_potential):
    def init(self, v1,v5):
        self.Ndim = 2
        self.v0=246.
        self.renormScaleSq = self.v0
        self.l1=.0323
        self.l2=2.
        self.l3=-1.
        self.l4=3.
        self.l5=2.
        self.v1=v1
        self.v5=v5
        self.v0=self.v1**2+8*self.v5**2
        self.mu22=-4*self.v1**2*self.l1+3*self.v5*(self.l5-2*self.l2)
        self.mu32=self.v1**2*(self.l5-2*self.l2)-4*self.v5**2*(self.l3+3*self.l4)

        # SM coupling constants in natural units
        self.thetaW = np.arcsin(np.sqrt(0.23129))  # Weinberg angle
        self.e = np.sqrt(4 * np.pi * 7.2973525664e-3)  # electric charge
        self.g = self.e / np.sin(self.thetaW)  # eletroweak coupling constant
        self.gprime = self.e / np.cos(self.thetaW)  # eletroweak coupling constant
        self.yt = 1  # top quark Yukawa coupling constant

    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry
        in the theory and you don't want to double-count all of the phases. In
        this case, we're throwing away all phases whose zeroth (since python
        starts arrays at 0) field component of the vev goes below -5. Note that
        we don't want to set this to just going below zero, since we are
        interested in phases with vevs exactly at 0, and floating point numbers
        will never be accurate enough to ensure that these aren't slightly
        negative.
        """
        return (np.array([X])[..., 0] < -5.0).any()

    def V0(self, X):
        X = np.asanyarray(X)
        v1, v5 = X[..., 0], X[..., 1]
        y=1/2*(2*self.l1*v1**4+v1**2*(self.mu22+v5**2*(6*self.l2-3*self.l5))+6*v5**4*(self.l3+3*self.l4)+3*self.mu32*v5**2)
        return y

    def boson_massSq(self, X, T):
        X = np.array(X)
        v1, v5 = X[..., 0], X[..., 1]

        g = self.g
        gprime = self.gprime
        yt = self.yt

        # scalar
        A =4*self.l1*v1**2
        B =4*v5**2*(self.l3+3*self.l4)
        C = v1*v5*(self.l5-2*self.l2)
        mh_Sq = A + B - 2*np.sqrt((A - B) ** 2 + 3 * C ** 2)
        mH_Sq = A + C + 2*np.sqrt((A - C) ** 2 + 4 * B ** 2)
        m3_Sq = 1/2*self.l5*(v1**2+8*v5**2)
        m5_Sq = 3/2*self.l5*v1**2+8*self.l3*v5**2

        # W boson
        mW_Sq = g ** 2 * np.abs(v1**2 + 8 * v5**2) / 4

        # Z boson
        mZ_Sq= (g ** 2+ gprime**2) * np.abs(v1**2 + 8 * v5**2) / 4

        # Without Goldstone Bosons
        dof = np.array([1, 1, 3, 5, 6, 3])
        c = np.array([1.5, 1.5, 1.5, 1.5, 5/6, 5/6])
        MSq = np.array([mh_Sq, mH_Sq, m3_Sq, m5_Sq, mW_Sq, mZ_Sq])
        MSq = np.rollaxis(MSq, 0, len(MSq.shape))

        return MSq,  dof, c

    def fermion_massSq(self, X):

        X = np.array(X)
        v1, v5 = X[..., 0], X[..., 1]

        # Top quark
        mt_Sq = self.yt ** 2 * v1**2 / 2

        MSq = np.array([mt_Sq])
        MSq = np.rollaxis(MSq, 0, len(MSq.shape))
        dof = np.array([12])
        return MSq,  dof

    def V1(self, bosons, fermions):
        """
        Compute the one-loop Coleman-Weinberg potential
        in the so-called on-shell renormalization.
        """
        MSq, dof, c = bosons
        y = np.sum(dof * MSq * MSq * (np.log(np.abs(MSq /self.renormScaleSq) + 1e-100) - c), axis=-1)

        MSq, dof = fermions
        c = np.asarray([1.5])
        y -= np.sum(dof * MSq * MSq * (np.log(np.abs(MSq / self.renormScaleSq) + 1e-100) - c) , axis=-1)

        return y / (64 * np.pi ** 2)

    def V1T(self, bosons, fermions, T, include_radiation=True):
        # This does not need to be overridden.
        T2 = (T*T)[..., np.newaxis] + 1e-100
             # the 1e-100 is to avoid divide by zero errors
        T4 = T*T*T*T
        MSq, nb, c =bosons
        y = np.sum(nb*Jb(MSq/T2), axis=-1)
        MSq, nf=fermions
        y += np.sum(nf*Jf(MSq/T2), axis=-1)
        if include_radiation:
            if self.num_boson_dof is not None:
                nb = self.num_boson_dof - np.sum(nb)
                y -= nb * np.pi**4 / 45.
            if self.num_fermion_dof is not None:
                nf = self.num_fermion_dof - np.sum(nf)
                y -= nf * 7*np.pi**4 / 360.
        return y*T4/(2*np.pi*np.pi)

    def V1T_from_X(self, X, T, include_radiation=True):
        """
        Calculates the mass matrix and resulting one-loop finite-T potential.

        Useful when calculate temperature derivatives, when the zero-temperature
        contributions don't matter.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V1T(bosons, fermions, T, include_radiation)
        return y

    def thermal_Pi_massSq(self, X, T):

        X = np.array(X)
        v1, v5 = X[..., 0], X[..., 1]

        g = self.g
        gprime = self.gprime
        yt = self.yt

        A = 4 * self.l1 * v1 ** 2
        B = 4 * v5 ** 2 * (self.l3 + 3 * self.l4)
        C = v1 * v5 * (self.l5 - 2 * self.l2)
        D=1/3*self.l1
        E=1/3*self.l3+self.l4
        F=1/8*(9/2*(g**2+gprime**2)+27/2*g**2+4*yt**2)
        G=v1**2+8*v5**2

        # Thermal corrections
        Pi_h = A+ B +(D+ E)*T**2- 2* np.sqrt((A - C+(D- E)*T**2) ** 2 + 4 * C ** 2)+T**2*F
        Pi_H = A+ B +(D+ E)*T**2+ 2* np.sqrt((A - C+(D- E)*T**2) ** 2 + 4 * C ** 2)+T**2*F
        Pi_W = 1/4 * g ** 2 * G +2*g**2*T**2  # W boson
        Pi_Z = (g**2+gprime**2)*(T**2+G/8)+1/8*np.sqrt((g**2+gprime**2)**2*(64*T**4+16*T**2*G)+(g**2+gprime**2)**2*G**2)  # Z boson
        Pi_gamma = (g**2+gprime**2)*(T**2+G/8)-1/8*np.sqrt((g**2+gprime**2)**2*(64*T**4+16*T**2*G)+(g**2+gprime**2)**2*G**2) # photon
        Pi_chi = T**2*F  # Goldstone boson
        Pi_3 = 1/2*self.l5*(v1**2+8*v5**2)+ 3/8*self.l5*T**2+ T**2*F
        Pi_5 = 3/2*self.l5*v1**2+8*self.l3*v5**2+ T**2/24*(3*self.l5+16*self.l3)+ T**2*F

        # With Goldstone Bosons
        Pi_dof = np.array([1, 1, 3, 5, 3, 2, 1, 1])
        Pi_MSq = np.array([Pi_h, Pi_H, Pi_3, Pi_5, Pi_chi,Pi_W,Pi_Z,Pi_gamma])
        Pi_MSq = np.rollaxis(Pi_MSq, 0, len(Pi_MSq.shape))

        return Pi_MSq,  Pi_dof

    def V_d1T(self, thermal, bosons, T):

        Pi_h, Pi_H, Pi_3, Pi_5, Pi_chi,Pi_W,Pi_Z,Pi_gamma = thermal
        mh_Sq, mH_Sq, m3_Sq, m5_Sq, mW_Sq, mZ_Sq=bosons

        y = -T/(12*np.pi)*(Pi_h**(3/2)-mh_Sq**(3/2))
        y += -T/(12*np.pi)*(Pi_H**(3/2)-mH_Sq**(3/2))
        y += -T / (12 * np.pi) * 3*(Pi_3 ** (3 / 2) - m3_Sq ** (3 / 2))
        y += -T / (12 * np.pi) * 5*(Pi_5 ** (3 / 2) - m5_Sq ** (3 / 2))
        y += -T / (12 * np.pi) * 3*(Pi_chi ** (3 / 2) - 0)
        y += -T / (12 * np.pi) * 2*(Pi_W ** (3 / 2) - mW_Sq ** (3 / 2))
        y += -T / (12 * np.pi) * (Pi_Z ** (3 / 2) - mZ_Sq ** (3 / 2))
        y += -T / (12 * np.pi) * (Pi_H ** (3 / 2) - 0)

        return y

    def Vtot(self, X, T, include_radiation=True):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        thermal =self.thermal_Pi_massSq( X, T)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.V1T(bosons, fermions, T, include_radiation)
        y += self.V_d1T( thermal, bosons, T)
        return y