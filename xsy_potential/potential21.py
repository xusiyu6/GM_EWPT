import numpy as np
from cosmoTransitions import generic_potential

import matplotlib.pyplot as plt

from cosmoTransitions.finiteT import Jb_spline as Jb
from cosmoTransitions.finiteT import Jf_spline as Jf

v02 = 246.**2

class model2(generic_potential.generic_potential):
    def init(self, v1=246):
        self.Ndim = 2
        self.v02 = 246.**2
        self.renormScaleSq = self.v02
        self.l1 = .05
        self.l2 = 2.
        self.l3 = -.5
        self.l4 = 2
        self.l5 = 5.
        self.v1 = v1
        self.v5 = np.sqrt((self.v02-self.v1 ** 2)/8)
        self.mu22 = -4 * self.v1 ** 2 * self.l1 + 3 * self.v5 **2* (self.l5 - 2 * self.l2)
        self.mu32 = self.v1 ** 2 * (self.l5 - 2 * self.l2) - 4 * self.v5 ** 2 * (self.l3 + 3 * self.l4)

        # SM coupling constants in natural units
        self.thetaW = np.arcsin(np.sqrt(0.23129))  # Weinberg angle
        self.e = np.sqrt(4 * np.pi * 7.2973525664e-3)  # electric charge
        self.g = self.e / np.sin(self.thetaW)  # eletroweak coupling constant
        self.gprime = self.e / np.cos(self.thetaW)  # eletroweak coupling constant
        self.yt = 1  # top quark Yukawa coupling constant

        # print(self.v5)#这个是零温时候的v5
        # print(self.mu22)
        # print(self.mu32)
        # print(self.g)
        # print(self.gprime)

    def forbidPhaseCrit(self, X):
        return (np.array([X])[..., 0] < -5.0).any() or (np.array([X])[..., 1] < -5.0).any()

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

        # 构建矩阵
        matrix = np.array([
            [self.mu22  + 12 * self.l1 * v1 ** 2 + (6 * self.l2 - 3 * self.l5) * v5 ** 2, 0, 0, 0,
             2 * (2 * self.l2 - self.l5) * v1 * v5, 0, 0,
             2 * np.sqrt(2) * (2 * self.l2 - self.l5) * v1 * v5, 0, 0, 0, 0, 0],
            [0, self.mu22  + 4 * self.l1 * v1 ** 2 + (6 * self.l2 + self.l5) * v5 ** 2, 0, 0, 0, 0, 0, 0,
             -np.sqrt(2) * self.l5 * v1 * v5, 0,
             0, 0, 0],
            [0, 0, self.mu22  + 4 * self.l1 * v1 ** 2 + (6 * self.l2 + self.l5) * v5 ** 2, 0, 0,
             self.l5 * (-v1) * v5, 0, 0, 0,
             self.l5 * (-v1) * v5, 0, 0, 0],
            [0, 0, 0, self.mu22  + 4 * self.l1 * v1 ** 2 + (6 * self.l2 + self.l5) * v5 ** 2, 0, 0,
             self.l5 * (-v1) * v5, 0, 0, 0,
             self.l5 * (-v1) * v5, 0, 0],
            [2 * (2 * self.l2 - self.l5) * v1 * v5, 0, 0, 0,
             self.mu32  + 2 * self.l2 * v1 ** 2 + 4 * (3 * self.l3 + 5 * self.l4) * v5 ** 2, 0, 0,
             (16 * self.l4 * v5 ** 2 - self.l5 * v1 ** 2) / np.sqrt(2), 0, 0, 0, 0, 0],
            [0, 0, self.l5 * (-v1) * v5, 0, 0,
             self.mu32 + 2 * self.l2 * v1 ** 2 + 4 * (2 * self.l3 + 3 * self.l4) * v5 ** 2, 0, 0, 0,
             -1 / 2 * self.l5 * v1 ** 2 - 4 * self.l3 * v5 ** 2, 0, 0, 0],
            [0, 0, 0, self.l5 * (-v1) * v5, 0, 0,
             self.mu32+ 2 * self.l2 * v1 ** 2 + 4 * (2 * self.l3 + 3 * self.l4) * v5 ** 2, 0, 0, 0,
             -1 / 2 * self.l5 * v1 ** 2 - 4 * self.l3 * v5 ** 2, 0, 0],
            [2 * np.sqrt(2) * (2 * self.l2 - self.l5) * v1 * v5, 0, 0, 0,
             (16 * self.l4 * v5 ** 2 - self.l5 * v1 ** 2) / np.sqrt(2), 0, 0,
             self.mu32 + (2 * self.l2 - self.l5 / 2) * v1 ** 2 + 4 * (3 * self.l3 + 7 * self.l4) * v5 ** 2, 0, 0,
             0, 0, 0],
            [0, -np.sqrt(2) * self.l5 * v1 * v5, 0, 0, 0, 0, 0, 0,
             self.mu32 + (2 * self.l2 - self.l5 / 2) * v1 ** 2 + 4 * (self.l3 + 3 * self.l4) * v5 ** 2, 0, 0, 0,
             0],
            [0, 0, self.l5 * (-v1) * v5, 0, 0, -1 / 2 * self.l5 * v1 ** 2 - 4 * self.l3 * v5 ** 2, 0, 0, 0,
             self.mu32 + 2 * self.l2 * v1 ** 2 + 4 * (2 * self.l3 + 3 * self.l4) * v5 ** 2, 0, 0, 0],
            [0, 0, 0, self.l5 * (-v1) * v5, 0, 0, -1 / 2 * self.l5 * v1 ** 2 - 4 * self.l3 * v5 ** 2, 0, 0, 0,
             self.mu32  + 2 * self.l2 * v1 ** 2 + 4 * (2 * self.l3 + 3 * self.l4) * v5 ** 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             self.mu32  + 1 / 2 * (4 * self.l2 + self.l5) * v1 ** 2 + 12 * (self.l3 + self.l4) * v5 ** 2,
             0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             self.mu32  + 1 / 2 * (4 * self.l2 + self.l5) * v1 ** 2 + 12 * (self.l3 + self.l4) * v5 ** 2]
        ])

        # 计算特征值
        eigenvalues = np.linalg.eigvals(matrix)

        # 将特征值赋值给对应的变量
        m12, m22, m32, m42, m52, m62, m72, m82, m92, m102, m112,m122, m132 = eigenvalues

        # 打印结果
        # print("m12:", m12)
        # print("m22:", m22)
        # print("m32:", m32)
        # print("m42:", m42)
        # print("m52:", m52)
        # print("m62:", m62)
        # print("m72:", m72)
        # print("m82:", m82)
        # print("m92:", m92)
        # print("m102:", m102)
        # print("m112:", m112)
        # print("m122:", m122)
        # print("m132:", m132)

        # W boson
        mW_Sq = g ** 2 * np.abs(v1**2 + 8 * v5**2) / 4

        # Z boson
        mZ_Sq= (g ** 2+ gprime**2) * np.abs(v1**2 + 8 * v5**2) / 4

        return m12, m22, m32, m42, m52, m62, m72, m82, m92, m102, m112,m122, m132,mW_Sq, mZ_Sq

    def fermion_massSq(self, X):

        X = np.array(X)
        v1, v5 = X[..., 0], X[..., 1]

        # Top quark
        mt_Sq = self.yt ** 2 * v1**2 / 2

        dof = np.array([12])
        return mt_Sq,  dof

    def V1(self, bosons, fermions):
        m12, m22, m32, m42, m52, m62, m72, m82, m92, m102,m112, m122, m132,mW_Sq, mZ_Sq = bosons

        y = np.sum(m12 * m12 * (1/2*np.log((m12 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m22 * m22 * (1/2*np.log((m22 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m32 * m32 * (1/2*np.log((m32 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m42 * m42 * (1/2*np.log((m42 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m52 * m52 * (1/2*np.log((m52 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m62 * m62 * (1/2*np.log((m62 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m72 * m72 * (1/2*np.log((m72 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m82 * m82 * (1/2*np.log((m82 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m92 * m92 * (1/2*np.log((m92 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m102 * m102 * (1/2*np.log((m102 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m112 * m112 * (1/2*np.log((m112 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m122 * m122 * (1/2*np.log((m122 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(m132 * m132 * (1/2*np.log((m132 / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)
        y += np.sum(6 * mW_Sq * mW_Sq * (1/2*np.log((mW_Sq / self.renormScaleSq)**2 + 1e-100) - 5 / 6), axis=-1)
        y += np.sum(3 * mZ_Sq * mZ_Sq * (1/2*np.log((mZ_Sq / self.renormScaleSq)**2 + 1e-100) - 5 / 6), axis=-1)

        mt_Sq, dof = fermions
        y -= 12 * np.sum(mt_Sq * mt_Sq * (1/2*np.log((mt_Sq / self.renormScaleSq)**2 + 1e-100) - 1.5), axis=-1)

        return y / (64 * np.pi ** 2)

    def V1T(self, bosons, fermions, T, include_radiation=True):
        # This does not need to be overridden.
        T2 = (T * T)+ 1e-100
        # the 1e-100 is to avoid divide by zero errors
        T4 = T * T * T * T
        m12, m22, m32, m42, m52, m62, m72, m82, m92, m102,m112, m122, m132,mW_Sq, mZ_Sq = bosons
        y = np.sum(Jb(abs(m12) / T2), axis=-1)
        y += np.sum(Jb(abs(m22) / T2), axis=-1)
        y += np.sum(Jb(abs(m32) / T2), axis=-1)
        y += np.sum(Jb(abs(m42) / T2), axis=-1)
        y += np.sum(Jb(abs(m52) / T2), axis=-1)
        y += np.sum(Jb(abs(m62) / T2), axis=-1)
        y += np.sum(Jb(abs(m72) / T2), axis=-1)
        y += np.sum(Jb(abs(m82 )/ T2), axis=-1)
        y += np.sum(Jb(abs(m92) / T2), axis=-1)
        y += np.sum(Jb(abs(m102) / T2), axis=-1)
        y += np.sum(Jb(abs(m112 )/ T2), axis=-1)
        y += np.sum(Jb(abs(m122 )/ T2), axis=-1)
        y += np.sum(Jb(abs(m132 )/ T2), axis=-1)
        y += np.sum(6 * Jb(mW_Sq / T2), axis=-1)
        y += np.sum(3 * Jb(mZ_Sq / T2), axis=-1)
        mt_Sq = fermions[0]
        y -= np.sum(12 * Jf(mt_Sq / T2), axis=-1)
        return y * T4 / (2 * np.pi * np.pi)

    def V1T_from_X(self, X, T, include_radiation=True):
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
        mu22=self.mu22
        mu32=self.mu32
        l1=self.l1
        l2=self.l2
        l3=self.l3
        l4=self.l4
        l5=self.l5

        # 构建矩阵
        matrix = np.array([
            [self.mu22 + 12 * self.l1 * v1 ** 2 + (6 * self.l2 - 3 * self.l5) * v5 ** 2 + (1 / 24) * T ** 2 * (
                        3 * ((3 * g ** 2) / 2 + (gprime ** 2) / 2) + 48 * self.l1 + 36 * self.l2 + 6 * yt ** 2), 0, 0, 0,
             2 * (2 * l2 - l5) * v1 * v5, 0, 0, 2 * np.sqrt(2) * (2 * l2 - l5) * v1 * v5, 0, 0, 0, 0, 0],
            [0, mu22+ 4 * l1 * v1 ** 2 + (6 * l2 + l5) * v5 ** 2 + (1 / 24) * T ** 2 * (
                        3 * ((3 * g ** 2) / 2 + (gprime ** 2) / 2) + 48 * l1 + 36 * l2 + 6 * yt ** 2), 0, 0, 0, 0, 0, 0,
             -np.sqrt(2) * l5 * v1 * v5, 0, 0, 0, 0],
            [0, 0, mu22 + 4 * l1 * v1 ** 2 + (6 * l2 + l5) * v5 ** 2 + (1 / 24) * T ** 2 * (
                        3 * ((3 * g ** 2) / 2 + (gprime ** 2) / 2) + 48 * l1 + 36 * l2 + 6 * yt ** 2), 0, 0, l5 * (-v1) * v5,
             0, 0, 0, l5 * (-v1) * v5, 0, 0, 0],
            [0, 0, 0, mu22  + 4 * l1 * v1 ** 2 + (6 * l2 + l5) * v5 ** 2 + (1 / 24) * T ** 2 * (
                        3 * ((3 * g ** 2) / 2 + (gprime ** 2) / 2) + 48 * l1 + 36 * l2 + 6 * yt ** 2), 0, 0, l5 * (-v1) * v5,
             0, 0, 0, l5 * (-v1) * v5, 0, 0],
            [2 * (2 * l2 - l5) * v1 * v5, 0, 0, 0,
             mu32  + 2 * l2 * v1 ** 2 + 4 * (3 * l3 + 5 * l4) * v5 ** 2 + (1 / 24) * T ** 2 * (
                         3 * (4 * g ** 2 + 2 * gprime ** 2) + 8 * (2 * l2 + 7 * l3 + 11 * l4)), 0, 0,
             (16 * l4 * v5 ** 2 - l5 * v1 ** 2) / np.sqrt(2), 0, 0, 0, 0, 0],
            [0, 0, l5 * (-v1) * v5, 0, 0,
             mu32  + 2 * l2 * v1 ** 2 + 4 * (2 * l3 + 3 * l4) * v5 ** 2 + (1 / 24) * T ** 2 * (
                         3 * (4 * g ** 2 + 2 * gprime ** 2) + 8 * (2 * l2 + 7 * l3 + 11 * l4)), 0, 0, 0,
             -1 / 2 * l5 * v1 ** 2 - 4 * l3 * v5 ** 2, 0, 0, 0],
            [0, 0, 0, l5 * (-v1) * v5, 0, 0,
             mu32 + 2 * l2 * v1 ** 2 + 4 * (2 * l3 + 3 * l4) * v5 ** 2 + (1 / 24) * T ** 2 * (
                         3 * (4 * g ** 2 + 2 * gprime ** 2) + 8 * (2 * l2 + 7 * l3 + 11 * l4)), 0, 0, 0,
             -1 / 2 * l5 * v1 ** 2 - 4 * l3 * v5 ** 2, 0, 0],
            [2 * np.sqrt(2) * (2 * l2 - l5) * v1 * v5, 0, 0, 0, (16 * l4 * v5 ** 2 - l5 * v1 ** 2) / np.sqrt(2), 0, 0,
             mu32  + (2 * l2 - l5 / 2) * v1 ** 2 + 4 * (3 * l3 + 7 * l4) * v5 ** 2 + (1 / 24) * T ** 2 * (
                         3 * (4 * g ** 2 + 2 * gprime ** 2) + 8 * (2 * l2 + 7 * l3 + 11 * l4)), 0, 0, 0, 0, 0],
            [0, -np.sqrt(2) * l5 * v1 * v5, 0, 0, 0, 0, 0, 0,
             mu32  + (2 * l2 - l5 / 2) * v1 ** 2 + 4 * (l3 + 3 * l4) * v5 ** 2 + (1 / 24) * T ** 2 * (
                         3 * (4 * g ** 2 + 2 * gprime ** 2) + 8 * (2 * l2 + 7 * l3 + 11 * l4)), 0, 0, 0, 0],
            [0, 0, l5 * (-v1) * v5, 0, 0, -1 / 2 * l5 * v1 ** 2 - 4 * l3 * v5 ** 2, 0, 0, 0,
             mu32  + 2 * l2 * v1 ** 2 + 4 * (2 * l3 + 3 * l4) * v5 ** 2 + (1 / 24) * T ** 2 * (
                         3 * (4 * g ** 2 + 2 * gprime ** 2) + 8 * (2 * l2 + 7 * l3 + 11 * l4)), 0, 0, 0],
            [0, 0, 0, l5 * (-v1) * v5, 0, 0, -1 / 2 * l5 * v1 ** 2 - 4 * l3 * v5 ** 2, 0, 0, 0,
             mu32  + 2 * l2 * v1 ** 2 + 4 * (2 * l3 + 3 * l4) * v5 ** 2 + (1 / 24) * T ** 2 * (
                         3 * (4 * g ** 2 + 2 * gprime ** 2) + 8 * (2 * l2 + 7 * l3 + 11 * l4)), 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             mu32  + 1 / 2 * (4 * l2 + l5) * v1 ** 2 + 12 * (l3 + l4) * v5 ** 2 + (1 / 24) * T ** 2 * (
                         3 * (4 * g ** 2 + 2 * gprime ** 2) + 8 * (2 * l2 + 7 * l3 + 11 * l4)), 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             mu32  + 1 / 2 * (4 * l2 + l5) * v1 ** 2 + 12 * (l3 + l4) * v5 ** 2 + (1 / 24) * T ** 2 * (
                         3 * (4 * g ** 2 + 2 * gprime ** 2) + 8 * (2 * l2 + 7 * l3 + 11 * l4))]
        ])

        # 计算特征值
        eigenvalues = np.linalg.eigvals(matrix)

        # 将特征值赋值给对应的变量
        Pi_m12, Pi_m22, Pi_m32, Pi_m42, Pi_m52, Pi_m62, Pi_m72, Pi_m82, Pi_m92, Pi_m102,Pi_m112, Pi_m122, Pi_m132 = eigenvalues

        # print("Pi_m12:", Pi_m12)
        # print("Pi_m22:", Pi_m22)
        # print("Pi_m32:", Pi_m32)
        # print("Pi_m42:", Pi_m42)
        # print("Pi_m52:", Pi_m52)
        # print("Pi_m62:", Pi_m62)
        # print("Pi_m72:", Pi_m72)
        # print("Pi_m82:", Pi_m82)
        # print("Pi_m92:", Pi_m92)
        # print("Pi_m102:", Pi_m102)
        # print("Pi_m112:", Pi_m112)
        # print("Pi_m122:", Pi_m122)
        # print("Pi_m132:", Pi_m132)

        # Thermal corrections

        # 构建矩阵
        matrix = np.array([
            [1 / 4 * g ** 2 * (v1 ** 2 + 8 * v5 ** 2) + 77 / 6 * g ** 2 * T ** 2, 0, 0, 0],
            [0, 1 / 4 * g ** 2 * (v1 ** 2 + 8 * v5 ** 2) + 77 / 6 * g ** 2 * T ** 2, 0, 0],
            [0, 0, 1 / 4 * g ** 2 * (v1 ** 2 + 8 * v5 ** 2) + 77 / 6 * g ** 2 * T ** 2,
             -1 / 4 * g * gprime * (v1 ** 2 + 8 * v5 ** 2)],
            [0, 0, -1 / 4 * g * gprime * (v1 ** 2 + 8 * v5 ** 2),
             1 / 4 * gprime ** 2 * (v1 ** 2 + 8 * v5 ** 2) + 77 / 6 * g ** 2 * T ** 2]
        ])

        # 计算特征值
        eigenvalues = np.linalg.eigvals(matrix)

        Pi_Z,Pi_gamma,Pi_W1,Pi_W2= eigenvalues

        # print("Pi_W1:", Pi_W1)
        # print("Pi_W2:", Pi_W2)
        # print("Pi_Z:", Pi_Z)
        # print("Pi_gamma:", Pi_gamma)

        return Pi_m12, Pi_m22, Pi_m32, Pi_m42, Pi_m52, Pi_m62, Pi_m72, Pi_m82, Pi_m92, Pi_m102,Pi_m112, Pi_m122, Pi_m132,Pi_W1,Pi_W2,Pi_Z,Pi_gamma

    def V_d1T(self, thermal, bosons, T):
        Pi_m12, Pi_m22, Pi_m32, Pi_m42, Pi_m52, Pi_m62, Pi_m72, Pi_m82, Pi_m92, Pi_m102,Pi_m112, Pi_m122, Pi_m132,Pi_W1,Pi_W2,Pi_Z,Pi_gamma = thermal
        m12, m22, m32, m42, m52, m62, m72, m82, m92, m102, m112,m122, m132,mW_Sq, mZ_Sq = bosons

        y = -(np.abs(Pi_m12) ** (3 / 2) - np.abs(m12) ** (3 / 2))
        y -= (np.abs(Pi_m22) ** (3 / 2) - np.abs(m22) ** (3 / 2))
        y -= (np.abs(Pi_m32) ** (3 / 2) - np.abs(m32) ** (3 / 2))
        y -= (np.abs(Pi_m42) ** (3 / 2) - np.abs(m42) ** (3 / 2))
        y -= (np.abs(Pi_m52) ** (3 / 2) - np.abs(m52)  ** (3 / 2))
        y -= (np.abs(Pi_m62) ** (3 / 2) - np.abs(m62) ** (3 / 2))
        y -= (np.abs(Pi_m72) ** (3 / 2) - np.abs(m72) ** (3 / 2))
        y -= (np.abs(Pi_m82) ** (3 / 2) - np.abs(m82)  ** (3 / 2))
        y -= (np.abs(Pi_m92) ** (3 / 2) - np.abs(m92) ** (3 / 2))
        y -= (np.abs(Pi_m102) ** (3 / 2) - np.abs(m102) ** (3 / 2))
        y -= (np.abs(Pi_m112) ** (3 / 2) - np.abs(m112)  ** (3 / 2))
        y -= (np.abs(Pi_m122) ** (3 / 2) - np.abs(m122) ** (3 / 2))
        y -= (np.abs(Pi_m132) ** (3 / 2) - np.abs(m132) ** (3 / 2))

        y -= (Pi_W1** (3 / 2) - mW_Sq ** (3 / 2))
        y -= (Pi_W2 ** (3 / 2) - mW_Sq ** (3 / 2))
        y -= (Pi_Z ** (3 / 2) - mZ_Sq ** (3 / 2))
        y -= (np.abs(Pi_gamma) ** (3 / 2) - 0)

        return y*T / (12 * np.pi)

    def Vtot(self, X, T, include_radiation=True):
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X, T)
        fermions = self.fermion_massSq(X)
        thermal = self.thermal_Pi_massSq(X, T)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.V1T(bosons, fermions, T, include_radiation)
        y += self.V_d1T(thermal, bosons, T)
        return y

m = model2()

p= m.Vtot([0,146],100)
print(p)

q= m.Vtot([146, 0],100)
print(q)

q= m.Vtot([176, 0],100)
print(q)

q= m.Vtot([50, 0],100)
print(q)

# Define a grid of points
v1_values = np.linspace(-200, 200, 400)  # Adjust the range and number of points as needed
v5_values = np.linspace(-100, 100, 400)  # Adjust the range and number of points as needed
results = np.zeros((len(v1_values), len(v5_values)))

# Evaluate the potential at each point in the grid
# for i, v1 in enumerate(v1_values,-200):
#     for j, v5 in enumerate(v5_values,-100):
for i, v1 in enumerate(v1_values):
    for j, v5 in enumerate(v5_values):
        m = model2()
        r = m.Vtot([v1, v5], 100)
        results[i, j] = np.real(r)

# Now 'results' contains the calculated values for each combination of v1 and v5
# print(results)

# Find the indices of the minimum value in the results matrix
min_index = np.unravel_index(np.argmin(results), results.shape)
min_value = results[min_index]

# # Create a heatmap with a different colormap ('jet') for better visibility
# plt.imshow(results, cmap='jet', origin='lower',   aspect='auto')

# Plot the contour lines
plt.figure(figsize=(8, 6))
contour_plot = plt.contour(v1_values, v5_values,results, levels=40, cmap='viridis')
plt.colorbar(contour_plot, label='Potential Energy')

# Mark the minimum point with a red dot
plt.scatter( v1_values[min_index[0]],v5_values[min_index[1]], color='red', label=f'Minimum Point\n({ {v1_values[min_index[0]]},v5_values[min_index[1]]}, {min_value})')

# Add labels and title
plt.xlabel('v1')
plt.ylabel('v5')
plt.title('Potential Energy Landscape with Contour Lines')

# Show the legend
plt.legend()

# Show the plot
plt.show()
