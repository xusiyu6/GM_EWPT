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

        # scalar
        A =4*self.l1*v1**2
        B =4*v5**2*(self.l3+3*self.l4)
        C = v1*v5*(self.l5-2*self.l2)
        mh_Sq = A + B - np.sqrt((A - B) ** 2 + 12 * C ** 2)
        mH_Sq = A + B + np.sqrt((A - B) ** 2 + 12 * C ** 2)
        m3_Sq = 1/2*self.l5*(v1**2+8*v5**2)
        m5_Sq = 3/2*self.l5*v1**2+8*self.l3*v5**2

        # W boson
        mW_Sq = g ** 2 * np.abs(v1**2 + 8 * v5**2) / 4

        # Z boson
        mZ_Sq= (g ** 2+ gprime**2) * np.abs(v1**2 + 8 * v5**2) / 4

        # Without Goldstone Bosons
        dof = np.array([1, 1, 3, 5, 6, 3])
        c = np.array([1.5, 1.5, 1.5, 1.5, 5/6, 5/6])
        #MSq = np.array([mh_Sq, mH_Sq, m3_Sq, m5_Sq, mW_Sq, mZ_Sq])
        #MSq = np.rollaxis(MSq, 0, len(MSq.shape))

        return np.array([mh_Sq, mH_Sq, m3_Sq, m5_Sq, mW_Sq, mZ_Sq]), dof, c

    def fermion_massSq(self, X):

        X = np.array(X)
        v1, v5 = X[..., 0], X[..., 1]

        # Top quark
        mt_Sq = self.yt ** 2 * v1**2 / 2

        #MSq = np.array([mt_Sq])
        #MSq = np.rollaxis(MSq, 0, len(MSq.shape))
        dof = np.array([12])
        return mt_Sq,  dof

    def V1(self, bosons, fermions):
        mh_Sq, mH_Sq, m3_Sq, m5_Sq, mW_Sq, mZ_Sq = bosons[0]

        y = np.sum(mh_Sq * mh_Sq * (np.log(np.abs(mh_Sq / self.renormScaleSq) + 1e-100) - 1.5), axis=-1)
        y += np.sum(mH_Sq * mH_Sq * (np.log(np.abs(mH_Sq / self.renormScaleSq) + 1e-100) - 1.5), axis=-1)
        y += np.sum(3 * m3_Sq * m3_Sq * (np.log(np.abs(m3_Sq / self.renormScaleSq) + 1e-100) - 1.5), axis=-1)
        y += np.sum(5 * m5_Sq * m5_Sq * (np.log(np.abs(m5_Sq / self.renormScaleSq) + 1e-100) - 1.5), axis=-1)
        y += np.sum(6 * mW_Sq * mW_Sq * (np.log(np.abs(mW_Sq / self.renormScaleSq) + 1e-100) - 5 / 6), axis=-1)
        y += np.sum(3 * mZ_Sq * mZ_Sq * (np.log(np.abs(mZ_Sq / self.renormScaleSq) + 1e-100) - 5 / 6), axis=-1)

        mt_Sq, dof = fermions
        y -= 12 * np.sum(mt_Sq * mt_Sq * (np.log(np.abs(mt_Sq / self.renormScaleSq) + 1e-100) - 1.5), axis=-1)

        return y / (64 * np.pi ** 2)

    def V1T(self, bosons, fermions, T, include_radiation=True):
        # This does not need to be overridden.
        T2 = (T * T)+ 1e-100
        # the 1e-100 is to avoid divide by zero errors
        T4 = T * T * T * T
        mh_Sq, mH_Sq, m3_Sq, m5_Sq, mW_Sq, mZ_Sq = bosons[0]
        y = np.sum(Jb(mh_Sq / T2), axis=-1)
        y += np.sum(Jb(mH_Sq / T2), axis=-1)
        y += np.sum(3 * Jb(m3_Sq / T2), axis=-1)
        y += np.sum(5 * Jb(m5_Sq / T2), axis=-1)
        y += np.sum(6 * Jb(mW_Sq / T2), axis=-1)
        y += np.sum(3 * Jb(mZ_Sq / T2), axis=-1)
        mt_Sq = fermions[0]
        y += np.sum(-12 * Jf(mt_Sq / T2), axis=-1)
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
        Pi_Z = np.float64((g ** 2 + gprime ** 2) * (T ** 2 + G / 8) + 1 / 8 * np.sqrt((g ** 2 + gprime ** 2) ** 2 * (64 * T ** 4 + 16 * T ** 2 * G) + (g ** 2 + gprime ** 2) ** 2 * G ** 2))  # Z boson
        Pi_gamma = np.float64((g ** 2 + gprime ** 2) * (T ** 2 + G / 8) - 1 / 8 * np.sqrt((g ** 2 + gprime ** 2) ** 2 * (64 * T ** 4 + 16 * T ** 2 * G) + (g ** 2 + gprime ** 2) ** 2 * G ** 2))  # photon
        # print(G)
        # print(g)
        # print(gprime)
        # print(Pi_Z)
        # print(Pi_gamma)
        # Z=(0.629**2+0.345**2)* (100 ** 2 + 90000 / 8) + 1 / 8 * np.sqrt((0.629 ** 2 + 0.345 ** 2) ** 2 * (64 * 100 ** 4 + 16 * 100 ** 2 * 90000) + (0.629 ** 2 + 0.345 ** 2) ** 2 * 90000 ** 2)
        # print(Z)

        Pi_chi = T**2*F  # Goldstone boson
        Pi_3 = 1/2*self.l5*(v1**2+8*v5**2)+ 3/8*self.l5*T**2+ T**2*F
        Pi_5 = 3/2*self.l5*v1**2+8*self.l3*v5**2+ T**2/24*(3*self.l5+16*self.l3)+ T**2*F

        # With Goldstone Bosons
        Pi_dof = np.array([1, 1, 3, 5, 3, 2, 1, 1])
        #Pi_MSq = np.array([Pi_h, Pi_H, Pi_3, Pi_5, Pi_chi,Pi_W,Pi_Z,Pi_gamma])8++
        return Pi_h, Pi_H, Pi_3, Pi_5, Pi_chi,Pi_W,Pi_Z,Pi_gamma,Pi_dof

    def V_d1T(self, thermal, bosons, T):
        Pi_h, Pi_H, Pi_3, Pi_5, Pi_chi, Pi_W, Pi_Z, Pi_gamma, Pi_dof = thermal
        mh_Sq, mH_Sq, m3_Sq, m5_Sq, mW_Sq, mZ_Sq = bosons[0]

        y = -(np.abs(Pi_h) ** (3 / 2) - np.abs(mh_Sq) ** (3 / 2))
        y -= ( Pi_H ** (3 / 2) - mH_Sq ** (3 / 2))
        y -= 3 * (np.abs(Pi_3) ** (3 / 2) - np.abs(m3_Sq) ** (3 / 2))
        y -= 5 * (np.abs(Pi_5) ** (3 / 2) - np.abs(m5_Sq) ** (3 / 2))
        y -= 3 *(Pi_chi ** (3 / 2) - 0)
        y -= 2 * (Pi_W ** (3 / 2) - mW_Sq ** (3 / 2))
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

p= m.Vtot([100, 100],100)
print(p)

q= m.Vtot([146, 0],100)
print(q)

v1_values = np.linspace(-200, 200, num=200)  # Adjust the range and number of points as needed
v5_values = np.linspace(-200, 200, num=200)  # Adjust the range and number of points as needed

results = np.zeros((len(v1_values), len(v5_values)))

for i, v1 in enumerate(v1_values):
    for j, v5 in enumerate(v5_values):
        m = model2()
        # m.init(v1=v1)
        r = m.Vtot([v1, v5], 100)
        results[i, j] = r

# Now 'results' contains the calculated values for each combination of v1 and v5
# You can analyze or visualize the results as needed
print(results)

# Find the indices of the minimum value in the results matrix
min_index = np.unravel_index(np.argmin(results), results.shape)
min_value = results[min_index]

# Create a heatmap with a different colormap ('jet') for better visibility
plt.imshow(results, cmap='jet', origin='lower', extent=[-200, 200, -200, 200], aspect='auto')

# Add colorbar
plt.colorbar(label='Potential Energy')

# # Add contour lines for better visualization
# contour_levels = np.linspace(np.min(results), np.max(results), 20)  # Adjust the number of contour levels as needed
# contour = plt.contour(results, levels=contour_levels, colors='white', extent=[-300, 300, -300, 300], origin='lower')
#
# # Label contour lines
# plt.clabel(contour, contour_levels, inline=True, fmt='%1.1f', colors='white')

# Mark the minimum point with a red dot
plt.scatter(v5_values[min_index[1]], v1_values[min_index[0]], color='red', label=f'Minimum Point\n({v5_values[min_index[1]]}, {v1_values[min_index[0]]}, {min_value})')

# Add labels and title
plt.xlabel('v5')
plt.ylabel('v1')
plt.title('Potential Energy Landscape with Contour Lines')

# Show the legend
plt.legend()

# Show the plot
plt.show()

# # 生成 x 和 y 的值
# x_vals = np.linspace(-100,100, 100)
# y_vals = np.linspace(-100, 100, 100)
#
# # 创建网格点
# x1, y1 = np.meshgrid(x_vals, y_vals)
#
# # 注意这里修改了输入，将两个数组合并成一个 (2, 100, 100) 的数组
# z = m.Vtot([x1, y1], 100)
#
# # 画图
# plt.contourf(x1, y1, z, cmap='viridis')  # 使用contourf函数填充等高线图，并使用viridis颜色映射
# plt.colorbar()  # 添加颜色条
#
# # 添加标签和标题
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Intensity of V(x, y)')
#
# # 显示图形
# plt.show()

# print(v)

# v0=m.V0([100,100])
# print(v0)
#
# boson=m.boson_massSq([100,100],100)
# print(boson)
#
# fermion=m.fermion_massSq([100,100])
# print(fermion)
#
# v1=m.V1([[  1264.86460013, 442735.13539987, 225000.        ,  35000.        ,
#          8920.73945779,  11604.81775675],[1, 1, 3, 5, 6, 3],[1.5       , 1.5       , 1.5       , 1.5       , 0.83333333,
#        0.83333333]],[5000.0, [12]])
# print(v1)
#
# v1t=m.V1T([[  1264.86460013, 442735.13539987, 225000.        ,  35000.        ,
#          8920.73945779,  11604.81775675],[1, 1, 3, 5, 6, 3],[1.5       , 1.5       , 1.5       , 1.5       , 0.83333333,
#        0.83333333]],[5000.0, [12]],100)
# print(v1t)
#
# thermal=m.thermal_Pi_massSq([100.,100.],100.)
# print(thermal)
#
# vd1t=m.V_d1T([189222.35962903383, 320961.1584360334, 258341.75903253362, 52508.42569920028, 14591.759032533615, 16850.285642499315, 21920.21131831161, 1.8189894035458565e-12, [1, 1, 3, 5, 3, 2, 1, 1]],[[  1264.86460013, 442735.13539987, 225000.        ,  35000.        ,
#          8920.73945779,  11604.81775675],[1, 1, 3, 5, 6, 3],[1.5       , 1.5       , 1.5       , 1.5       , 0.83333333,
#        0.83333333]],100.)
# print(vd1t)
