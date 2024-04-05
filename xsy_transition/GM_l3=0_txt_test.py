import numpy as np
from cosmoTransitions import generic_potential
import random

# 改变输出位置
import sys

file = open('result.txt', 'a+')
# a+是每次输出直接累加，w+是覆盖上一次的结果，输出新的结果
sys.stdout = file
# 输出到result.txt

C_VEV = 246.22
C_MW = 80.385
C_MZ = 91.1776
C_MT = 173.5
C_MB = 4.18
C_MH = 125.0
C_PI = 3.1415927


def generate_random_number(range_start, range_end):
    """生成指定范围内的单个随机浮点数。

    Args:
        range_start (float): 随机数生成的起始值。
        range_end (float): 随机数生成的终止值。

    Returns:
        float: 生成的随机浮点数。
    """
    return random.uniform(range_start, range_end)


def eqtest():
    eq1 = ((3 * g ** 2 + gp ** 2) / 2 + 2 * lam2 + 11 * lam4) / (
            (3 * g ** 2 + gp ** 2) / 16 + yt ** 2 / 4 + 0.25 * MH ** 2 / v ** 2 + 1.5 * lam2)
    eq2 = (-3) * (M1 ** 2 + 2 * M5 ** 2 - 4 * lam2 * v ** 2) / MH ** 2
    eq3 = (6 * lam4 ** 0.5) / (0.5 * MH ** 2 / v ** 2) ** 0.5
    eq4 = (6 * (2 * lam2 * v ** 2 - 2 * (M5 ** 2 - M1 ** 2))) / MH ** 2

    def omega(zeta):
        B = (1.5 * (zeta - 1 / 3)) ** 0.5
        return (1 - B) / 6 - 2 ** 0.5 / 3 * ((1 - B) * (0.5 + B)) ** 0.5

    def zetatest1():
        zeta = 1 / 3
        return lam2 > omega(zeta) * lam5 - 2 * (lam1 * lam4) ** 0.5

    def zetatest2():
        for zeta in np.arange (1/3, 1, 0.001):
            if lam2 > omega(zeta) * lam5 - 2 * (lam1 * lam4) ** 0.5:
                return 0
        return 1
    if (
            eq1 < eq2 and eq2 < eq3 and eq3 < eq4 and

            ((6 * lam1 - 11 * lam4) ** 2 + 36 * lam2 ** 2) ** 0.5 + abs(6 * lam1 + 11 * lam4) < 4 * C_PI and
            ((2 * lam1 - 2 * lam4) ** 2 + lam5 ** 2) ** 0.5 + abs(2 * lam1 + 2 * lam4) < 4 * C_PI and
            abs(lam4) < C_PI and
            abs(lam2 - lam5) < 2 * C_PI and

            lam1 > 0 and
            lam4 > 0 and
            ((lam2 > (0.5 * lam5 - 2 * (lam1 * lam4) ** 0.5) and lam5 > 0) or (zetatest1() and lam5 < 0))

    ):
        return 1
    else:
        return 0


class GM_Z2_15(generic_potential.generic_potential):  # 创建一个类GM_Z2_15并继承已有的类generic_potential
    """
    The effective potential for Z2 symmetric GM model with v1 and v8=\sqrt{2}v5 (neglecting other background fields)
    """

    def init(self):  # 重写generic_potential里的方法init
        self.M5 = M5
        self.M1 = M1
        self.lam2 = lam2
        self.lam4 = lam4

        self.Ndim = Ndim
        self.renormScaleSq = renormScaleSq
        self.g = g
        self.gp = gp
        self.CW = CW
        self.SW = SW
        self.TW = TW
        self.yt = yt
        self.yb = yb
        self.v = v
        self.MH = MH
        self.M3 = M3
        self.lam1 = lam1
        self.lam5 = lam5
        self.mu22 = mu22
        self.mu32 = mu32

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
        y += 1 / 6 * self.mu22 * T ** 2 + 3 / 8 * self.mu32 * T ** 2 + self.lam1 * omg1 ** 2 * T ** 2 + 3 / 4 * self.lam2 * omg1 ** 2 * T ** 2
        y += self.lam2 * omg5 ** 2 * T ** 2 + 11 / 2 * self.lam4 * omg5 ** 2 * T ** 2
        y += 1 / 16 * self.g ** 2 * (omg1 ** 2 + 8 * omg5 ** 2) * T ** 2 + 1 / 32 * (self.g ** 2 + self.gp ** 2) * (
                omg1 ** 2 + 8 * omg5 ** 2) * T ** 2
        y += 1 / 8 * self.yt ** 2 * omg1 ** 2 * T ** 2
        return y

    def approxZeroTMin(self):  # 重写generic_potential里的方法approxZeroTMin
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        v = C_VEV
        return [np.array([v, 0])]


def makePlots(m=None):
    if m is None:
        m = GM_Z2_15()
        m.calcTcTrans()


nn = 1
while nn <= 20000000:
    M1 = generate_random_number(0, 3000)
    M5 = generate_random_number(0, 3000)
    lam2 = generate_random_number(-10, 10)
    lam4 = generate_random_number(0, 10)
    # M1, M5, lam2, lam4已经生成，均为全局变量
    # def set_physical_parameters中已直接访问全局变量M1, M5, lam2, lam4，故调用该方法时不需要再传入M1, M5, lam2, lam4
    nn += 1

    Ndim = 2
    renormScaleSq = C_VEV ** 2
    g = 2.0 * C_MW / C_VEV
    gp = np.sqrt(4.0 * (C_MZ ** 2 - C_MW ** 2) / C_VEV ** 2)
    CW = C_MW / C_MZ
    SW = np.sqrt(1.0 - CW ** 2)
    TW = SW / CW
    yt = np.sqrt(2) * C_MT / C_VEV
    yb = np.sqrt(2) * C_MB / C_VEV
    v = C_VEV
    MH = C_MH
    M3 = (M5 ** 2 + 2 * M1 ** 2) / 3
    lam1 = MH ** 2 / v ** 2
    lam5 = 2 * (M5 ** 2 - M1 ** 2) / v ** 2
    mu22 = -4.0 * lam1 * v ** 2
    mu32 = (M1 ** 2 + 2 * M5 ** 2 - 4 * lam2 * v ** 2) / 2

    if eqtest() == 0:
        continue

    print("M1=", M1, "M5=", M5, "lam2=", lam2, "lam4=", lam4)
    # makePlots 函数已经定义好，这里调用它
    makePlots()

file.close()
# 关闭result.txt
