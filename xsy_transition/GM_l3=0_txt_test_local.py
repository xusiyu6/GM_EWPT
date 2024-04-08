import numpy as np
from cosmoTransitions import generic_potential
import random

# 改变输出位置。导入了sys模块。
import sys

file = open('result1.txt', 'a+')
# a+是每次输出直接累加，w+是覆盖上一次的结果，输出新的结果。
# 创建了一个名为file的文件对象，打开名为result.txt的文件以进行附加写入（a+模式）。
sys.stdout = file
# 输出到result.txt。
# 接下来，它将标准输出sys.stdout重定向到这个打开的文件对象上，这意味着所有后续的标准输出都将被写入到result.txt文件中。
# sys.stdout代表“标准输出”的属性。通过将这个属性的值修改为某个文件对象，可以将本来要打印到屏幕上的内容写入文件。
# Python中sys 模块中的一个方法是stdout ，它使用其参数直接显示在控制台窗口上。
# 这段代码的主要功能是将程序的标准输出重定向到指定的文件result.txt中，以便将输出结果写入到文件中而不是控制台。

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
        self.M3 = (self.M5 ** 2 + 2 * self.M1 ** 2) / 3
        self.lam1 = self.MH ** 2 / self.v ** 2
        self.lam5 = 2 * (self.M5 ** 2 - self.M1 ** 2) / self.v ** 2
        self.mu22 = -4.0 * self.lam1 * self.v ** 2
        self.mu32 = (self.M1 ** 2 + 2 * self.M5 ** 2 - 4 * self.lam2 * self.v ** 2) / 2

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

def generate_random_number(range_start, range_end):
    """生成指定范围内的单个随机浮点数。

    Args:
        range_start (float): 随机数生成的起始值。
        range_end (float): 随机数生成的终止值。

    Returns:
        float: 生成的随机浮点数。
    """
    return random.uniform(range_start, range_end)


def eqtest(M1, M5, lam2, lam4):
    # class类里只能放函数，eqtest是和循环语句搭配使用的（因为不通过就进入下一次循环），class类里不能塞一个循环把所有的函数放这个循环里
    g = 2.0 * C_MW / C_VEV
    gp = np.sqrt(4.0 * (C_MZ ** 2 - C_MW ** 2) / C_VEV ** 2)
    yt = np.sqrt(2) * C_MT / C_VEV
    v = C_VEV
    MH = C_MH
    lam1 = MH ** 2 / v ** 2
    lam5 = 2 * (M5 ** 2 - M1 ** 2) / v ** 2
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
        for zeta in np.arange(1 / 3, 1, 0.001):
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


def makePlots(M1, M5, lam2, lam4, m=None):
    if m is None:
        m = GM_Z2_15(M1, M5, lam2, lam4)
        m.calcTcTrans()


nn = 1
while nn <= 2000000:

    M1_totest = generate_random_number(0, 3000)
    M5_totest = generate_random_number(0, 3000)
    lam2_totest = generate_random_number(-10, 10)
    lam4_totest = generate_random_number(0, 10)
    # 待测试的M1_totest, M5_totest, lam2_totest, lam4_totest已经生成，为了区分送到GM_Z2_15的参数，加了后缀_totest
    nn += 1

    if eqtest(M1_totest, M5_totest, lam2_totest, lam4_totest) == 0:  # 测试M1, M5, lam2, lam4
        continue  # 测试未通过则重新取随机数
    else:  # 通过则打印4个随机数并调用makePlots
        print("M1=", M1_totest, "M5=", M5_totest, "lam2=", lam2_totest, "lam4=", lam4_totest)
        makePlots(M1_totest, M5_totest, lam2_totest, lam4_totest)
        # 将通过测试的M1_totest, M5_totest, lam2_totest, lam4_totest送给makePlots

file.close()
# 关闭result.txt
