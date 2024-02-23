import sys
from os import path

curdir=path.dirname(path.abspath(__file__))
sys.path.append(curdir+'/..')

from models.GM_Z2_15 import GM_Z2_15


mod = GM_Z2_15(0.2,0.15,180,200,220)

# print(mod.Vtot([50,100],0))
phase = mod.getPhases()

print(phase)
