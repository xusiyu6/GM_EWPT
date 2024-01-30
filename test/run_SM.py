import sys
from os import path

curdir=path.dirname(path.abspath(__file__))
sys.path.append(curdir+'/..')

from Models.SM import SM


mod = SM(50.0) # Using SM with Higgs mass at 50 GeV

# print(mod.Vtot([50,100],0))
phase = mod.getPhases()

print(phase)
