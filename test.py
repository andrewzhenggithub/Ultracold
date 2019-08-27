#test


import pathlib

import matplotlib.pyplot as pyplot
import numpy as np
import scipy as sp
import scipy.special
from scipy.special import gamma





################################################
ai=np.linspace(0,3,4)
aj=np.linspace(0,2,3)
def multi(ai:float, aj:float) -> float:
	return (ai*aj)

def add(new_state=None):
	if new_state==None:
		return (55)
	else:
		return (new_state+1)

ai,aj = np.meshgrid(ai, aj)
b=multi(ai,aj)
c=add(b)

print(b)
print(c)