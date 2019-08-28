#test


import pathlib

import matplotlib.pyplot as pyplot
import numpy as np
import scipy as sp
import scipy.special
from scipy.special import gamma





################################################
'''ai=np.linspace(0,3,4)
aj=np.linspace(1,2,2)
def multi(ai:float, aj:float) -> float:
	return (ai*aj)

def add(new_state=None):


	return (new_state+1)

z=ai,aj = np.meshgrid(ai, aj)
b=multi(ai,aj)
c=add(b)
print(multi(ai,aj))
#print(b)
#print(c)
'''
array=np.random.uniform(0,10,7)
print(array)
g=np.argpartition(array,1)
print(array[g[:1]])

