
import minpy.numpy as np
from minpy.context import cpu, gpu
import minpy.context as c
import minpy.numpy.random as random
import time
from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *


OPS = 5

arr = np.zeros((OPS,2,2))
print (arr)
