
import minpy.numpy as np
from minpy.context import cpu, gpu
import minpy.numpy.random as random
from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *


qvm = api.QVMConnection()

def make_setup(bits):
    setup = Program()
    for i in range(0, bits):
        setup.inst(I(i))

    setup.inst(X(0))
    return setup


def make_observation(bits):
    meas = Program()
    for i in range(0, bits):
        meas.inst(MEASURE(i, [bits-i-1]))
    return meas

#def grover_steps()

bits = 4
p = make_setup(bits) + make_observation(bits)
wavefunction = qvm.wavefunction(p)
print(wavefunction)
print(qvm.run(p, list(range(0, bits))))


