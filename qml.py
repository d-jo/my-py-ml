
import minpy.numpy as np
from minpy.context import cpu, gpu
import minpy.numpy.random as random
from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *
import _pickle as cPickle
import gzip
from mnist import MNIST


mndata = MNIST("./data")
images, labels = mndata.load_training()
bits = 4



