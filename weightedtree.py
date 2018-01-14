
import time
import minpy.numpy.random as random
from minpy.context import cpu, gpu
from anytree import AnyNode, RenderTree
import minpy.numpy as np


ORDER = 3


inputLayers = []
outputLayers = []

Nodes = {}


def nodeRange(depth, height):
    with(gpu()):
        arr = [None]*height
        for i in range(0, height):
            arr[i] = AnyNode(parent=None, height=i, depth=depth, weight=np.random.rand, bias=np.random.rand, parents=[], output=0.0, metab=np.random.rand)
        Nodes[depth] = arr



def fullR2LLink(depth1, depth2, gpun=0):
    with(gpu(gpun)):
        for node in Nodes[depth1]:
            for node2 in Nodes[depth2]:
                node2.parents.append(node)
                

def fullL2RLink(depth1, depth2, gpun=0):
    with(gpu(gpun)):
        for node in Nodes[depth2]:
            for node2 in Nodes[depth1]:
                node2.parents.append(node)


def fullLink(depth1, depth2):
    fullL2RLink(depth1, depth2)
    fullR2LLink(depth1, depth2)

def addInput(d):
    inputLayers.append(d)

def addOutput(d):
    outputLayers.append(d)

def printOutput(gpun=0):
    with(gpu(gpun)):
        for num in outputLayers:
            for node in Nodes[num]:
                print(node.output)

def calculate(node):
    # step 1 build input vector
    i = len(node.parents)
    inputVectors = np.array([None]*i)
    i = 0
    for n in node.parents:
        inputVectors[i] = n.output
        i = i + 1
        
    # step 2 calculate output based on input average (maybe other in future?)
    in_mean = np.mean(inputVectors)
    print(type(in_mean))
    print(inputVectors)
    print(in_mean)
    out = np.multiply(in_mean, node.weight)
    return out

# must be same shape as inputs, condensed l2r
def prop(startingInput):
    
    for i in inputLayers:
        for n in Nodes[i]:
            n.output = startingInput[n.height]

    for i in range(1, len(Nodes)):
        for node in Nodes[i]:
            node.output = calculate(node)


nodeRange(0, 4)
nodeRange(1, 4)
nodeRange(2, 2)
addInput(0)
addOutput(2)
fullL2RLink(0,1)
fullL2RLink(1,2)
printOutput()

prop([1,1,1,1])

printOutput()


