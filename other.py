
import matplotlib.pyplot as plt
import time
import minpy.numpy as np
from minpy.context import gpu


def calculate(fixedNum, itr):
    mem = np.zeros((len(itr),3))
    i = 0
    tstart = time.clock()
    with(gpu()):
        for item in itr.__iter__():
            mem[i] = (fixedNum % item)
            mem[i][1] = item
            mem[i][0] = i
            i += 1
        
    tend = time.clock()
    mem.asnumpy()
    return (mem, i, tstart, tend)

# copys the axis to the provided result array
# the resulting arr
def fixed_2daxis_slice(arr, resultarr, length, axis=0, customitr=None):
    index = 0
    if customitr is None:
        customitr = range(0, length)
    with(gpu(0)):
        for i in customitr.__iter__():
            resultarr[index] = arr[i][axis]
            index += 1


def statistics(data, length):
    xax = np.zeros(length)
    yax = np.zeros(length)
    fixed_2daxis_slice(data, xax, length, axis=1)
    fixed_2daxis_slice(data, yax, length, axis=2)
    print(xax)
    print(yax)
    xax = xax.asnumpy()
    yax = yax.asnumpy()
    plt.plot(xax, yax)
    
    plt.grid()
    plt.show()

nu = 20681 
nu = 18

data = calculate(nu, range(1,nu))
print(data)
statistics(data[0], data[1])

print ("time:", data[3] - data[2])


