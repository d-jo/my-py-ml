
import minpy.numpy as np
from minpy.context import cpu, gpu
import minpy.numpy.random as random
from mnist import MNIST

mndata = MNIST("./data")
images, labels = mndata.load_training()



# !gpu safe, copies and normalizes data to training batch
def prepare_batch(input_batch,batch_size,offset=0):
    for i in range(offset, offset + batch_size):
        input_batch[i] = np.divide(np.array(images[i]), 255)

# !gpu safe
def prepare_nodes_array(count,initialization=0):
    # weight, bias, metabolism
    return np.full((count,3))

def prepare_model():
        layers = []
        

batch_size = 100
data_size = 28*28
print(data_size)
input_batch = np.zeros((batch_size,data_size))

prepare_batch(input_batch, 100, 0)
prepare_model()


