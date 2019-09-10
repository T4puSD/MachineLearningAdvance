from scipy import signal
import numpy as np
def conv2d(x,w,mode = 'same'):
    return signal.convolve2d(x,w,mode)
x = np.random.rand(5,5)
w = np.random.rand(3,3)

print(conv2d(x,w),)
print(conv2d(x,w,'valid'),)
print(conv2d(x,w,'full'))
