import numpy as np
x = [1,3,6,2,7,9,5]
w = [0,1,2]
def conv1d(x,w,mode='same'):
    
    print(np.convolve(x,w,mode = mode))
    print(str(np.convolve(x,w,mode=mode))[1:-1])

    #print(np.convolve(x,w,mode = 'valid'))

    #print(np.convolve(x,w,mode = 'full'))
conv1d(x,w)
conv1d(x,w,'valid')
conv1d(x,w,'full')
