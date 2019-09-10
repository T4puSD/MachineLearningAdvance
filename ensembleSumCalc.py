#for 11 classifier , 0.25 error
from scipy.misc import comb #combination
import math
def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier/2.))
    probs = [comb(n_classifier,k) * error**k * (1-error)**(n_classifier-k)
            for k in range(k_start,n_classifier+1)]
    return sum(probs)
n = int(input('ENter NUmber of classifiers'))
e = float(input('ENter Average base error of classifiers'))
s = ensemble_error(n,e)
print(s)

import numpy as np 
import matplotlib.pyplot as plt
error_range = np.arange(0.0,1.01,0.01)
ens_error = [ensemble_error(n,error=error) for error in error_range]
plt.plot(error_range,ens_error,label = 'Ens_errrro')
plt.plot(error_range,error_range,label = 'base_errro',linestyle = '--')
plt.xlabel('base error')
plt.ylabel('base/ens errro')
plt.legend(loc = 'upper left')
plt.grid(alpha = 0.5)
plt.show()
