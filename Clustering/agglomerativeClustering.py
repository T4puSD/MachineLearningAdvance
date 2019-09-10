import pandas as pd
import numpy as np
np.random.seed(123)
variables = ['x','y','z']
labels = ['ID1','ID2','ID3','ID4','ID5']
X = np.random.random([5,3])*10
df = pd.DataFrame(X,columns = variables,index = labels)
print(df)
##

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 3,affinity = 'euclidean',
                              linkage = 'complete')
labels = ac.fit_predict(X)
print('Cluster Labels:%s'%labels)
