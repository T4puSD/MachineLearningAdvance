import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(123)
variables = ['x','y','z']
labels = ['ID1','ID2','ID3','ID4','ID5']
X = np.random.random([5,3])*10
df = pd.DataFrame(X,columns = variables,index = labels)
print(df)
##
from scipy.spatial.distance import pdist,squareform
row_dist=pd.DataFrame(squareform(pdist(df,metric='euclidean')),columns=labels,index=labels)
print(row_dist)
##
from scipy.cluster.hierarchy import linkage
row_clusters=linkage(df.values,method='complete',metric='euclidean')
df1=pd.DataFrame(row_clusters,
                 columns=['row label 1','row label 2','distance','no. of items in cluster'],
                 index=['cluster %d'%(i+1) for i in range(row_clusters.shape[0])])
print(df1)
##
from scipy.cluster.hierarchy import dendrogram
row_dendr=dendrogram(row_clusters,labels=labels)
plt.tight_layout()
plt.ylabel('Euclidean Distance')
plt.show()
