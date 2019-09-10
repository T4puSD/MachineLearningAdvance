from sklearn.datasets import make_moons
X,y=make_moons(n_samples=200,noise=0.05,random_state=0)
import matplotlib.pyplot as plt
plt.figure(1)
plt.scatter(X[:,0],X[:,1])
#plt.show()
##
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=0.2,min_samples=5,metric='euclidean')
y_db=db.fit_predict(X)
plt.figure()
plt.scatter(X[y_db==0,0],X[y_db==0,1],c='lightblue',edgecolor='black',marker='o',s=40,label='cluster 1')
plt.scatter(X[y_db==1,0],X[y_db==1,1],c='red',edgecolor='black',marker='s',s=40,label='cluster 1')
plt.title('DBSCAN Clustering')
plt.legend()
plt.show()
