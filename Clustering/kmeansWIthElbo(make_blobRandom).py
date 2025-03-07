from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples = 150,n_features = 2,centers = 3,cluster_std = 0.5,
                 shuffle = True,random_state = 0)
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c  ='white',edgecolor = 'black',marker = 'o',s = 50)
plt.grid()
plt.show()
##clustering
from sklearn.cluster import KMeans
km = KMeans(n_clusters  =3,init = 'random',n_init = 10,max_iter = 300,
            tol = 1e-04,random_state = 0)
y_km = km.fit_predict(X)
##plotting
plt.figure()
plt.scatter(X[y_km == 0,0],X[y_km == 0,1],s = 50,c = 'lightgreen',
            marker = 'd',edgecolor = 'black',label = 'Cluster 1')
plt.scatter(X[y_km == 1,0],X[y_km==1,1],s = 50,c = 'orange',marker = 'o',
            edgecolor = 'black',label = 'Cluster 2')
plt.scatter(X[y_km == 2,0],X[y_km ==2,1],s = 50,c = 'blue',marker = '>',
            edgecolor ='black',label = 'Cluster 3')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s = 250,marker='*',
            color = 'red',edgecolor = 'black',label  = 'Centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

##elbo method to handle no of k in kmeans   
distortions = []
for i in range(1,11):
    km = KMeans(n_clusters = i,init = 'k-means++',n_init = 10,max_iter = 300,
                random_state = 0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1,11),distortions,marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortions')
plt.show()
