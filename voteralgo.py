from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X,y = iris.data[50:,[1,2]],iris.target[50:]
#X,y = iris.data,iris.target
le=LabelEncoder()
y = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5,
                                                 random_state =1)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
clf1 = LogisticRegression(penalty = 'l2', C = 0.001,random_state = 1)#L2 regularization
clf2 = DecisionTreeClassifier(max_depth = 1,criterion = 'entropy', random_state = 0)
clf3 = KNeighborsClassifier(n_neighbors=1, p = 2, metric = 'minkowski')
pipe1 = Pipeline ([['sc', StandardScaler()],['clf',clf1]])
pipe3 = Pipeline([['sc',StandardScaler()],['clf',clf3]])
clf_labels = ['LR','DT','KNN']
clf_name = [pipe1,clf2,pipe3]

for clf,label in zip(clf_name,clf_labels):
    scores = cross_val_score(estimator = clf,X = X_train,y = y_train,cv = 10,
                             scoring = 'roc_auc') #roc_auc only for 2 class problem
    print('ROC AUC: %.2f (+/- %.2f) [%s] '%(scores.mean(),scores.std(),label))
#votting

from sklearn.ensemble import VotingClassifier
mv_clf = VotingClassifier(estimators = [('lr',pipe1),('dt',clf2),('knn',pipe3)],
                          voting = 'hard', weights = [1,1,5])
mv_clf = mv_clf.fit(X,y)
s = mv_clf.score(X_test,y_test)
print(s)

print(mv_clf.predict(X))
print(mv_clf.predict([[1,1]]))
print(mv_clf.predict([[1,2],[2,1]]))
