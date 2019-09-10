import pandas as pd
df_wine  = pd.read_csv('wine.csv')
X = df_wine.iloc[:,1:].values
y = df_wine.iloc[:,0].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.3,random_state = 1)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion = 'entropy',max_depth = None)
bag = BaggingClassifier(base_estimator = tree,n_estimators  = 50,
			max_samples  =1.0,
			max_features = 1.0,
			bootstrap = True,
			bootstrap_features = False,
			n_jobs = 1,
			random_state  =1)

from sklearn.metrics import accuracy_score
tree = tree.fit(X_train,y_train)
t_train_pred = tree.predict(X_train)
t_test_pred = tree.predict(X_test)
print('Decision tree train/test accuracy score:%.3f/%.3f'%(accuracy_score(t_train_pred,y_train),accuracy_score(t_test_pred,y_test)))

from sklearn.metrics import accuracy_score
bag = bag.fit(X_train,y_train)
b_train_pred = bag.predict(X_train)
b_test_pred = bag.predict(X_test)
print('Bag train/test accuracy score:%.3f/%.3f'%(accuracy_score(b_train_pred,y_train),accuracy_score(b_test_pred,y_test)))
