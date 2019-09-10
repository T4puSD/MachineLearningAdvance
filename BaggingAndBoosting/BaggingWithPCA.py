import pandas as pd
try:
    df_wine = pd.read_csv('wine.csv')
except FileNotFoundError:
    df_wine = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data')
    #print(df_wine.head())
    df_wine.columns = ['class','alcohol','malic acid','ash','alcalinity of ash','magnesium','total phenols',
                       'flavanoids','nonflavanoid phenols','proanthocyanins','color intensity','hue','od','proline']
    #print(df_wine.head())
    df_wine.to_csv('wine.csv',index = False)
    
print(df_wine.head())
#df_wine.to_csv('wine.csv')
#X = df_wine[['alcohol','od','hue']].values
X = df_wine.iloc[:,1:].values #iloc start from 0 indeces
y = df_wine['class'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=1)



##dimension reduction using pca
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(pca.components_)


##

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion = 'entropy' , random_state = 1,max_depth = None)
bag = BaggingClassifier(base_estimator=tree, n_estimators = 50,
                        max_samples = 1.0,
                        max_features = 1.0,
                        bootstrap = True,
                        bootstrap_features = False,
                        n_jobs = 1,
                        random_state = 1)

##
from sklearn.metrics import accuracy_score
tree = tree.fit(X_train,y_train)
t_train_pred = tree.predict(X_train)
t_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train,t_train_pred)
tree_test = accuracy_score(y_test,t_test_pred)
print('Decision tree tain/test accuracy %.3f/%.3f'%(tree_train,tree_test))

##
from sklearn.metrics import accuracy_score
bag = bag.fit(X_train,y_train)
b_train_pred = bag.predict(X_train)
b_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train,b_train_pred)
bag_test = accuracy_score(y_test,b_test_pred)
print('Bag tain/test accuracy %.3f/%.3f'%(bag_train,bag_test))
