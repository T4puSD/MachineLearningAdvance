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
X = df_wine[['alcohol','od','proline']].values
y = df_wine['class'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=1)


##
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion = 'entropy' , random_state = 1,max_depth = None)
adab = AdaBoostClassifier(base_estimator=tree, n_estimators = 500,
                        learning_rate=0.1,
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
#from sklearn.metrics import accuracy_score
adab = adab.fit(X_train,y_train)
adab_train_pred = adab.predict(X_train)
adab_test_pred = adab.predict(X_test)
adab_train = accuracy_score(y_train,adab_train_pred)
adab_test = accuracy_score(y_test,adab_test_pred)
print('AdaBoost tain/test accuracy %.3f/%.3f'%(adab_train,adab_test))
