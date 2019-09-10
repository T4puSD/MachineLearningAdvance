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
#df_wine  = pd.read_csv('wine.csv')
print(df_wine.head())
X = df_wine.iloc[:,1:].values
y = df_wine['class'].values
