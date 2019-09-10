import numpy as np
#generating dummy traninning data(Embedding method)
x_train= np.random.random((1000,256))
y_train = np.random.random((1000,1))
#test
x_test = np.random.random((100,256))
y_test = np.random.random((100,1))

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Embedding
from keras.layers import LSTM
max_features = 1024
model = Sequential()
model.add(Embedding(max_features, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = 'rmsprop',metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 16,epochs = 10)
score = model.evaluate(x_test,y_test,batch_size = 16)
print(score)
