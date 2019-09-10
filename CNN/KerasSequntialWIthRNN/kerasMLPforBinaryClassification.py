#Keras MultilayerPerceptron(MLP) for Binary Classicfication
import keras
from keras.layers import Dense,Activation,Dropout
from keras.models import Sequential
import numpy as np

#creting random data
x_train = np.random.random((1000,20))
y_train = np.random.randint(2,size =(1000,1))
x_test = np.random.random((100,20))
y_test = np.random.randint(2,size = (100,1))

model = Sequential()
model.add(Dense(10,activation='relu',input_dim = 20))
model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation = 'sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 128,epochs = 20)
model.evaluate(x_test,y_test,batch_size = 128)
