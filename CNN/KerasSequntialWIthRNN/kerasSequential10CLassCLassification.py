#wap to sequential model for 10 class classification
#seqential model with tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

#for a single input model with 10 classes(categorical Classification)
model = Sequential()
model.add(Dense(32,activation = 'relu',input_dim=100))
model.add(Dense(10,activation = 'softmax'))
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#generate dummy data
import numpy as np
data = np.random.random((1000,100))
labels = np.random.randint(10,size = (1000,1))
#convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels,num_classes = 10)
#Train the model,interating on the data in batches of 32 samples
model.fit(data,one_hot_labels, epochs = 10,batch_size = 32)
score = model.evaluate (data,one_hot_labels,batch_size = 32)
print(model.metrics_names)
print(score)
