from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential([Dense(32,input_shape = (784,)),Activation('relu'),
                    Dense(10),Activation('softmax'),])
'''or this mehtod is also work
model = Sequential()
model.add(Dense(32,input_dim=784))
modell.add(Activation('relu'))'''


