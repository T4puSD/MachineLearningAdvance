import os
import struct
import numpy as np

def load_mnist(path,kind = 'train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'%kind)
    images_path= os.path.join(path,'%s-images-idx3-ubyte.gz'%kind)

    with open(labels_path,'rb') as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols =struct.unpack('>IIII',imgpath.read(16))
        imges = np.fromfile(imgpath,dtype=np.uint8)
        imges = imges.reshape(len(labels),784)
        images = ((imges/255.)-5.)*2
    return images,labels

X_train,y_train=load_mnist('./mnist/',kind = 'train')
print('Rows:{}, columns:{}'.format(X_train.shape[0],X_train.shape[1]))
X_test,y_test = load_mnist('./mnist/',kind='t10k')
print('Rows:{}, columns:{}'.format(X_test.shape[0],X_test.shape[1]))
