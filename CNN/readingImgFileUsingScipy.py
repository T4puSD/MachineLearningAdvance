from scipy import misc
import matplotlib.pyplot as plt
img = misc.imread('abc.jpg',mode = 'RGB')
print('imgae shape:',img.shape)
print('channel size:',img.shape[2])
print('image datatype:',img.dtype)
#misc.imshow(img)
plt.imshow(img)
plt.axis('off')
plt.show()


