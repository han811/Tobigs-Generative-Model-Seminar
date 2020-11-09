import os
import glob
import sys
import gzip

import numpy as np

def mnist():
    image_size = 28
    file_names = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    nums_list = [60000,60000,10000,10000]
    data_size = [28,1,28,1]
    return_data = [0,0,0,0]
    current_dir = os.getcwd()
    mnist_path = os.path.join(current_dir+'/data/mnist')
    
    for n, idx in enumerate(file_names):
        num_images = nums_list[n]
        image_size = data_size[n]
        if not os.path.isfile(os.path.join(mnist_path,idx)):
            print(f'there is no file name {idx} in data folder!')
            return None
        f = gzip.open(os.path.join(mnist_path,idx),'r')
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if n%2==0:
            return_data[n] = data.reshape(num_images, image_size, image_size, 1)
        else:
            return_data[n] = data.reshape(num_images)
    
    return return_data[0],return_data[1],return_data[2],return_data[3]
    
if __name__ == "__main__":
    train_data, train_label, test_data, test_label = mnist()
    print('train_data size :',train_data.shape)
    print('train_label size :',train_label.shape)
    print('test_data size :',test_data.shape)
    print('test_label size :',test_label.shape)
    