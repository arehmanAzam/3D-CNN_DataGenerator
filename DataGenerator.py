import numpy as np
from tensorflow.python import keras as keras
from matplotlib.image import imread
import matplotlib.pyplot as plt
from augmentation import *
import random
import time

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True,real_batchsize_custom=2,frames_chunk=18):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.real_batchsize_custom=real_batchsize_custom
		self.frames_chunk=frames_chunk
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X= np.empty((self.real_batchsize_custom, *self.dim, self.n_channels))
        # Find list of IDs
        y = np.empty((self.real_batchsize_custom,self.n_classes), dtype=float)
		
		initial_point=0
		final_point=self.frames_chunk-1
        for x in range(self.real_batchsize_custom):
			
			indexes_orig=indexes[initial_point:final_point]
            list_IDs_temp = [self.list_IDs[k] for k in indexes_orig]
        # for i, ID in enumerate(list_IDs_temp):
        #     # Store sample
        #
        #     # Convert to a numpy array and return it.
        #     class_id=ID.split('/')[5]
        #     # if("running"==class_id):
        #     #     print('mismatch 1')

        # Generate data
			
            X_in, y_in = self.__data_generation(list_IDs_temp)
            X[x]=X_in
            y[x]=y_in[0]
			initial_point=final_point+1
			final_point=final_point+(self.frames_chunk-1)
			
        return X,y
        #return [X, X, X],[y,y,y]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    batch=1
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch, *self.dim, self.n_channels))
        y = np.empty((self.batch), dtype=int)
        y=[0]*self.batch
        # Generate data
        #random num[1-5]
        random.seed(time.time())
        idChoice = random.randint(1, 2)


        random.seed(time.time())
        id = random.randint(1, 9)
        # if idChoice == 1:
        #     if id ==1 :
        #         print("\nBrightness or Contrast Augmented")
        #     elif id ==2 :
        #         print("\nFlip Augmented")
        #     elif id ==3 :
        #         print("\nScale Augmented")
        #     elif id ==4 :
        #         print("\nRotate Augmented")
            # id =9
            # print(id)

        # if idChoice == 2:
        #     print("\nOriginal Data:")

        random.seed(time.time())
        idRand = random.randint(1, 6)

        actual_class=list_IDs_temp[0].split('/')[5]
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image = imread(ID)
            # plt.imshow(image)

            if idChoice == 1:
                #conditions [1-4]
                if id == 1:
                    # Call Brightness


                    if idRand == 1:
                        image = augmentation.apply_brightness_contrast(image, 0, 0)
                    if idRand == 2:
                        image = augmentation.apply_brightness_contrast(image, -127, 0)
                    if idRand == 3:
                        image = augmentation.apply_brightness_contrast(image, 0, -64)
                    if idRand == 4:
                        image = augmentation.apply_brightness_contrast(image, 127, 0)
                    if idRand == 5:
                        image = augmentation.apply_brightness_contrast(image, 0, 64)
                    if idRand == 6:
                        image = augmentation.apply_brightness_contrast(image, 64, 64)



                if id == 2:
                    # Call Flip
                    image = augmentation.flip(image)
                    # plt.imshow(image)
                    # print("flip image")

                if id == 3:
                    image = augmentation.scaleImage(image)
                    # plt.imshow(image)
                    # print("crop image")

                if id == 4:
                    image = augmentation.rotate(image)
                    # plt.imshow(image)
                    # print("Rotated image")

            # Convert to a numpy array and return it.
            class_id=ID.split('/')[5]
            # if(actual_class!=class_id):
            #     print('mismatch 2')

            X[0,i] =  np.asarray(image)

            #X[i,] =  np.asarray(image)
            # Store class
            for label in self.labels:
                if label.get('path')==ID:
                    y[0] = label.get('class')
        #if(4)
        # if id == 8:
        #     # Call Scaling
        #     random.seed(time.time())
        #     id1 = random.randint(6,10)
        #     X = augmentation.central_scale_images(X, [id1/10])
        #     plt.imshow(X[0])
        #     cv2.imshow('',X[0])
        #     cv2.waitKey(1)
        #     print("call Scaling")
        #X=scaling(X)
        # X= X.reshape((3,18,400,256))
        return X[0], keras.utils.to_categorical(y, num_classes=self.n_classes)