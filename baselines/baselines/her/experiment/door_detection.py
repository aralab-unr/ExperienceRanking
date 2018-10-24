import mujoco_py
import gym
import imageio
import numpy as np
import os
import time
import sys

from keras import models
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
import cv2

def test():
    env = gym.make("FetchReach-v1")
    ob = env.reset()

    count = 0
    N_sample = 20

    # Load trained CNN
    model = models.Sequential()
    model.add(Conv2D(24, kernel_size=20, strides=2, input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(7, 7), strides=2))
    model.add(Conv2D(48, kernel_size=15, strides=2))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=2))
    model.add(Conv2D(96, kernel_size=10, strides=2))
    model.add(Activation('relu'))
    model.add(Conv2D(2, kernel_size=1, strides=1))
    model.add(Activation('softmax'))
    model.add(Flatten())
    model.load_weights('/home/hainguyen/model_save.hdf5')

    action = [0,0,0,0]

    for step in range(0,N_sample):
        ob, reward, done, info = env.step(action)
        
        # Collect image from simulator
        img = env.render()

        # Collect image and resize
        image = cv2.resize(img, (256,256))
        image = np.asarray(image) 
        image = image.astype('float32')
        image /= 255.0

        # Use the trained CNN to predict whether the door is present
        prediction = model.predict(image.reshape(-1,256,256,3))
        if (prediction[0][0] > 0.5):
            door_present = 1
            count += 1
        else:
            door_present = 0

    door_present = (count > 0.85*N_sample)
    # print(door_present)
    # Output result
    # print("Door" if door_present else "Not a door")

    return door_present        

if __name__== '__main__':
    door_present = test()
    if (door_present):
        open('1.txt', 'w')
    else:
        open('0.txt', 'w')



