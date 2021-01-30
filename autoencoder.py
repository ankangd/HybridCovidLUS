"""Autoencoder.ipynb

**
 * This file is part of Hybrid CNN-LSTM for COVID-19 Severity Score Prediction paper.
 *
 * Written by Ankan Ghosh Dastider and Farhan Sadik.
 *
 * Copyright (c) by the authors under Apache-2.0 License. Some rights reserved, see LICENSE.
 */
 
"""
#denoising with autoencoder + classification
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,concatenate,SeparableConv2D
from keras.models import Model
import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

densenet = DenseNet201(weights='imagenet', include_top=False)
#Build the model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#s = Lambda(lambda x: x / 255)(inputs)
s=inputs

############
# Encoding #
############

# Conv1 #
x_128 = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same')(inputs) #128,128,16
x = MaxPooling2D(pool_size = (2, 2), padding='same')(x_128)#64,64,16

# Conv2 #
x_64 = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same')(x)#64,64,8
x_32 = MaxPooling2D(pool_size = (2, 2), padding='same')(x_64)#32,32,8 

# Conv 3 #
x_32_1 = Conv2D( 8, (3, 3), activation='relu', padding='same')(x_32) #32,32,8
#x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

# Conv 4 #
#x = Conv2D( 8, (3, 3), activation='relu', padding='same')(x) #16
#x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
# Conv 5 #
#x = Conv2D( 8, (3, 3), activation='relu', padding='same')(x) #8
encoded_16 = MaxPooling2D(pool_size = (2, 2), padding='same')(x_32_1) #16,16,8
#conv 6
#x = Conv2D( 8, (3, 3), activation='relu', padding='same')(x) 
#encoded = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

############
# Decoding #
############

# DeConv1
y_16 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_16)#16,16,8
y_32 = UpSampling2D((2, 2))(y_16)#32,32,8

y_32=concatenate([x_32,y_32])#32,32,8

#temp_32=MaxPooling2D(pool_size = (2, 2), padding='same')(x_64) 
#y_32= 

# DeConv2
y_32= Conv2D(8, (3, 3), activation='relu', padding='same')(y_32)#32,32,8
y_64= UpSampling2D((2, 2))(y_32) #64,64,8
y_64=concatenate([x_64,y_64]) #64,64,8

# DeConv2
y_64 = Conv2D(16, (3, 3), activation='relu', padding='same')(y_64)#64,64,16
y_128= UpSampling2D((2, 2))(y_64)#128,128,16

y_128=concatenate([y_128,x_128])

# DeConv2
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)

# DeConv2
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)

# Deconv3
#x = Conv2D(16, (3, 3), activation='relu')(x)
#x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(y_128)#128,128,3

decoded=concatenate([decoded,inputs ])
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)

x=Conv2D(3, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(decoded)
