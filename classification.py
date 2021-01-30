# -*- coding: utf-8 -*-
"""Classification.ipynb
**
 * This file is part of Hybrid CNN-LSTM for COVID-19 Severity Score Prediction paper.
 *
 * Written by Ankan Ghosh Dastider and Farhan Sadik.
 *
 * Copyright (c) by the authors under Apache-2.0 License. Some rights reserved, see LICENSE.
 */
"""
from keras.applications import ResNet152V2,DenseNet201,NASNetMobile,Xception
#from keras.applications import DenseNet121
from keras.layers.merge import concatenate
import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
 
#Build the model
#Branch 1
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

#s = Lambda(lambda x: x / 255)(inputs)
s=inputs

#Make 3 positional Arguments
c1 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
p1=  MaxPool2D(pool_size=(2,2))(c1)
p1=  Dropout(0.2)(p1)

mid1 = p1

c1_1= Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1) #64,128

c2=  Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
p2=   MaxPool2D(pool_size=(2,2))(c2)
#p2= Dropout(0.5)(p2)

mid2 = p2
mid2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(mid2)
mid2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(mid2)
mid2 =  Dropout(0.2)(mid2)
P1_R = MaxPool2D(pool_size=(2,2))(mid2)

              
R1=concatenate([c1_1,p2])
R1.shape

#R1=Dropout(0.5)(R1) #Extra
            
C1_R=Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(R1)
mid1 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(mid1)
mid1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(mid1)
mid1 =  Dropout(0.2)(mid1)
            
mid1_1 = concatenate([C1_R,mid1])

C11_R=Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(mid1_1)
C11_R=Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(C11_R)
C11_R = MaxPool2D(pool_size=(2,2))(C11_R)
C11_R =  Dropout(0.2)(C11_R)

mid2_1 = concatenate([C11_R,P1_R])

mid2_1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(mid2_1)
x = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(mid2_1)

densenet = DenseNet201(weights='imagenet', include_top=False)

# input = Input(shape=(SIZE, SIZE, N_ch))
#x = Conv2D(3, (3, 3), padding='same',activation='relu')(s)
#x = Conv2D(3, (3, 3), padding='same',activation='relu')(x)
#x = Conv2D(3, (3, 3), padding='same',activation='relu')(x)
x = (Flatten())(x)

#branch 2

c_b_1=Conv2D(3, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)

branch_2 = densenet(c_b_1)
    
branch_2 = GlobalAveragePooling2D()(branch_2)
branch_2= BatchNormalization()(branch_2)
branch_2 = Dropout(0.5)(branch_2)
branch_2= Dense(256, activation='relu')(branch_2)

#concatenate model

final=concatenate([x,branch_2])
final = BatchNormalization()(final)
final = Dropout(0.2)(final)
final = Dense(1024, activation='relu')(final)
final= Dropout(0.2)(final)
final= Dense(512, activation='relu')(final)
final= Dropout(0.2)(final) #Extra
final= Dense(128, activation='relu')(final)
final= Dropout(0.5)(final) #Extra
final= Dense(64, activation='relu')(final)
final= Dropout(0.5)(final) #Extra

#multi output
output = Dense(3,activation = 'softmax', name='root')(final)
      
# model
model = Model(inputs,output)

optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)#lr=0.002
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])#kullback_leibler_divergence#categorical_crossentropy
model.summary()

