# -*- coding: utf-8 -*-
"""grad_cam.ipynb

**
 * This file is part of Hybrid CNN-LSTM for COVID-19 Severity Score Prediction paper.
 *
 * Written by Ankan Ghosh Dastider and Farhan Sadik.
 *
 * Copyright (c) by the authors under Apache-2.0 License. Some rights reserved, see LICENSE.
 */
 
"""
import tensorflow as tf
import keras
print('tensorflow version: ', tf.__version__)
print('keras version: ', keras.__version__)
import numpy as np
import cv2

#For images
from keras.preprocessing.image import array_to_img, img_to_array, load_img
#For model loading
from keras.models import load_model
#For Grad-CAM calculation
from tensorflow.keras import models
import tensorflow as tf

IMAGE_SIZE  = (128, 128)

def grad_cam(input_model, x, layer_name):
    """
    Args: 
        input_model(object):Model object
        x(ndarray):image
        layer_name(string):The name of the convolution layer
    Returns:
        output_image(ndarray):Colored image of the original image
    """

    #Image preprocessing
    #Since there is only one image to read, mode must be increased..I can't predict
    X = np.expand_dims(x, axis=0)
    preprocessed_input = X.astype('float32')
    # / 255.0    

    grad_model = models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_input)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    #Calculate the gradient
    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')

    guided_grads = gate_f * gate_r * grads

    #Average the weights and multiply by the output of the layer
    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    #Scale the image to the same size as the original image
    cam = cv2.resize(cam, IMAGE_SIZE, cv2.INTER_LINEAR)
    #Instead of ReLU
    cam  = np.maximum(cam, 0)
    #Calculate heatmap
    heatmap = cam / cam.max()

    #Pseudo-color monochrome images
    jet_cam = cv2.applyColorMap(np.uint8(255.0*heatmap), cv2.COLORMAP_JET)
    #Convert to RGB
    rgb_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    #Combined with the original image
    output_image = (np.float32(rgb_cam) + x / 2)  
    
    return output_image

target_layer = 'concatenate_5'
cam = grad_cam(model, x, target_layer)
out_1=rgb2gray(z)*cam[:,:,0]
out_2=rgb2gray(z)*cam[:,:,1]
out_3=rgb2gray(z)*cam[:,:,2]
out=np.stack((out_1,out_2,out_3),axis=2)
out=out/255.

