# -*- coding: utf-8 -*-
"""Train_Loader.ipynb

**
 * This file is part of Hybrid CNN-LSTM for COVID-19 Severity Score Prediction paper.
 *
 * Written by Ankan Ghosh Dastider and Farhan Sadik.
 *
 * Copyright (c) by the authors under Apache-2.0 License. Some rights reserved, see LICENSE.
 */

"""

IMAGE_SIZE =128

def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
# Resize image to target size
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)

X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
X_train_ft=np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    if image is not None:
        
        X_train[i] =resize_image(image, (IMAGE_SIZE, IMAGE_SIZE))
    

X_Train = X_train / 255.
print('Train Shape: {}'.format(X_Train.shape))

Y_train = train['DiseaseID'].values
print(len(Y_train))
Y_train = to_categorical(Y_train, num_classes=4)

