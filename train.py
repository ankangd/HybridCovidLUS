# -*- coding: utf-8 -*-
"""train.ipynb

**
 * This file is part of Hybrid CNN-LSTM for COVID-19 Severity Score Prediction paper.
 *
 * Written by Ankan Ghosh Dastider and Farhan Sadik.
 *
 * Copyright (c) by the authors under Apache-2.0 License. Some rights reserved, see LICENSE.
 */

"""

#EPOCHS=70
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('Model.h5', verbose=1, save_best_only=True)
# Generates batches of image data with data augmentationV
datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
                        width_shift_range=0.2, # Range for random horizontal shifts
                        height_shift_range=0.2, # Range for random vertical shifts
                        zoom_range=0.2, # Range for random zoom
                        horizontal_flip=True, # Randomly flip inputs horizontally
                        vertical_flip=True) # Randomly flip inputs vertically

datagen.fit(X_train)
# Fits the model on batches with real-time data augmentation
hist = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
               steps_per_epoch=X_train.shape[0] //BATCH_SIZE,
               epochs=EPOCHS,
               verbose=2,
               callbacks=[annealer, checkpoint],
               validation_data=(X_val, Y_val))

#model.save("check.h5")
#model.fit(X_train,Y_train,batch_size=BATCH_SIZE,steps_per_epoch=X_train.shape[0] // BATCH_SIZE, epochs=EPOCHS)

