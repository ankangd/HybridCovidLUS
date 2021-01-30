# -*- coding: utf-8 -*-
"""Train_LSTM.ipynb
**
 * This file is part of Hybrid CNN-LSTM for COVID-19 Severity Score Prediction paper.
 *
 * Written by Ankan Ghosh Dastider and Farhan Sadik.
 *
 * Copyright (c) by the authors under Apache-2.0 License. Some rights reserved, see LICENSE.
 */

"""

'''
Loading frames of the videos sequentially
'''
video_types=['Video 01', 'Video 05', 'Video 06', 'Video 07', 'Video 08', 'Video 09', 'Video 10', 'Video 14', 
             'Video 15', 'Video 16', 'Video 17', 'Video 20', 'Video 21', 'Video 27', 'Video 29']

NUM_VIDEOS = len(video_types)
NUM_FRAMES = 302

data_dir_lstm = ''  #Link Training Directory videowise
train_dir_lstm = os.path.join(data_dir_lstm)

train_data_lstm = []
for defects_id, sp in enumerate(video_types):
    temporary = []
    for file in sorted(os.listdir(os.path.join(train_dir_lstm, sp))):
        temporary.append(['{}/{}'.format(sp, file), defects_id, sp])

    total_frames = len(temporary)
    index = np.linspace(start = 0, stop = total_frames-1, num = NUM_FRAMES, dtype = int)

    for i in range(NUM_FRAMES):
        train_data_lstm.append(temporary[index[i]])
        
train_on_lstm = pd.DataFrame(train_data_lstm, columns=['File', 'FolderID','Video Type'])
train_on_lstm.head(NUM_VIDEOS*NUM_FRAMES)

video_types=['Video 01', 'Video 05', 'Video 06', 'Video 07', 'Video 08', 'Video 09', 'Video 10', 'Video 14', 
             'Video 15', 'Video 16', 'Video 17', 'Video 20', 'Video 21', 'Video 27', 'Video 29']

data_dir_lstm = '' #Link Training Directory videowise
train_dir_lstm = os.path.join(data_dir_lstm)

train_data_lstm = []
for defects_id, sp in enumerate(video_types):
    for file in sorted(os.listdir(os.path.join(train_dir_lstm, sp))):
        # print(file)
        train_data_lstm.append(['{}/{}'.format(sp, file), defects_id, sp])
        
train_on_lstm = pd.DataFrame(train_data_lstm, columns=['File', 'FolderID','Video Type'])
train_on_lstm.head()

IMAGE_SIZE = 128
SEED = 42

BATCH_SIZE_LSTM = 25
EPOCHS_LSTM = 120

def read_image_lstm(filepath):
    return cv2.imread(os.path.join(data_dir_lstm, filepath)) # Loading a color image is the default flag

#Resize image to target size
def resize_image(newimage, image_size):
    return cv2.resize(newimage.copy(), image_size, interpolation=cv2.INTER_AREA)

from tensorflow.keras.models import load_model
import re
from keras import backend as K

X_Train_Total = np.zeros((NUM_VIDEOS, NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3))
Y_Train_Total = np.zeros((NUM_VIDEOS, NUM_FRAMES, 1))
k = 0
j = 0
for i, file in tqdm(enumerate(train_on_lstm['File'].values)):
    if i % NUM_FRAMES == 0 and i != 0 :
        k = k + 1
        j = 0
    if k == NUM_VIDEOS:
        break
    # print(i,file)
    newimage = read_image_lstm(file)
    if newimage is not None:
        # print(k,j)
        X_Train_Total[k,j] = resize_image(newimage, (IMAGE_SIZE, IMAGE_SIZE))
        match = re.search('Score(\d)',file)
        score = int(match.group(1))
        Y_Train_Total[k,j] = score
        #print(file)
        #print(score)
        #print(Y_test[k,j])
    j = j + 1

Y_Train_Total = to_categorical(Y_Train_Total, num_classes=4)
# print(Y_Train_Total)
# Normalize the data
X_Train_Total = X_Train_Total / 255.
print('X_Train_Total Shape: {}'.format(X_Train_Total.shape))
print('Y_Train_Total Shape: {}'.format(Y_Train_Total.shape))

np.random.seed(42)
np.random.shuffle(X_Train_Total)

np.random.seed(42)
np.random.shuffle(Y_Train_Total)

print('X_Train_Total Shape: {}'.format(X_Train_Total.shape))
print('Y_Train_Total Shape: {}'.format(Y_Train_Total.shape))

model = load_model('') #Link the CNN weights
model.summary()

output = np.zeros((NUM_VIDEOS, NUM_FRAMES, 64))

for i in range(NUM_VIDEOS):
  X_New = X_Train_Total[i]
  specific_layer_output = K.function([model.layers[0].input], [model.get_layer('dropout_35').output])
  layer_output = specific_layer_output([X_New])[0]
  #print(layer_output.shape)
  #print(layer_output)
  output[i] = layer_output

print('Output from CNN Shape: {}'.format(output.shape))
#custom3 = model.predict(X_Test)
#print(custom3)

X_Train_Total = output
Y_Train_Total = Y_Train_Total

print('X_Train_Total Shape: {}'.format(X_Train_Total.shape))
print('Y_Train_Total Shape: {}'.format(Y_Train_Total.shape))

# Split the train and validation sets 
X_Train_LSTM, X_Val_LSTM, Y_Train_LSTM, Y_Val_LSTM = train_test_split(X_Train_Total, Y_Train_Total, 
                                                                      test_size=0.2, random_state = SEED)

from keras.layers import Reshape, LSTM, Lambda, TimeDistributed, Conv1D, MaxPool1D, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D


def build_lstm():

    input = Input(shape=(NUM_FRAMES, 64))
    

    x = LSTM(1000, return_sequences = True)(input)
    x = Dropout(0.5)(x)
    
    x = LSTM(1000, return_sequences = True)(x)
    x = Dropout(0.5)(x)

    x = LSTM(4, return_sequences=True)(x)
    # multi output
    output = Dense(4,activation = 'softmax', name='root')(x)

    # model
    model = Model(input,output)
    
    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    return model

model_lstm = build_lstm()
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
checkpoint = ModelCheckpoint('model_lstm.h5', verbose=1, save_best_only=True)
# Generates batches of image data with data augmentation
# datagen = ImageDataGenerator(rotation_range=360, # Degree range for random rotations
 #                       width_shift_range=0.2, # Range for random horizontal shifts
  #                      height_shift_range=0.2, # Range for random vertical shifts
   #                     zoom_range=0.2, # Range for random zoom
    #                    horizontal_flip=True, # Randomly flip inputs horizontally
     #                   vertical_flip=True) # Randomly flip inputs vertically

#datagen.fit(X_train)
# Fits the model on batches with real-time data augmentation
hist = model_lstm.fit(X_Train_LSTM, Y_Train_LSTM, batch_size = BATCH_SIZE_LSTM,
                     # steps_per_epoch=X_Train_LSTM.shape[0] // BATCH_SIZE,
                     epochs = EPOCHS_LSTM,
                     verbose = 2,
                     callbacks = [annealer, checkpoint],
                     validation_data = (X_Val_LSTM, Y_Val_LSTM))

final_loss_lstm, final_accuracy_lstm = model_lstm.evaluate(X_Val_LSTM, Y_Val_LSTM)
print('Final Loss LSTM: {}, Final Accuracy LSTM: {}'.format(final_loss_lstm, final_accuracy_lstm))

score_types = ['Score 0', 'Score 1', 'Score 2', 'Score 3']

Y_pred_lstm = model_lstm.predict(X_Val_LSTM)
Y_pred_lstm = np.reshape(Y_pred_lstm, (Y_pred_lstm.shape[0]*Y_pred_lstm.shape[1], Y_pred_lstm.shape[2]))
Y_pred_lstm = np.argmax(Y_pred_lstm, axis=1)

Y_true_lstm = np.reshape(Y_Val_LSTM, (Y_Val_LSTM.shape[0]*Y_Val_LSTM.shape[1], Y_Val_LSTM.shape[2]))
Y_true_lstm = np.argmax(Y_true_lstm, axis=1)

#print(Y_pred_lstm.shape)
#print(Y_Val_LSTM.shape)
cm = confusion_matrix(Y_true_lstm, Y_pred_lstm)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=score_types, yticklabels=score_types)
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)

'''
# accuracy plot 
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss plot
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''