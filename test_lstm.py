# -*- coding: utf-8 -*-
"""Test_LSTM.ipynb

**
 * This file is part of Hybrid CNN-LSTM for COVID-19 Severity Score Prediction paper.
 *
 * Written by Ankan Ghosh Dastider and Farhan Sadik.
 *
 * Copyright (c) by the authors under Apache-2.0 License. Some rights reserved, see LICENSE.
 */

"""

model = load_model('') #Link CNN model weight directory
model_lstm = load_model('') #Link LSTM model weight directory

data_dir_test = '' #Link test video
test_dir = os.path.join(data_dir_test)

test_data = []
for file in sorted(os.listdir(test_dir)):
    # print(file)
    test_data.append(['{}'.format(file)])
        
test_on = pd.DataFrame(test_data, columns=['File'])
test_on.head()

IMAGE_SIZE = 128
NUM_FRAMES = test_on.shape[0]

def read_image_test(filepath):
    return cv2.imread(os.path.join(data_dir_test, filepath)) # Loading a color image is the default flag
# Resize image to target size
def resize_image(newimage, image_size):
    return cv2.resize(newimage.copy(), image_size, interpolation=cv2.INTER_AREA)

X_Test = np.zeros((NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3))
Y_Test = np.zeros((NUM_FRAMES, 1))

for i, file in tqdm(enumerate(test_on['File'].values)):
    newimage = read_image_test(file)
    if newimage is not None:
        X_Test[i] = resize_image(newimage, (IMAGE_SIZE, IMAGE_SIZE))
        match = re.search('Score(\d)',file)
        score = int(match.group(1))
        Y_Test[i] = score
        #print(file)
        #print(score)

Y_Test = to_categorical(Y_Test, num_classes=4)
# print(Y_Test)
# Normalize the data
X_Test = X_Test / 255.
print('X_Test Shape: {}'.format(X_Test.shape))
print('Y_Test Shape: {}'.format(Y_Test.shape))

output = np.zeros((1, NUM_FRAMES, 64))

specific_layer_output = K.function([model.layers[0].input], [model.get_layer('dropout_35').output])
layer_output = specific_layer_output([X_Test])[0]
#print(layer_output.shape)
#print(layer_output)
output[0] = layer_output

print('Output from CNN Shape: {}'.format(output.shape))
#custom3 = model.predict(X_Test)
#print(custom3)

score_types = ['Score 0', 'Score 1', 'Score 2', 'Score 3']
Y_pred_test = model_lstm.predict(output)
Y_pred_test = np.reshape(Y_pred_test, (Y_pred_test.shape[1], Y_pred_test.shape[2]))
Y_pred_test = np.argmax(Y_pred_test, axis=1)

Y_true_test = np.argmax(Y_Test, axis=1)

#print(Y_pred_lstm.shape)
#print(Y_Val_LSTM.shape)

cm = confusion_matrix(Y_true_test, Y_pred_test)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=score_types, yticklabels=score_types)
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)

final_loss, final_accuracy = model.evaluate(X_Test, Y_Test)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))

