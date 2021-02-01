# -*- coding: utf-8 -*-
"""matrix.ipynb
**
 * This file is part of Hybrid CNN-LSTM for COVID-19 Severity Score Prediction paper.
 *
 * Written by Ankan Ghosh Dastider and Farhan Sadik.
 *
 * Copyright (c) by the authors under Apache-2.0 License. Some rights reserved, see LICENSE.
 */
 
"""
model = load_model('') #Link CNN model weight directory
final_loss, final_accuracy = model.evaluate(Final_X, Y_train)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))

Y_pred = model.predict(X_Train)

Y_pred = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_train, axis=1)

cm = confusion_matrix(Y_true, Y_pred)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=disease_types, yticklabels=disease_types)
ax.set_ylabel('Actual', fontsize=40)
ax.set_xlabel('Predicted', fontsize=40)

