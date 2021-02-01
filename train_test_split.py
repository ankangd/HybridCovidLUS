# -*- coding: utf-8 -*-
"""Train_test_split.ipynb

**
 * This file is part of Hybrid CNN-LSTM for COVID-19 Severity Score Prediction paper.
 *
 * Written by Ankan Ghosh Dastider and Farhan Sadik.
 *
 * Copyright (c) by the authors under Apache-2.0 License. Some rights reserved, see LICENSE.
 */
 
"""

get_ipython().__class__.__name__ = "ZMQInteractiveShell"

print(X_Train.shape)
print(Y_train.shape)

BATCH_SIZE =64


# Split the train and validation sets 
#X_train, X_val, Y_train, Y_val = train_test_split(Final_X, Y_train, test_size=0.2, random_state=SEED)
X_train, X_val, Y_train, Y_val = train_test_split(X_Train, Y_train, test_size=0.2, random_state=SEED)

