# Hybrid CNN-LSTM Model for COVID-19 Scoring from Lung Ultrasound

## Proposed Method
The proposed method consists of two stages:
1. CNN block is introduced by an autoencoder block and separable convolutional branches that adjoin the various parallel convolutional branches along with the DenseNet-201 network at different points
2. A deep CNN is followed by a recurrent neural network, LSTM to perform frame-based classification of the lung ultrasound images into four severity scores

![Pipeline](https://github.com/ankangd/HybridCovidLUS/blob/main/imgs/pipeline.png)

## Dataset

The dataset used in this study is presented [here](https://www.disi.unitn.it/iclus).
