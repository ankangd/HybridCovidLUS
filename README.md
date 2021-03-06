# Hybrid CNN-LSTM Model for COVID-19 Scoring from Lung Ultrasound

> **An integrated autoencoder-based hybrid CNN-LSTM model for COVID-19 severity prediction from lung ultrasound**<br>
> [Ankan Ghosh Dastider](https://github.com/ankangd), [Farhan Sadik](https://github.com/farhan1606125), and Shaikh A. Fattah<br>
> In Computers in Biology and Medicine.<br>

> Paper: [ScienceDirect link](https://www.sciencedirect.com/science/article/pii/S0010482521000901)<br>

> **Abstract:** *The COVID-19 pandemic has become one of the biggest threats to the global healthcare system, creating an unprecedented condition worldwide. The necessity of rapid diagnosis calls for alternative methods to predict the condition of the patient, for which disease severity estimation on the basis of Lung Ultrasound (LUS) can be a safe, radiation-free, flexible, and favorable option. In this paper, a frame-based 4-score disease severity prediction architecture is proposed with the integration of deep convolutional and recurrent neural networks to consider both spatial and temporal features of the LUS frames. The proposed convolutional neural network (CNN) architecture implements an autoencoder network and separable convolutional branches fused with a modified DenseNet-201 network to build a vigorous, noise-free classification model. A five-fold cross-validation scheme is performed to affirm the efficacy of the proposed network. In-depth result analysis shows a promising improvement in the classification performance by introducing the Long Short-Term Memory (LSTM) layers after the proposed CNN architecture by an average of , which is approximately  more than the traditional DenseNet architecture alone. From an extensive analysis, it is found that the proposed end-to-end scheme is very effective in detecting COVID-19 severity scores from LUS images.*


## Proposed Method
The proposed method consists of two stages:
1. CNN block is introduced by an autoencoder block and separable convolutional branches that adjoin the various parallel convolutional branches along with the DenseNet-201 network at different points
2. A deep CNN is followed by a recurrent neural network, LSTM to perform frame-based classification of the lung ultrasound images into four severity scores

![Pipeline](https://github.com/ankangd/HybridCovidLUS/blob/main/imgs/pipeline.png)


## Results
### Frame-based Score Prediction

<p align="center">
    <img src="https://github.com/ankangd/HybridCovidLUS/blob/main/imgs/Table_2.png" width="800"> <br />
    <em> 
    </em>
</p>


## Dataset

The dataset used in this study is presented [here](https://www.disi.unitn.it/iclus).

## Citation

Please cite our paper if you find the work useful:

    @article{DASTIDER2021104296,
    title = {An integrated autoencoder-based hybrid CNN-LSTM model for COVID-19 severity prediction from lung ultrasound},
    journal = {Computers in Biology and Medicine},
    volume = {132},
    pages = {104296},
    year = {2021},
    issn = {0010-4825},
    doi = {https://doi.org/10.1016/j.compbiomed.2021.104296},
    url = {https://www.sciencedirect.com/science/article/pii/S0010482521000901},
    author = {Ankan Ghosh Dastider and Farhan Sadik and Shaikh Anowarul Fattah}.    }
