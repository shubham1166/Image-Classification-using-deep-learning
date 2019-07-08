#Image Classification using deep learning

## Classification
Image classification basically deals with classifying images in different classes. The dataset that we have used is MNIST dataset of hand written digits. It consists of 70,000 gray-scale images of hand-written digits of dimension (28,28)
**![](https://lh3.googleusercontent.com/lsRaeabUc3hWWO8v-3jAJ_68Atld7YafVAFFXnNLQCUZ46_nYDVqKrUNee-woiCcGgdKdF7Vb80KcS9GrSACg6ly8ATHNXStRef0paCDNUImogn307u4_rT1Q_HIXSLmdlm2I8xq)**

---
## Basic dense architecture
In the first part, we have tried to make a network. The network architecture consists of two hidden layers with 512 and 256 and then the output layer with softmax classifier. We have run the network for 40 epochs . 
**![](https://lh3.googleusercontent.com/QoXnbGwMqp90PBZQNNPGGYTTSeqEjAVupoeFbygydFC2SZF-sDwVd_IFNFAXeoMd6Z0CpUTtR-C8R0enZSjmh2jMT7bnD9a1oAyF9ePLlVsV3tMzkfo2BLSJe97N0HZiZck55HqU)**


The accuracy and loss on the test data is as follows:
**![](https://lh3.googleusercontent.com/UQo0-mMZ_xn5ROYsz_HzJW1u1XHEE5IRt-Sf3rb4EOT1eZfhGUvaWGZ44yQakJymeoPvAzoTYGyieiXicXhceWp6l9Lat59gErS7-mZ09vkEWbSV22LiacJDp4A9drYFbTNlxt6k)**

---

## CNN Classification network
In this part, the network architecture that we are using contains two convolutional layers one after the other with a filter size of (3,3). The first block contains 32 different fuilters of (3,3) and second layer consists of 64 different filters of (3,3) which are followed by a maxpool layer with shride 2 to reduce its size, then there is a dense layer which is then followed by the output layer with softmax classifier. We have used dropouts in dense layers.
**![](https://lh6.googleusercontent.com/fcly-c2bFv8q9SEgsHS6uMQIgNudz9vgSmekEQ_dECJ7bcK1XJ-ChLirooaSJ8EQG6oZVuXs2M3oLJ9_LR6gvwxZyaV4iHv7ozoUOfEv5KkdKXmiF1xL9ZEDQiUPeIGhka-EPBY_)**

The Network architecture is as follows(by tensorboard):
**![](https://lh3.googleusercontent.com/IisFSQfQ-EyTeAQ-p1CqF1N9wTeZ0W_qpp7q09nwEyAovAUl-7HkJajaxr1Fzq6vCiFjfrgKnRNAFrEEF9MKLJiCuLV14XqdaAoi-FOGosrd4CrXsdLDPFsYT2RXqllkHTaBhV-J)**
The accuracy and loss on the test data is as follows:
**![](https://lh5.googleusercontent.com/rCEHbIhTwlV6zegpEkgvnlc55vvbB12BTA5d-Dud4e4LwnMr-lzQw6Nkgm4ZfEEPfv8lcb4R7m3B85DipDrt9-IVOrqhJtlec1bIXBBDQEYygG9zEqlnn_JaG0XdeM52Y3U4tMI9)**

---
## VGGNet network
![](https://www.pyimagesearch.com/wp-content/uploads/2017/03/imagenet_vgg16.png)
A viualization of VGG architecture ([Source](https://www.cs.toronto.edu/~frossard/post/vgg16/))

This network is characterized by its simplicity, using only 3×3 convolutional layers stacked on top of each other in increasing depth. VGGNet is invented by VGG ([Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/)) from University of Oxford. VGGNet is the 1st runner-up of the ILSVRC (ImageNet Large Scale Visual Recognition Competition) 2014 in the classification task. We have tried to classiffy images using VGGNet. The main use of VGG was earlier 11×11 or 5×5 filters were used in convolutional steps. The main idea of VGGNet is to use 3×3 filters in convolutional layers followed by other 3×3 filters convolutional layers. Thus, it is computationally better than 11×11 or 5×5 filters convolutions. Two consecutive convolutional layer step is then followed by a Max-pool layer which will reduce the size of the image, allowing the next convolutional layer to learn bigger patterns in the images. 
**![](https://lh5.googleusercontent.com/aU5VoP3NTcYXZiyVCQrgF1IQoCkofbBpwiEAVhkAVUPVgQofj8Ce-OiwhDvUVs3WpVSGlfWmxYfzlnvjDHwmMyZN3vedvhHkRIf1QoFtQKcQIMWjmESxp19Yb2sAFrqFFVfDqFSz)**
The Network architecture is as follows(by tensorboard):
**![](https://lh4.googleusercontent.com/MaqkfNaHjAP3VmjyoLPNR9A-qerG23PlVCMGPPk9czL4bjfbAurYP5GQ9sAmK1EKeBAY9OYfk268p01c4f7GPudtth19LYh-WiBGHJoIq7TuXT2UYdOTRPhg8281a14aauVsLtdA)**


The accuracy and loss on the test data is as follows:
**![](https://lh6.googleusercontent.com/GqW8MRHTU9lRg9AhM8u2uB416a080xG5vzNn0-O12kiFVJzPElHT4es512D-nTGPQ8QqFtxOVJffKVoN-wvuTvb_1sTJMvJMtJBGbZIhuuH6HdHBTDYCtvyfDN-C5R7YlNsfFvwh)**

---
## ResNet Network

![](https://cdn-images-1.medium.com/max/1200/1*2ns4ota94je5gSVjrpFq3A.png)

ResNet Architecture([Source](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035))
In case of very large networks, there is a problem of accuracy saturation. Thus, ResNet is based on the idea of use of residual blocks. 
![](https://cdn-images-1.medium.com/max/1200/1*ByrVJspW-TefwlH7OLxNkg.png)
a residual block([source](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035))

Residual blocks are based on the idea of skip connections or identity shortcut connections, that can skip one or more layer. Thus, helps to build bigger networks. We are using transfer learning and the inbuilt resnet from Keras. The activation from a previous layer is being added to the activation of a deeper layer in the network.
**![](https://lh5.googleusercontent.com/dANHYzL_9JVKcmn0SFsZsxlmiXfkTWAhKvUsBXvtTZEPNrOoj6QDJRdoNDdDhoDvXks2rSuQmrPj80P8XkAc7mf3RMuCVgh-BzOKuRTryK7CY_sxIMyw9WRCvVa407BxW004g9I8)**

The Network architecture is as follows(by tensorboard):
**![](https://lh5.googleusercontent.com/58URlCqofXbwNlbwHOs6CPO2PxFIl25F81XQVxosratAQ7fSRxkd_5Su50i5y913eMIEXsvCSqCj7HU5-uaaW7BAUrciE9WN9-_dfyL9fl9FopvWLvPQ2ADnLMh2kep48iBkrd1d)**



The accuracy and loss on the test data is as follows:
**![](https://lh6.googleusercontent.com/9QGUMccDaceCOHZPPALrx8afY2dD0D_cyi3OjSQuP-sBTM0tSL2eI_6jGvXNPtT0i6ARs-EgbXLNZ2HKZcRslySmCbg0pAmAV6kZTbUgeALXLQ1xXqXd5LreGfu6LQjVLRjTZ0CU)**


