# Simplified DeepLearning
Simplified implementations of some interesting works related to deep learning.

## SGD Comparison
Comparison of various SGD algorithms on logistic regression and MLP. The relation of these algorithms is shown in the following figure, please refer to [sgd-comparison](https://github.com/SwordYork/simplified-deeplearning/tree/master/sgd-comparison) for the details. 
<p align="center">
<img src="/sgd-comparison/figures/relation.png?raw=true" width="80%">
</p>


## Function Approximation
Use neural network to approximate functions. The approximated functions are shown in the following figures, please refer to [function-approximation](https://github.com/SwordYork/simplified-deeplearning/tree/master/function-approximation) for the details. 
<p align="center">
<img src="/function-approximation/figures/lovefunc.png?raw=true" width="45%">
<img src="/function-approximation/figures/sinxx.png?raw=true" width="45%">
</p>


## Generative Adversarial Networks 
A simple demonstration of Generative Adversarial Networks (GAN), maybe problematic. 
<p align="center">
<img src="/GAN/figures/gaussian.png?raw=true" width="45%">
<img src="/GAN/figures/training.gif?raw=true" width="45%">
</p>
According to the [paper](https://arxiv.org/abs/1406.2661), we also use GAN to generate Gaussian distribution which shown in the left figure. Then we try to generate digits based on MNIST dataset, however, we encouter "the Helvetica scenario" in which G collapses too many values of z to the same value of x. Nevertheless, it is a simple demonstration, please see the [details](https://github.com/SwordYork/simplified-deeplearning/tree/master/GAN).
