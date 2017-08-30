# Generative Adversarial Networks
A simple demonstration of Generative Adversarial Networks (GAN). The code is based on [keras](https://keras.io/), please install relative packages.

Please check the code or refer to my [blog](https://blog.slinuxer.com/2016/10/generative-adversarial-networks) for more details.

## Generative Adversarial Networks 
A simple demonstration of Generative Adversarial Networks (GAN), maybe problematic. 
<p align="center">
<img src="img/gaussian.png?raw=true" width="45%">
<img src="img/training.gif?raw=true" width="45%">
</p>

According to the [paper](https://arxiv.org/abs/1406.2661), we also use GAN to generate Gaussian distribution which shown in the left figure. Then we try to generate digits based on MNIST dataset, however, we encouter "the Helvetica scenario" in which G collapses too many values of z to the same value of x. Nevertheless, it is a simple demonstration, please see the [details](https://github.com/SwordYork/simplified-deeplearning/tree/master/GAN).

