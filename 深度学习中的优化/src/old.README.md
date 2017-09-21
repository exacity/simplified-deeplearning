# SGD Comparison
Test various SGD algorithms on logistic regression and MLP, including

 - vanilla SGD
 - Momentum
 - Nesterov Accelerated Gradient
 - AdaGrad
 - RMSProp
 - AdaDelta
 - Adam
 - Adamax

The relation of these algorithms is shown in the following figure (my personal view).
![relation](img/relation.png?raw=true "relation")

This code is based on [Theano](https://github.com/Theano/Theano), please install relative packages.  The implementation of  logistic regression and MLP is based on the Theano [tutorial](http://deeplearning.net/tutorial/logreg.html).

## Test results
We measure the performance of these SGD algorithms by comparing the training curve and validation error.
### Logistic Regression
![LR](img/lr.png?raw=true "lr")
### MLP
![MLP](img/mlp.png?raw=true "mlp")

For more details about these algorithms, please refer to my [blog](https://blog.slinuxer.com/2016/09/sgd-comparison) (Chinese). 
