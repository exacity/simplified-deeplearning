# Basic RNN
An implement of RNN Using mnist to test.

The size of mnist image is 28*28.

For a row, the previous row and next row can be considered as its context.

Thus, an image can be transformed into a sample with 28 steps. Each step is 28 dimensional.

However, overfitting is common for RNN. For mnist, we can get an accuracy of 80%-90% in 1000 step. After 1000 step, the accuracy will decrease.

# Requirement
- Python3
- tensorflow 1.0