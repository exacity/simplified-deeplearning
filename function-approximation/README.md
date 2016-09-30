# Function Approximation
Use neural network to approximate functions. The code is based on [keras](https://keras.io/), please install relative packages.

## MLP
We can use MLP to approximate some nonlinear functions in a bounded region, for example, the [love function](http://functionspace.com/topic/360/What-are-the-funniest-beautiful-graph-s-equations-), $x^2-(y-\sqrt[3]{x^2})^2$. This figure shows the learnt function.

![lovefunc](/function-approximation/figures/lovefunc.png?raw=true "lovefunc")

## RNN
However, MLP is unable to approximate the functions outside the region. We could use RNN to approximate some periodic functions such as $sin(x)$, and it could be used in time series prediction.

![sinx](/function-approximation/figures/sinx.png?raw=true "sinx")
![sinxx](/function-approximation/figures/sinxx.png?raw=true "sinxx")

We use RNN to approximate $sin(x)$ and $sin(x) * \sqrt[10]{x}$ which respectively shown in the above figures.

