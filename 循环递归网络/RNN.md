# 循环神经网络
## RNN简介

&emsp;&emsp;**循环神经网络**(recurren neural network **RNN**)是一类用来处理序列数据的神经网络。就像卷积网络专门用来处理网格化数据**X**(如一个图像)的神经网络，循环神经网络是专门来处理序列$x^{(1)},x^{(2)},...x^{(\tau)}$的神经网络。正如卷积网络可以很容易地扩展到具有很大宽度和高度的图像，以及处理大小可变的图像，循环网络可以扩展到更长的序列，大多数循环网络也可以处理可变长度的序列。目前RNN已经广泛的用于自然语言处理中的语音识别，机器翻译等领域。


## RNN模型原理
&emsp;&emsp;RNN模型有比较多的变种，最主流的RNN模型结构如下图所示：

![rnn](img/rnn.png?raw=true "rnn")

&emsp;&emsp;图中损失函数$L$衡量每个$o$与相应的训练目标$y$的距离。当使用softmax输出时，我们假设$o$是未归一化的对数概率。损失函数$L$内部计算$\hat y= softmax(o)$，并将其与目标$y$比较。RNN输入到隐藏的连接由权重矩阵$U$参数化，隐藏到隐藏的循环连接由权重矩阵$W$参数化以及隐藏到输出的连接由权重矩阵$V$参数化。(左)使用循环连接绘制的RNN和它的损失。(右)同一网络被视为展开的计算图，其中每个节点现在与一个特定的与一个特定的时间实例相关联。

## RNN向前传播算法
&emsp;&emsp;现在我们研究上图中RNN的向前传播公式，假设该图隐藏层使用双曲正切函数作为激活函数，此外图中没有明确指定何种形式的输出和激活函数。假定输出是离散的，如用于预测词或者字符的RNN。表示离散变量的常规方式是把输出${\omicron}$作为每个离散变量可能值的非标准化对数概率。然后，应用softmax函数后续处理，获得标准化后概率的输出向量$\hat{y}$。RNN从特定初始状态$h^{(0)}$开始向前传播。从$t$=1到$t$=$\tau$的每个时间步，我们应用以下更新方程：
$$ a = b + Wh^{(t-1)} + Ux^{(t)}$$
$$h^{(t)} = tanh(a^{(h)})$$
$$o^{(t)} = c + Vh^{(t)}$$
$$\hat{y} = softmax(o^{(t)})$$
其中的参数的偏置$b$和$c$连接权重矩阵$U$、$V$和$W$，分别对应于输入到隐藏、隐藏到输出和隐藏到隐藏的连接。这个循环网络将一个输入序列映射到相同长度的输出序列。与$x$序列配对的$y$的总损失就是所有时间步的损失之和，例如$L^{(t)}$为给定的$x^{(1)},x^{(2)},...x^{(\tau)}$后$y^{(t)}$的负对数似然则
$$L(\{ x^{(1)},...,x^{(\tau)}\},\{ y^{(1)},...,y^{(\tau)}\}) = \sum\limits_t L^{(t)} $$
$$ = -\sum\limits_t log\ pmodel(y^{(t)}|\{ x^{(1)},...,x^{(\tau)}\}) $$
其中$pmodel(y^{(t)}|\{ x^{(1)},...,x^{(\tau)}\})$需要读取模型的输出向量$\hat y$中对应于$y^{(t)}$的项。
## 基于反向传播计算RNN梯度

&emsp;&emsp;计算循环神经网络的梯度是容易的，可以将反向传播算法应用于展开的计算图，由反向传播计算得到梯度，并结合任何通用的基于梯度的技术可以训练RNN。计算图的节点包括参数$U$、$V$、$W$、$b$和$c$，以及$t$为索引的节点序列$x^{(t)}$,$h^{(t)}$,$o^{(t)}$,$L^{(t)}$。对于每一个节点**N**，基于**N**后面的节点，递归的计算梯度$\nabla_NL$。从紧着的最终的损失节点开始递归：
$$
\frac{\partial L}{\partial L^{(t)}} = 1
$$
在这个导数中，假设输出$o^{(t)}$作为softmax函数的参数，$\hat y$作为softmax函数输出的概率。同时假设损失是迄今为止给定了输入后的真是目标$y^{(t)}$的负对数似然。对于所有$i,t$,关于时间同步输出的梯度$\nabla_{o^{(t)}}L$如下：
$$
(\nabla_{o^{(t)}}L)_i=\frac{\partial L}{\partial o_i^{(t)}}
=\frac{\partial L}{\partial L^{(t)}}
\frac{\partial L^{(t)}}{\partial o_i^{(t)}}
={\hat y}_i^{(t)} - y^{(t)}
$$
从序列的末尾开始，反向进行计算。在最后的时间步$\tau$,$h^{(\tau)}$只有$o^{(\tau)}$作为后续节点，计算梯度。
$$
\nabla_{h^{(\tau)}}L = V^{T}\nabla_{o^{(\tau)}}L
$$  
然后从$t = \tau -1$ 到$t =1$反向迭代，通过时间反向迭代梯度，$h^{(t)} \ (t < \tau)$同时具有$o^{(t)}$和$h^{(t+1)}$两个后续节点，梯度计算公式如下：
$$
\nabla_{h^{(t)}}L=
{\left( \frac{\partial h^{(t)}}{\partial h^{(t+1)}}\right)}^{T}
(\nabla_{h^{(t+1)}}L)
+ {\left( \frac{\partial o^{(t)}}{\partial h^{(t)}}\right)}^{T}
(\nabla_{o^{(t)}}L)$$
$$
=W^{T}(\nabla_{h^{(t+1)}}L)diag\left( 1- (h^{(t+1)})^2\right)
+V^{T}(\nabla_{o ^{(t)}}L)
$$
其中$diag\left( 1- (h^{(t+1)})^2\right)$表示包含元素$\left( 1- (h_i^{(t+1)})^2\right)$的对角矩阵，它是关于时刻$t+1$与隐藏单元$i$关联的双曲正切Jacobian矩阵，一旦获得了计算图内部节点的梯度，就可以得出关于各个参数节点的梯度。

$$
\nabla_cL=\sum\limits_t
{\left( \frac{\partial o^{(t)}}{\partial c}\right)}^{T}
\nabla_{o^{(t)}}L
=\sum\limits_t
\nabla_{o^{(t)}}L
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ 
$$
$$
\nabla_bL=\sum\limits_t
{\left( \frac{\partial h^{(t)}}{\partial b^{(t)}}\right)}^{T}
\nabla_{h^{(t)}}L
=\sum\limits_t
diag\left( 1- (h^{(t)})^2\right)
\nabla_{h^{(t)}}L
$$
$$
\nabla_VL=\sum\limits_t\sum\limits_i
{\left( \frac{\partial L}{\partial o_i^{(t)}}\right)}^{T}
\nabla_{V}L
=\nabla_Vo_i^{(t)}
=\sum\limits_t
(\nabla_{o^{(t)}}L)h^{(t)^T}
$$
$$
\nabla_WL=\sum\limits_t\sum\limits_i
{\left( \frac{\partial L}{\partial h_i^{(t)}}\right)}^{T}
\nabla_{W^{(t)}}h_i^{(t)} 
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ 
\ \ \ \ \ \ \ \ \ \ \ \ \ 
$$
$$
=\sum\limits_t
diag\left( 1- (h^{(t)})^2\right)
(\nabla_{h^{(t)}}L)
h^{(t-1)^T}
\ \ \ \ \ \ \ \ \ \ \ \ 
$$
$$
\nabla_UL=\sum\limits_t\sum\limits_i
{\left( \frac{\partial L}{\partial h_i^{(t)}}\right)}^{T}
\nabla_{U^{(t)}}h_i^{(t)} 
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ 
\ \ \ \ \ \ \ \ \ \ \ \ \ 
$$
$$
=\sum\limits_t
diag\left( 1- (h^{(t)})^2\right)
(\nabla_{h^{(t)}}L)
x^{(t)^T}
\ \ \ \ \ \ \ \ \ \ \ \ \ \ 
$$
除了梯度表达式不同，RNN的反向传播算法和DNN区别不大。
## RNN梯度消失问题
&emsp;&emsp;在深度学习中的优化章节曾提到过 在一些算中，当计算图变得非常深的时候，会面临一个长期依赖的问题。循环神经网络需要在很长的时间序列的各个时刻重复应用相同操作来构建非常深的计算图。经过多阶传到后的梯度倾向于消失或爆炸（多数情况下会消失）。假如某个计算图包含一条反复与矩阵 $W$ 相乘的路径，那么 $t$ 步之后，相当于乘以 $W^t$。假设$W$ 有特征分解：
$
W = Vdiag(\lambda)V^{-1}
$
则：
$$ 
W^t = (Vdiag(\lambda)V^{-1})^t
= Vdiag(\lambda)^tV^{-1}
$$

当特征值 $\lambda_i$ 不在 1 附近时，若大于1，则会爆炸，若小于1，则会消失。梯度消失与爆炸问题是指该计算图上面的梯度会因 $diag(\lambda)^t$ 发生大幅度的变化。
## RNN总结
&emsp;&emsp;本章节介绍了最经典的RNN模型结构，RNN的向前传播算法，反向梯度计算，以及RNN的梯度消失问题。RNN还有很多拓展，例如双向RNN、基于编码-解码的序列到序列的架构、深度循环网络、递归神经网络等。

&emsp;&emsp;RNN虽然理论上可以很漂亮的解决序列数据的训练，但是由于梯度消失的问题，无法应用到很长的序列。在语音识别，手写识别以及机器翻译等NLP领域实际应用比较广泛的是基于RNN模型的一个拓展LSTM，接下来我们就来讨论LSTM模型。

