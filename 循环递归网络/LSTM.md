
# LSTM
## LSTM概述
&emsp;&emsp;长短记忆(Long Short Term Memory,LSTM)是一种 RNN 特殊的类型，可以学习长期依赖信息,它引入了自循环的巧妙构思，以产生梯度长时间持续流动的路径，解决RNN梯度消失或爆炸的问题。在手写识别、机器翻译、语音识别等应用上，LSTM 都取得相当巨大的成功，并得到了广泛的使用。

&emsp;&emsp;LSTM循环网络除了外部的RNN循环外，还具有内部的“LSTM细胞”自循环，和RNN相比LSTM每个单元有相同的输入和输出参数，但也有更多的参数和控制信息流流动的门控单元系统，门控单元包括输入门、输出门、遗忘门。模型图如下：
![rnn](img/lstm_model.png?raw=true "rnn")
&emsp;&emsp;如上图所示，查看$X_t$时刻的细胞状态，细胞彼此循环连接，代替一般循环网络中的普通的隐藏单元。这里使用人工的神经元计算输入特征，如果sigmoid输入门允许，它的值可以累加到状态。状态单元具有线性自循环，其权重由遗忘门控制。细胞的输出可以被输出门关闭。所有的门控单元都具有sigmoid非线性，而输入单元具有任意的压缩非线性。状态单元也可以作为门空单元的额外输入。

## LSTM核心思想
LSTM的主要思想是采用一个叫做“细胞状态(state)”的通道来贯穿整个时间序列。

&emsp;![rnn](img/state1.png?raw=true "rnn")

&emsp;&emsp;通过精心设计“门”的结构来去除或增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。它们包含一个 sigmoid 神经网络层和一个逐元乘法操作。

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![rnn](img/state2.png?raw=true "rnn")

&emsp;&emsp;Sigmoid 层输出 0 到 1 之间的数值，描述每个部分有多少量可以通过。0 代表“不许任何量通过”，1 就指“允许任意量通过”。LSTM 拥有三个门，来保护和控制细胞状态。下面详细介绍LSTM三个门控。
### 遗忘门
- “遗忘门”决定之前状态中的信息有多少应该舍弃。它会读取 $h_{t-1}$ 和 $x_t$的内容,$\sigma$符号代表Sigmoid函数，它会输出一个0到1之间的值。其中0代表舍弃之前细胞状态$C_{t-1}$中的内容，1代表完全保留之前细胞状态$C_{t-1}$中的内容。0、1之间的值代表部分保留之前细胞状态$C_{t-1}$中的内容。

![rnn](img/forget.png?raw=true "rnn")

在我们 LSTM 中的第一步是决定我们会从细胞状态中丢弃什么信息。这个决定通过一个称为忘记门层完成。该门会读取 h_{t-1} 和 x_t，输出一个在 0 到 1 之间的数值给每个在细胞状态 C_{t-1} 中的数字。1 表示“完全保留”，0 表示“完全舍弃”。
### 输入门

- “输入门”决定什么样的信息保留在细胞状态$C_t$中，它会读取 $h_{t-1}$ 和 $x_t$的内容,$\sigma$符号代表Sigmoid函数，它会输出一个0到1之间的值。
- 和“输入门”配合的还有另外一部分，即下图中计算tanh层的部分，这部分输入也是$h_{t-1}$ 和 $x_t$，不过采用tanh激活函数，将这部分标记为$\tilde c^{(t)}$，称作为“候选状态”。
![rnn](img/input.png?raw=true "rnn")
### 细胞状态更新
- 由$C_{t-1}$ 计算得到$C_t$
- 旧“细胞状态”$C_{t-1}$和“遗忘门”的结果进行计算，决定旧的“细胞状态”保留多少，忘记多少。接着“输入门”$i^{(t)}$和候选状态$\tilde c^{(t)}$进行计算，将所得到的结果加入到“细胞状态”中，这表示新的输入信息有多少加入到“细胞状态中”。
![rnn](img/update.png?raw=true "rnn")
### 输出门
- 和其他门计算一样，它会读取 $h_{t-1}$ 和 $x_t$的内容,然后计算Sigmoid函数，得到“输出门”的值。接着把“细胞状态”通过tanh进行处理(得到一个在-1到1之间的值)，并将它和输出门的结果相乘，最终得到确定输出的部分。
![rnn](img/output.png?raw=true "rnn")

## 参考
[Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/ "title")

[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/ "title")

[Sequence prediction using recurrent neural networks(LSTM) with TensorFlow](http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html "title")




