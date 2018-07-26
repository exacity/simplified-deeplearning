# DeepLearningBook读书笔记

## 前言

&emsp;&emsp;作为人工智能领域目前的最大研究热点，同时也是近年来为各种智能任务带来最大突破的技术方向 – 深度学习或者说神经网络正吸引着无数研究人员的眼球。事实上，传统的神经网络结构和算法早在上个世纪就已经被提出，但由于当时的任务需求仍远未达到传统机器学习算法的瓶颈，同时神经网络算法也受限于计算和数据资源，因此并未被普遍关注。

&emsp;&emsp;近些年来，依靠人工设计高质量特征的传统机器学习算法在语音识别、自然语言处理以及图像处理等方面逐渐达到瓶颈，人们开始将目光重新转向神经网络，利用已经积累的大量数据资源在这一系列智能任务上取得了突破性的进展。包括语音识别、语义理解、图像识别等在内的研究领域中目前state-of-the-art的结果几乎清一色的都是采用了基于深度学习的方法。同时，GPU强大的并行计算能力以及包括[TensorFlow](https://www.tensorflow.org/)、[MXNet](https://mxnet.incubator.apache.org/)、[Pytorch](http://pytorch.org/)等在内的一系列深度学习框架的推出也为研究者和应用开发者提供了极大便利。

&emsp;&emsp;[DeepLearningBook](http://www.deeplearningbook.org/)是目前第一本系统和完整的介绍深度学习的书籍，其作者包括深度学习领域的奠基人、处于研究生涯中期的领域中坚、更有近年来涌现的新星，非常适合搭建理论基础。但是直至去年，本书只有英文原版，对于大多数开发者来说，啃一本800页7*9英寸的书籍，难度可想而知。
好消息是，在翻译人员的不懈努力下，[DeepLearningBook中文版](https://github.com/exacity/deeplearningbook-chinese)也已在GitHub上公开，中文翻译版已经由人民邮电出版社出版。

&emsp;&emsp;这个项目记录了我们对DeepLearningBook的学习笔记，我们按照全书的脉络对Deep Learning的基础框架进行了梳理和学习，同时将会附上使用TensorFlow实现的相关代码。

>GitHub的markdown不再支持tex公式的解析显示，使用Chrome的同学可以安装[GitHub with MathJax](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima)添加MathJax的解析以对公式正常显示。

>如果需要直接阅读模式，可以移步至我们的github.io进行阅读：[DeepLearningBook读书笔记](https://discoverml.github.io/simplified-deeplearning/)

## 目录

1. [数学基础](数学基础/README.md)
    1. [线性代数](数学基础/线性代数.md)
    1. [概率与信息论](数学基础/概率与信息论.md)
    1. [数值计算](数学基础/数值计算.md)
1. [机器学习基础与实践](机器学习基础与实践/README.md)
    1. [机器学习基础](机器学习基础与实践/机器学习基础.md)
    1. [TensorFlow实战](机器学习基础与实践/TensorFlow实战.md)
1. [深度前馈网络](深度前馈网络/README.md)
1. [深度学习中的正则化](深度学习中的正则化/README.md)
1. [深度学习中的优化](深度学习中的优化/README.md)
1. [卷积网络](卷积网络/README.md)
    1. [简单卷积网络示例](卷积网络/简单卷积网络.md)
    1. [经典CNN模型(LeNet and AlexNet)](卷积网络/卷积网络进阶.ipynb)
    1. [GoogLeNet](卷积网络/GoogLeNet.ipynb)
    1. [ResNet](卷积网络/ResNet.ipynb)
1. [循环递归网络](循环递归网络/README.md)
    1. [RNN示例](循环递归网络/RNN.md)
    1. [LSTM](循环递归网络/LSTM.md)
    1. [基于CharRNN的古诗生成](循环递归网络/poetry-charRNN.ipynb)
        <!-- 1. [序列到序列学习](循环递归网络/Sequence.md) -->
1. [实践调参](实践调参/README.md)
1. [线性因子模型](线性因子模型/README.md)
1. [自编码器](自编码器/README.md)
1. [表示学习](表示学习/README.md)
1. [结构化概率模型](结构化概率模型/README.md)
1. [蒙特卡洛方法](蒙特卡洛方法/README.md)
1. [玻尔兹曼机](玻尔兹曼机/README.md)
1. [有向生成网络](有向生成网络)
1. [生成对抗网络](生成对抗网络/README.md)


>持续更新中，欢迎贡献简单易懂便于理解的代码示例，推荐使用Tensorflow和Jupyter Notebook提交代码和说明，详见：[如何贡献代码](pending/README.md)。

致谢
--------------------
我们分为两个类别的贡献者。
 - 负责人也就是对应的该章节案例维护者。
 - 贡献者对应于主要的案例开发者。

| 原书章节 | 对应案例  | 负责人 | 贡献者 |
| ------------ | ------------ | ------------ | ------------ |
| [第一章 前言](https://exacity.github.io/deeplearningbook-chinese/Chapter1_introduction/) | [前言介绍](README.md) | @swordyork | @daweicheng |
| [第二章 线性代数](https://exacity.github.io/deeplearningbook-chinese/Chapter2_linear_algebra/) | [线性代数](数学基础/线性代数.md) | @zengxy | |
| [第三章 概率与信息论](https://exacity.github.io/deeplearningbook-chinese/Chapter3_probability_and_information_theory/) | [概率与信息论](数学基础/概率与信息论.md) | @zengxy |  |
| [第四章 数值计算](https://exacity.github.io/deeplearningbook-chinese/Chapter4_numerical_computation/) | [数值计算](数学基础/数值计算.md) | @zengxy |  |
| [第五章 机器学习基础](https://exacity.github.io/deeplearningbook-chinese/Chapter5_machine_learning_basics/) |[机器学习基础与实践](机器学习基础与实践/README.md) |@zengxy  | @fangjie  |
| [第六章 深度前馈网络](https://exacity.github.io/deeplearningbook-chinese/Chapter6_deep_feedforward_networks/) | [深度前馈网络](深度前馈网络/README.md) | @kimliu0803 | @hjptriplebee @fangjie  |
| [第七章 深度学习中的正则化](https://exacity.github.io/deeplearningbook-chinese/Chapter7_regularization/) | [深度学习中的正则化](深度学习中的正则化/README.md) | @lupeng666 | @titicaca |
| [第八章 深度模型中的优化](https://exacity.github.io/deeplearningbook-chinese/Chapter8_optimization_for_training_deep_models/) | [深度学习中的优化](深度学习中的优化/README.md) | @jinshengwang92 | @lupeng666  |
| [第九章 卷积网络](https://exacity.github.io/deeplearningbook-chinese/Chapter9_convolutional_networks/) | [卷积网络](卷积网络/README.md) | @LiuCheng|  |
| [第十章 序列建模：循环和递归网络](https://exacity.github.io/deeplearningbook-chinese/Chapter10_sequence_modeling_rnn/) | [循环递归网络](循环递归网络/README.md) | @zengxy | @hjptriplebee |
| [第十一章 实践方法论](https://exacity.github.io/deeplearningbook-chinese/Chapter11_practical_methodology/) |[实践调参](实践调参/README.md)  | @daweicheng |  |
| [第十二章 应用](https://exacity.github.io/deeplearningbook-chinese/Chapter12_applications/) |  | |  |
| [第十三章 线性因子模型](https://exacity.github.io/deeplearningbook-chinese/Chapter13_linear_factor_models/) | [线性因子模型](线性因子模型/README.md) | @liqi | @YaoStriveCode |
| [第十四章 自编码器](https://exacity.github.io/deeplearningbook-chinese/Chapter14_autoencoders/) | [自编码器](自编码器/README.md) | @daweicheng |  |
| [第十五章 表示学习](https://exacity.github.io/deeplearningbook-chinese/Chapter15_representation_learning/) | [表示学习](表示学习/README.md) |@daweicheng  | |
| [第十六章 深度学习中的结构化概率模型](https://exacity.github.io/deeplearningbook-chinese/Chapter16_structured_probabilistic_modelling/) |[结构化概率模型](结构化概率模型/README.md) | @xuanming |
| [第十七章 蒙特卡罗方法](https://exacity.github.io/deeplearningbook-chinese/Chapter17_monte_carlo_methods/) | [蒙特卡洛方法](蒙特卡洛方法/README.md) | @xuanming |   |
| [第十八章 面对配分函数](https://exacity.github.io/deeplearningbook-chinese/Chapter18_confronting_the_partition_function/) |  | |  |
| [第十九章 近似推断](https://exacity.github.io/deeplearningbook-chinese/Chapter19_approximate_inference/) |  | | |
| [第二十章 深度生成模型](https://exacity.github.io/deeplearningbook-chinese/Chapter20_deep_generative_models/) |[玻尔兹曼机](玻尔兹曼机/README.md)<br> [有向生成网络](有向生成网络)<br> [生成对抗网络](生成对抗网络) | @vistep <br>@daweicheng<br>@swordyork | |
| 参考文献 | | |  |



还有很多同学提出了不少建议，我们都列在此处。

@endymecy ...

如有遗漏，请务必通知我们，可以发邮件至`echo c3dvcmQueW9ya0BnbWFpbC5jb20K | base64 --decode`。
这是我们必须要感谢的，所以不要不好意思。