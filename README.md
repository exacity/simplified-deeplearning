# DeepLearningBook读书笔记

## 前言

&emsp;&emsp;作为人工智能领域目前的最大研究热点，同时也是近年来为各种智能任务带来最大突破的技术方向 – 深度学习或者说神经网络正吸引着无数研究人员的眼球。事实上，传统的神经网络结构和算法早在上个世纪就已经被提出，但由于当时的任务需求仍远未达到传统机器学习算法的瓶颈，同时神经网络算法也受限于计算和数据资源，因此并未被普遍关注。

&emsp;&emsp;近些年来，依靠人工设计高质量特征的传统机器学习算法在语音识别、自然语言处理以及图像处理等方面逐渐达到瓶颈，人们开始将目光重新转向神经网络，利用已经积累的大量数据资源在这一系列智能任务上取得了突破性的进展。包括语音识别、语义理解、图像识别等在内的研究领域中目前state-of-the-art的结果几乎清一色的都是采用了基于深度学习的方法。同时，GPU强大的并行计算能力以及包括[TensorFlow](https://www.tensorflow.org/)、[MXNet](https://mxnet.incubator.apache.org/)、[Pytorch](http://pytorch.org/)等在内的一系列深度学习框架的推出也为研究者和应用开发者提供了极大便利。

&emsp;&emsp;[DeepLearningBook](http://www.deeplearningbook.org/)是目前第一本系统和完整的介绍深度学习的书籍，其作者包括深度学习领域的奠基人、处于研究生涯中期的领域中坚、更有近年来涌现的新星，非常适合搭建理论基础。但是直至去年，本书只有英文原版，对于大多数开发者来说，啃一本800页7*9英寸的书籍，难度可想而知。
好消息是，在翻译人员的不懈努力下，[DeepLearningBook中文版](https://github.com/exacity/deeplearningbook-chinese)也已在GitHub上公开，高质量的中文翻译版已经由人民邮电出版社出版。

&emsp;&emsp;这个项目记录了我们对DeepLearningBook的学习笔记，我们按照全书的脉络对Deep Learning的基础框架进行了梳理和学习，同时将会附上使用TensorFlow实现的相关代码。

>GitHub的markdown不再支持tex公式的解析显示，使用Chrome的同学可以安装[GitHub with MathJax](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima)添加MathJax的解析以对公式正常显示。

## 目录

1. [数学基础](数学基础/数学基础.md)
   1. [线性代数](数学基础/线性代数.md)
   1. [概率与信息论](数学基础/概率与信息论.md)
   1. [数值计算](数学基础/数值计算.md)
1. 机器学习基础与实践
