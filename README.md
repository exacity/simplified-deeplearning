# DeepLearningBook

&emsp;&emsp;作为人工智能领域目前的最大研究热点，同时也是近年来为各种智能任务带来最大突破的技术方向 – 深度学习或者说神经网络正吸引着无数研究人员的眼球。事实上，传统的神经网络结构和算法早在上个世纪就已经被提出，但由于当时的任务需求仍远未达到传统机器学习算法的瓶颈，同时神经网络算法也受限于计算和数据资源，因此并未被普遍关注。

&emsp;&emsp;近些年来，依靠人工设计高质量特征的传统机器学习算法在语音识别、自然语言处理以及图像处理等方面逐渐达到瓶颈，人们开始将目光重新转向神经网络，利用已经积累的大量数据资源在这一系列智能任务上取得了突破性的进展。包括语音识别、语义理解、图像识别等在内的研究领域中目前state-of-the-art的结果几乎清一色的都是采用了基于深度学习的方法。同时，GPU强大的并行计算能力以及包括[TensorFlow](https://www.tensorflow.org/)、[MXNet](https://mxnet.incubator.apache.org/)、[Pytorch](http://pytorch.org/)等在内的一系列深度学习框架的推出也为研究者和应用开发者提供了极大便利。

&emsp;&emsp;为此，星小环的AI-er经常被开发者问及，那么我们开发者如何开始学习使用深度学习技术呢。面对琳琅满目的各种blog和教程，如何能够系统地、理论地又能够和实践相结合来学习和理解呢。向来在重积累重技术重底层的小环AI-er一直推荐的是Bengio等人著作[DeepLearningBook](http://www.deeplearningbook.org/)。

![deeplearningbook](img/deeplearningbook.png)

&emsp;&emsp;[DeepLearningBook](http://www.deeplearningbook.org/)是目前第一本系统和完整的介绍深度学习的书籍，其作者包括深度学习领域的奠基人、处于研究生涯中期的领域中坚、更有近年来涌现的新星，非常适合搭建理论基础。

&emsp;&emsp;但是直至去年，本书只有英文原版，对于大多数开发者来说，啃一本800页7*9英寸的书籍，难度可想而知。而且而且英文原版动辄100美元的价格都可以买一个好键盘了。好消息是，国内AI开发者有福，高质量的中文翻译版已经由人民邮电出版社出版。赶紧去某东某当买中文版和星小环的AI-er一起研读吧。中文版价格也良心。[DeepLearningBook中文版](https://github.com/exacity/deeplearningbook-chinese)pdf版本在github有下载。

&emsp;&emsp;但是又有开发者问，这本书都是公式和理论啊，我们如何实践呢？小环的回答是：加入读书会。我们将按照章节先做读书理论分享，同时每一章节介绍会在小环机器学习平台上进行Python [TensorFlow](https://www.tensorflow.org/)实现，增加大家的理解。比如RNN，LSTM章节，我们除了提供Tensorflow例子之外，将满足之前多位开发者提的一个夙愿，如何从零搭建一个初级写唐诗的AI-er. CNN章节理论学习完，将开发一个自己的风格画。

&emsp;&emsp;本系列星小环AI读书会——深度学习将于2017年8月28日开始，每周三晚上7点至9点，在星小环AI实验室微信群里面直播。入群更可以近距离和小环AI-er团队（拥有数学系博士，机器学习方向博士、UIUC，普渡等多位海归，ICDM等顶级会议作者，等）直接沟通，该群接纳深度学习书籍的答疑讨论，不过小环AI-er工作时间会响应滞后。

&emsp;&emsp;怎么入群？怎么入群？ 欢迎评论区留言微信号申请入群，小环工作人员会联系你，审核并邀请入群。没有入群的也可以持续关注本系列，每周会更新读书会内容和代码

> GitHub的markdown不再支持tex公式的解析显示，使用Chrome的同学可以安装[GitHub with MathJax](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima)添加MathJax的解析以对公式正常显示。

## 目录

1. [预备知识](预备知识/数学基础.md)
   1. [数学基础](预备知识/数学基础.md)
   1. [机器学习基础](预备知识/机器学习基础.md)
1. 深度前馈网络
