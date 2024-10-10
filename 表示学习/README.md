# 15. 表示学习

笔记见[这里](./note/15.表示学习.md)。

## 15.1 贪心逐层无监督预训练(Greedy Layer-Wise Unsupervised Pretraining)

比较现代**监督学习方法**(CNN,**卷积神经网络**)和**无监督学习方法**(VAE,**变分自编码器**)在**MNIST数据集**上的训练效果。前者训练效果可通过测试集准确率评估，后者通常直接依赖于对重建图像的视觉评估。

代码如下。

### 监督学习：[CNN](./code/MNIST_CNN.py)

### 无监督学习：[VAE](./code/MNIST_Variational_AutoEncoder.py)

## 15.2 迁移学习(Transfer learning)和领域自适应(Domain adaption)

## 15.3 半监督解释因果关系

代码：[生成对抗网络(GAN)](./code/GAN.py)

## 15.4 分布式表示

## 15.5 得益于深度的指数增益(深度学习模型的优势) & 15.6 提供发现潜在原因的线索(正则化策略)

**全部代码来源均已在首行注释标注。**
