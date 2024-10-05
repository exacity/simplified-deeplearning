# 15. 表示学习

## [15.1 贪心逐层无监督预训练(Greedy Layer-Wise Unsupervised Pretraining)](./note/src/15.表示学习.md#151-贪心逐层无监督预训练greedy-layer-wise-unsupervised-pretraining)

### 15.1 代码

比较现代**监督学习方法**(CNN,[卷积神经网络](https://zh.wikipedia.org/zh-sg/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C))和**无监督学习方法**(VAE,[变分自编码器](https://zh.wikipedia.org/wiki/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8))在**MNIST数据集**上的训练效果。前者训练效果可通过测试集准确率评估，后者通常依赖于对重建图像的视觉评估。

#### 监督学习：[CNN](./code/MNIST_CNN.py)

#### 无监督学习：[VAE](./code/MNIST_Variational_AutoEncoder.py)

## [15.2 迁移学习(Transfer learning)和领域自适应(Domain adaption)](./note/src/15.表示学习.md#152-迁移学习transfer-learning和领域自适应domain-adaption)

## [15.3 半监督解释因果关系](./note/src/15.表示学习.md#153-半监督解释因果关系)

### 15.3 代码

- [生成对抗网络(GAN)](./code/GAN.py)

## [15.4 分布式表示](./note/src/15.表示学习.md#154-分布式表示)

## [15.5 得益于深度的指数增益(深度学习模型的优势) & 15.6 提供发现潜在原因的线索(正则化策略)](./note/src/15.表示学习.md#155-得益于深度的指数增益深度学习模型的优势--156-提供发现潜在原因的线索线索正则化策略)

**全部代码来源均已在首行注释标注。**
