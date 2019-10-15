---
title: >-
  论文学习-系统评估卷积神经网络各项超参数设计的影响-Systematic evaluation of CNN advances on the
  ImageNet
mathjax: true
date: 2018-11-10 11:23:17
tags:
- CNN
- 综述
- paper
categories:
- 深度学习基础
---


博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

# 写在前面

论文状态：Published in CVIU Volume 161 Issue C, August 2017
论文地址：https://arxiv.org/abs/1606.02228
github地址：https://github.com/ducha-aiki/caffenet-benchmark

在这篇文章中，作者在ImageNet上做了大量实验，**对比卷积神经网络架构中各项超参数选择的影响，对如何优化网络性能很有启发意义**，对比实验包括**激活函数**（sigmoid、ReLU、ELU、maxout等等）、**Batch Normalization (BN)**、**池化方法与窗口大小**（max、average、stochastic等）、**学习率decay策略**（step, square, square root, linear 等）、输入图像颜色空间与预处理、分类器设计、网络宽度、**Batch size**、**数据集大小**、**数据集质量**等等，具体见下图

![Table 1: List of hyper-parameters tested](https://s1.ax1x.com/2018/11/18/izqs1A.png)



实验所用的基础架构（Baseline）从CaffeNet修改而来，有以下几点不同：

 1. 输入图像resize为128（出于速度考虑）
 2. fc6和fc7神经元数量从4096减半为2048
 3. 网络使用[LSUV](https://arxiv.org/abs/1511.06422)进行初始化
 4. 移除了LRN层（对准确率无贡献，出于速度考虑移除）

所有性能比较均以基础架构为Baseline，实验中所有超参数调整也都是在Baseline上进行，Baseline accuracy为47.1%，Baseline网络结构如下

![Baseline Network](https://s1.ax1x.com/2018/11/18/izq6Xt.png)

# 论文实验结论
论文通过**控制变量**的方式进行实验，最后给出了如下建议：

 - **不加 BN时使用 ELU，加BN时使用ReLU**（加BN时，两者其实差不多）
 - 对输入RGB图学习一个颜色空间变换，再接网络
 - 使用linear decay学习策略
 - **池化层将average与max求和**
 - BatchSize使用128或者256，如果GPU内存不够大，在调小BatchSize的同时同比减小学习率
 - **用卷积替换全连接层，在最后决策时对输出取平均**
 - 当决定要扩大训练集前，先查看是否到了“平坦区”——即**评估增大数据集能带来多大收益**
 - **数据清理比增大数据集更重要**
 - 如果不能提高输入图像的大小，减小隐藏层的stride有近似相同的效果
 - 如果网络结构复杂且高度优化过，如GoogLeNet，做修改时要小心——即将**上述修改在简单推广到复杂网络时不一定有效**

需要注意的是，在Batch Size和学习率中，文章仅做了两个实验，一个是固定学习调整BatchSize，另一个学习率与Batch Size同比增减，但两者在整个训练过程中的Batch Size都保持不变，在这个条件下得出了 **学习率与Batch Size同比增减 策略是有效的**结论。最近Google有一篇文章[**《Don't Decay the Learning Rate, Increase the Batch Size》**](https://arxiv.org/abs/1711.00489)提出了在训练过程中逐步增大Batch Size的策略。

论文实验量非常大，每项实验均通过控制变量测试单一或少数因素变化的影响，相当于通过贪心方式一定意义上获得了每个局部最优的选择，最后将所有局部最优的选择汇总在一起仍极大地改善了性能（但**不意味着找到了所有组合中的最优选择**）。实验结果主要是在CaffeNet（改）上的得出的，并不见得能推广到所有其他网络。

但是，总的来讲，本篇文章做了很多笔者曾经想过但“没敢”做的实验，实验结果还是很有启发意义的，值得一读。

文章全部实验汇总如下，[github](https://github.com/ducha-aiki/caffenet-benchmark)上有更多实验结果：
![Results of all tests on ImageNet-128px](https://s1.ax1x.com/2018/11/18/izq20f.png)


# 论文细节

一图胜千言，本节主要来自论文图表。

## 激活函数

![Activation functions](https://s1.ax1x.com/2018/11/18/izqR78.png)

 在计算复杂度与ReLU相当的情况下，**ELU的单一表现最好，ELU（卷积后）+maxout（全连接后）联合表现最好**，前者提升约2个百分点，后者约4个百分点。值得注意的是，**不使用非线性激活函数时，性能down了约8个百分点，并非完全不能用。**

## 池化

![Pooling](https://s1.ax1x.com/2018/11/18/izqfAS.png)

 方法上，**max和average池化结合取得最好效果**（结合方式为 element-wise 相加），作者推测是因为同时具备了max的选择性和average没有扔掉信息的性质。尺寸上，**在保证输出尺寸一样的情况下，non-overlapping优于overlapping——前者的kernel size更大**。


## 学习率
![Learning rate policy](https://s1.ax1x.com/2018/11/18/izq4hQ.png)
 ![Learning rate policy ](https://s1.ax1x.com/2018/11/18/izqIpj.png)

 **linear decay取得最优效果**。

## BatchSize与学习率
![Batch size and initial learning rate impact to the accuracy](https://s1.ax1x.com/2018/11/18/izqbn0.png)

 文章中仅实验了固定学习调整BatchSize以及学习率与Batch Size同比增减两个实验，在整个训练过程中Batch Size保持不变，得出了 **学习率与Batch Size同比增减 策略是有效的**结论。

##  图像预处理

![learned colorspace transformations](https://s1.ax1x.com/2018/11/18/izqqBV.png)

![performance of using various colorspaces and pre-processing](https://s1.ax1x.com/2018/11/18/izqL7T.png) 

灰度及其他颜色空间均比RGB差，**通过两层1x1卷积层将RGB图映射为新的3通道图取得了最好效果**。

## BN层
![batch normalization](https://s1.ax1x.com/2018/11/18/izqjNF.png)

![Top-1 accuracy gain over ReLU without BN](https://s1.ax1x.com/2018/11/18/izqvh4.png)

**Sigmoid + BN 好于 ReLU无BN，当然，ReLU+BN更好。**

## 分类器设计
![Classier design](https://s1.ax1x.com/2018/11/18/izqz9J.png)

若将CNN网络拆成两个部分，前为特征提取，后为分类器。分类器部分一般有3种设计：

 1. **特征提取最后一层为max pooling，分类器为一层或两层全连接层**，如LeNet、AlexNet、VGGNet
 2. **使用spacial pooling代替max pooling，分类器为一层或两层全连接层**
 3. **使用average pooling，直接连接softmax，无全连接层**，如GoogLeNet、ResNet

作者实验发现，将全连接替换为卷积层（允许zero padding），经过softmax，最后average pooling，即Pool5-C3-C1-CLF-AvePool取得了最好效果。

## 网络宽度
![Network width impact on the accuracy](https://s1.ax1x.com/2018/11/18/izLS39.png)

对文章采用的基础网络，增大网络宽度，性能会提升，但增大超过3倍后带来的提升就十分有限了，即对某个特定的任务和网络架构，存在某个适宜的网络宽度。

## 输入图像大小

![Input image size impact on the accuracy](https://s1.ax1x.com/2018/11/18/izLpcR.png)

准确率随图像尺寸线性增长，但计算量是平方增长。如果不能提高输入图像的大小，减小隐藏层的stride有近似相同的效果。

## Dataset size and noisy labels

![Training dataset size and cleanliness impact on the accuracy](https://s1.ax1x.com/2018/11/18/izL9j1.png)

增大数据集可以改善性能，数据清理也可改善性能，但**数据清理比数据集大小更重要**，为了获得同样的性能，有错误标签的数据集需要更大。

## Bias有无的影响

![Influence of the bias](https://s1.ax1x.com/2018/11/18/izLPnx.png)

卷积层和全连接层无Bias比有Bias降了2.6个百分点。

## 改善项汇总
将 学到的颜色空间变换、ELU作为卷积层激活函数、maxout作为全连接层激活函数、linear decay学习率策略、average+max池化 结合使用，在CaffeNet、VGGNet、GoogLeNet上对比实验，如下：

![Best-of-all experiments](https://s1.ax1x.com/2018/11/18/izLiB6.png)

CaffeNet和VGGNet的表现均得以改善，GoogLeNet则不是，**对于复杂且高度优化过的网络，一些改进策略不能简单推广**。


# 参考
- [paper-Systematic evaluation of CNN advances on the ImageNet](https://arxiv.org/abs/1606.02228)
- [github-Systematic evaluation of CNN advances on the ImageNet](https://github.com/ducha-aiki/caffenet-benchmark)[图片上传失败...(image-eef9ec-1542507708571)]
