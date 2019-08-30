---
title: ImageNet主要网络benchmark对比
mathjax: false
date: 2019-08-28 19:46:23
tags:
- 综述
- paper
categories:
- backbone网络
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

深度神经网络繁多，各自的性能指标怎样？
实际应用中，在速度、内存、准确率等各种约束下，应该尝试哪些模型作为backbone？

有paper对各个网络模型进行了对比分析，形成了一个看待所有主要模型的完整视角，其分析结果可以在实践中提供指导和帮助。

这篇博客主要整合了其中3篇文章的结论，分别是
1. [201605-An Analysis of Deep Neural Network Models for Practical Applications](https://arxiv.org/abs/1605.07678)
2. [201809-Analysis of deep neural networks](https://medium.com/@culurciello/analysis-of-deep-neural-networks-dcf398e71aae)
3. [201810-Benchmark Analysis of Representative Deep Neural Network Architectures](https://arxiv.org/abs/1810.00736)

文章1和3是paper，2是篇博客（对1的更新）。这3篇文章对图像识别任务（ImageNet-1k）主要的state of the art网络进行了对比分析，采用的指标有：

 - **accuracy**，准确率，只使用cental crop，评估Top1、Top5在ImageNet-1k上的准确率
 - **model complexity**，模型复杂度，通过模型的可学习参数量衡量（近似为模型文件大小），反映了自由度
 - **computational complexity**，计算复杂度，操作次数，通过floating-point operations (FLOPs)衡量，Multiply-add乘加运算为2 FLOPS
 - **memory usage**，内存大小（空间复杂度）
 - **inference time**，推理时间
 - **accuracy density**，等于 accuracy / modle size，用来衡量**参数的利用效率**

比较重要的结论有：
 - 计算复杂度高，识别准确率不一定高；参数量大，识别准确率也不一定高。——**好的网络结构设计很重要**，比如ResNet系的模型。
 - **不同模型的参数利用效率不同**，目前来看针对移动端设计的网络参数利用效率较高，如MobileNet、ShuffleNet、SqueezeNet等，但在Top1准确率高于80%的模型中，Inception-V4和SE-ResNeXt-101的利用率较高
 - 操作次数（FLOPs）是推理时间的良好估计
 - 为了满足不同的内存和速度要求，可选的最优模型不同

其他一些更细致的结论可以参看论文，下面贴一下论文中的重要图表。

论文[An Analysis of Deep Neural Network Models for Practical Applications](https://arxiv.org/abs/1605.07678)诞生于2016年5月，文中对当时的主要模型（从AlexNet到Inception-v4）进行了对比分析，得到了那张流传甚广的ball chart。后来在2018年9月，文章作者Eugenio Culurciello在博客[Analysis of deep neural networks](https://medium.com/@culurciello/analysis-of-deep-neural-networks-dcf398e71aae)中，对图表进行了更新，包括了Shufflenet、Mobilenet、Xception、Densenet、Squeezenet等新近模型的对比分析，更新的ball chart如下：

![Top1 vs. operations, size ∝ parameters](https://s2.ax1x.com/2019/08/28/mTpCJs.png)
图中，blob的中心为模型在图表中的位置，blob的大小对应模型的参数量，横轴为操作次数，纵轴为Top-1 center crop的准确率，**越靠近左上角的模型计算复杂度越低、准确率越高，blob越小的模型参数越少**。

论文[An Analysis of Deep Neural Network Models for Practical Applications](https://arxiv.org/abs/1605.07678)中，推理时间和操作数的关系图表如下，不出意料的正相关
![Operations vs. inference time](https://s2.ax1x.com/2019/08/28/m75MHx.png)

论文[Benchmark Analysis of Representative Deep Neural Network Architectures](https://arxiv.org/abs/1810.00736)中，做了更详细的对比，如下图所示，**左上角ResNet系的模型表现强劲，右上角NASNet-A-Large的准确率最高但计算复杂度也最大**：
![accuracy vs computational complexity](https://s2.ax1x.com/2019/08/28/m7r4UA.png)

参数利用率如下：
![accuracy density and Top-1 accuracy](https://s2.ax1x.com/2019/08/28/m7ytw4.png)

速度（帧率）与准确率如下，图中的曲线为特定硬件下帧率与性能的上界，横轴为帧率的对数，
![Top-1 accuracyvs.number of images processed per second (with batch size 1) ](https://s2.ax1x.com/2019/08/28/m7RHFe.png)

模型参数量与内存占用大小如下，GPU上内存占用最少的也在0.6G以上，
![model size vs memory](https://s2.ax1x.com/2019/08/28/m7hEZV.png)

对于每个网络具体的推理时间和内存占用情况可以参见论文原文，有更详细的描述。

给定硬件平台上，在不同内存和速度约束下的最优模型如下：
![Top 5 models](https://s2.ax1x.com/2019/08/28/m7oifU.png)
[Benchmark Analysis of Representative Deep Neural Network Architectures](https://arxiv.org/abs/1810.00736)的代码基于pytorch，详见[models-comparison.pytorch](https://github.com/CeLuigi/models-comparison.pytorch)。

# 参考
- [Benchmark Analysis of Representative Deep Neural Network Architectures](https://arxiv.org/abs/1810.00736)
- [models-comparison.pytorch](https://github.com/CeLuigi/models-comparison.pytorch)
- [Analysis of deep neural networks](https://medium.com/@culurciello/analysis-of-deep-neural-networks-dcf398e71aae)
- [An Analysis of Deep Neural Network Models for Practical Applications](https://arxiv.org/abs/1605.07678)
- [torch-opCounter](https://github.com/apaszke/torch-opCounter)