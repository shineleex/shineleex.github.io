---
title: Batch Normalization详解
mathjax: true
date: 2019-11-27 15:35:08
tags:
categories:
- 深度学习基础
---



博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

# 动机

在博文《**为什么要做特征归一化/标准化？** [博客园](https://www.cnblogs.com/shine-lee/p/11779514.html) | [csdn](https://blog.csdn.net/blogshinelee/article/details/102875044) | [blog](https://blog.shinelee.me/2019/10-22-%E4%B8%BA%E4%BB%80%E4%B9%88%E9%9C%80%E8%A6%81%E7%89%B9%E5%BE%81%E5%BD%92%E4%B8%80%E5%8C%96or%E6%A0%87%E5%87%86%E5%8C%96%EF%BC%9F.html)》中，我们介绍了对输入进行Standardization后，梯度下降算法更容易选择到合适的（较大的）学习率，下降过程会更加稳定。

在博文《**网络权重初始化方法总结（下）：Lecun、Xavier与He Kaiming** [博客园](https://www.cnblogs.com/shine-lee/p/11908610.html) | [csdn](https://blog.csdn.net/firelx/article/details/103194358) | [blog](https://blog.shinelee.me/2019/11-11-%E7%BD%91%E7%BB%9C%E6%9D%83%E9%87%8D%E5%88%9D%E5%A7%8B%E5%8C%96%E6%96%B9%E6%B3%95%E6%80%BB%E7%BB%93%EF%BC%88%E4%B8%8B%EF%BC%89%EF%BC%9ALecun%E3%80%81Xavier%E4%B8%8EHe%20Kaiming.html)》中，我们介绍了如何通过权重初始化让网络在训练之初保持激活层的输出（输入）为zero mean unit variance分布，以减轻梯度消失和梯度爆炸。

但在训练过程中，权重在不断更新，导致激活层输出(输入)的分布会一直变化，可能无法一直保持zero mean unit variance分布，还是有梯度消失和梯度爆炸的可能，直觉上感到，这可能是个问题。下面具体分析。

## 单层视角

![http://gradientscience.org/batchnorm/](https://s2.ax1x.com/2019/12/03/QMURNF.png)

神经网络可以看成是上图形式，对于中间的某一层，其前面的层可以看成是对输入的处理，后面的层可以看成是损失函数。一次反向传播过程会同时更新所有层的权重$W_1, W_2, \dots, W_L$，前面层权重的更新会改变当前层输入的分布，**而跟据反向传播的计算方式，我们知道，对$W_k$的更新是在假定其输入不变的情况下进行的**。如果假定第$k$层的输入节点只有2个，对第$k$层的某个输出节点而言，相当于一个线性模型$y = w_1 x_1 + w_2 x_2 + b$，如下图所示，

<img src="https://s2.ax1x.com/2019/12/03/QMwSTf.png" alt="https://wiki2.org/en/Linear_classifier#/media/File:Svm_separating_hyperplanes.png" style="zoom: 50%;" />

假定当前输入$x_1$和$x_2$的分布如图中圆点所示，本次更新的方向是将直线$H_1$更新成$H_2$，本以为切分得不错，但是当前面层的权重更新完毕，当前层输入的分布换成了另外一番样子，直线相对输入分布的位置可能变成了$H_3$，下一次更新又要根据新的分布重新调整。**直线调整了位置，输入分布又在发生变化，直线再调整位置，就像是直线和分布之间的“追逐游戏”。对于浅层模型，比如SVM，输入特征的分布是固定的，即使拆分成不同的batch，每个batch的统计特性也是相近的，因此只需调整直线位置来适应输入分布，显然要容易得多。而深层模型，每层输入的分布和权重在同时变化，训练相对困难。**

## 多层视角

上面是从网络中单拿出一层分析，下面看一下多层的情况。在反向传播过程中，每层权重的更新是在假定其他权重不变的情况下，向损失函数降低的方向调整自己。问题在于，在一次反向传播过程中，所有的权重会同时更新，导致层间配合“缺乏默契”，每层都在进行上节所说的“追逐游戏”，而且层数越多，相互配合越困难，文中把这个现象称之为 **Internal Covariate Shift**，示意图如下。为了避免过于震荡，学习率不得不设置得足够小，足够小就意味着学习缓慢。

![https://gab41.lab41.org/batch-normalization-what-the-hey-d480039a9e3b](https://s2.ax1x.com/2019/12/04/QQqCxe.png)

为此，希望对每层输入的分布有所控制，于是就有了**Batch Normalization**，其出发点是对每层的输入做Normalization，只有一个数据是谈不上Normalization的，所以是对一个batch的数据进行Normalization。

# 什么是Batch Normalization

Batch Normalization，简称BatchNorm或BN，翻译为“批归一化”，是神经网络中一种特殊的层，如今已是各种流行网络的标配。**在原paper中，BN被建议插入在（每个）ReLU激活层前面**，如下所示，

![http://gradientscience.org/batchnorm/](https://s2.ax1x.com/2019/12/02/QnyE24.png)

如果batch size为$m$，则在前向传播过程中，网络中每个节点都有$m$个输出，所谓的Batch Normalization，就是对该层每个节点的这$m$个输出进行归一化再输出，具体计算方式如下，

![Batch Normalization Transform](https://s2.ax1x.com/2019/12/02/QngERS.png)

其操作可以分成2步，

1. **Standardization**：首先对$m$个$x$进行 Standardization，得到 zero mean unit variance的分布$\hat{x}$。
2. **scale and shift**：然后再对$\hat{x}$进行scale and shift，缩放并平移到新的分布$y$，具有新的均值$\beta$方差$\gamma$。

假设BN层有$d$个输入节点，则$x$可构成$d \times m$大小的矩阵$X$，BN层相当于通过**行操作**将其映射为另一个$d\times m$大小的矩阵$Y$，如下所示，

![Batch Normalization](https://s2.ax1x.com/2019/12/05/Q8e7jO.png)将2个过程写在一个公式里如下，
$$
y_i^{(b)} = BN\left(x_{i}\right)^{(b)}=\gamma \cdot\left(\frac{x_{i}^{(b)}-\mu\left(x_{i}\right)}{\sqrt{\sigma\left(x_{i}\right)^2 + \epsilon}}\right)+\beta
$$
其中，$x_i^{(b)}$表示输入当前batch的$b$-th样本时该层$i$-th输入节点的值，$x_i$为$[x_i^{(1)}, x_i^{(2)}, \dots, x_i^{(m)}]$构成的行向量，长度为batch size $m$，$\mu$和$\sigma$为该行的均值和标准差，$\epsilon$为防止除零引入的极小量（可忽略），$\gamma$和$\beta$为该行的scale和shift参数，可知

- $\mu$和$\sigma$为当前行的统计量，不可学习。
- $\gamma$和$\beta$为待学习的scale和shift参数，用于控制$y_i$的方差和均值。
- BN层中，$x_i$和$x_j$之间不存在信息交流$(i \neq j)$

可见，**无论$x_i$原本的均值和方差是多少，通过BatchNorm后其均值和方差分别变为待学习的$\beta$和$\gamma$。**

# Batch Normalization的反向传播

对于目前的神经网络计算框架，**一个层要想加入到网络中，要保证其是可微的，即可以求梯度**。BatchNorm的梯度该如何求取？

反向传播求梯度只需抓住一个关键点，如果一个变量对另一个变量有影响，那么他们之间就存在偏导数，找到直接相关的变量，再配合链式法则，公式就很容易写出了。

![BN 反向传播](https://s2.ax1x.com/2019/12/13/QgMPjs.png)

根据反向传播的顺序，首先求取损失$\ell$对BN层输出$y_i$的偏导$\frac{\partial \ell}{\partial y_{i}}$，然后是对可学习参数的偏导$\frac{\partial \ell}{\partial \gamma}$和$\frac{\partial \ell}{\partial \beta}$，用于对参数进行更新，想继续回传的话还需要求对输入 $x$偏导，于是引出对变量$\mu$、$\sigma^2$和$\hat{x}$的偏导，根据链式法则再求这些变量对$x$的偏导。

在实际实现时，通常以矩阵或向量运算方式进行，比如逐元素相乘、沿某个axis求和、矩阵乘法等操作，具体可以参见[Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)和[BatchNorm in Caffe](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/batch_norm_layer.cpp#L169)。

# Batch Normalization的预测阶段

在预测阶段，所有参数的取值是固定的，对BN层而言，意味着$\mu$、$\sigma$、$\gamma$、$\beta$都是固定值。

$\gamma$和$\beta$比较好理解，随着训练结束，两者最终收敛，预测阶段使用训练结束时的值即可。

对于$\mu$和$\sigma$，在训练阶段，它们为当前mini batch的统计量，随着输入batch的不同，$\mu$和$\sigma$一直在变化。在预测阶段，输入数据可能只有1条，该使用哪个$\mu$和$\sigma$，或者说，每个BN层的$\mu$和$\sigma$该如何取值？**可以采用训练收敛最后几批mini batch的 $\mu$和$\sigma$的期望，作为预测阶段的$\mu$和$\sigma$，**如下所示，

![Training a Batch-Normalized Network](https://s2.ax1x.com/2019/12/03/QMlxc8.png)

因为Standardization和scale and shift均为线性变换，在预测阶段所有参数均固定的情况下，参数可以合并成$y=kx+b$的形式，如上图中行号11所示。

# Batch Normalization的作用

使用Batch Normalization，可以获得如下好处，

- **可以使用更大的学习率**，训练过程更加稳定，极大提高了训练速度。
- **可以将bias置为0**，因为Batch Normalization的Standardization过程会移除直流分量，所以不再需要bias。
- **对权重初始化不再敏感**，通常权重采样自0均值某方差的高斯分布，以往对高斯分布的方差设置十分重要，有了Batch Normalization后，对与同一个输出节点相连的权重进行放缩，其标准差$\sigma$也会放缩同样的倍数，相除抵消。
- **对权重的尺度不再敏感**，理由同上，尺度统一由$\gamma$参数控制，在训练中决定。
- **深层网络可以使用sigmoid和tanh了**，理由同上，BN抑制了梯度消失。
- **Batch Normalization具有某种正则作用，不需要太依赖dropout，减少过拟合**。

# 几个问题

## 卷积层如何使用BatchNorm？

> For convolutional layers, we additionally want the normalization to obey the convolutional property – **so that different elements of the same feature map, at different locations, are normalized in the same way.** To achieve this, we jointly normalize all the activations in a mini-batch, over all locations.
>
> ...
>
> so for a mini-batch of size m and feature maps of size p × q, we use the effective mini-batch of size m′
>
> = |B| = m · pq. We learn a pair of parameters γ(k) and β(k) per feature map, rather than per activation.
>
> —— [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

1个卷积核产生1个feature map，1个feature map有1对$\gamma$和$\beta$参数，同一batch同channel的feature map共享同一对$\gamma$和$\beta$参数，若卷积层有$n$个卷积核，则有$n$对$\gamma$和$\beta$参数。

## 没有scale and shift过程可不可以？

BatchNorm有两个过程，Standardization和scale and shift，前者是机器学习常用的数据预处理技术，在浅层模型中，只需对数据进行Standardization即可，Batch Normalization可不可以只有Standardization呢？

答案是可以，但网络的表达能力会下降。

直觉上理解，**浅层模型中，只需要模型适应数据分布即可**。对深度神经网络，每层的输入分布和权重要相互协调，强制把分布限制在zero mean unit variance并不见得是最好的选择，加入参数$\gamma$和$\beta$，对输入进行scale and shift，**有利于分布与权重的相互协调**，特别地，令$\gamma=1, \beta = 0$等价于只用Standardization，令$\gamma=\sigma, \beta=\mu$等价于没有BN层，scale and shift涵盖了这2种特殊情况，在训练过程中决定什么样的分布是适合的，所以使用scale and shift增强了网络的表达能力。

## BN层放在ReLU前面还是后面？

原**paper建议将BN层放置在ReLU前，因为ReLU激活函数的输出非负，不能近似为高斯分布。**

> The goal of Batch Normalization is to achieve a stable distribution of activation values throughout training, and in our experiments **we apply it before the nonlinearity since that is where matching the first and second moments is more likely to result in a stable distribution.**
>
> —— [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

但是，在[**caffenet-benchmark-batchnorm**](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)中，作者基于caffenet在ImageNet2012上做了如下对比实验，

![https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md](https://s2.ax1x.com/2019/12/04/Q1tTXR.png)

实验表明，放在前后的差异似乎不大，**甚至放在ReLU后还好一些。**

**放在ReLU后相当于直接对每层的输入进行归一化，如下图所示，这与浅层模型的Standardization是一致的。**

![https://www.microsoft.com/en-us/research/video/how-does-batch-normalization-help-optimization/](https://s2.ax1x.com/2019/12/04/Q1UYIf.png)

[**caffenet-benchmark-batchnorm**](https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md)中，还有BN层与不同激活函数、不同初始化方法、dropout等排列组合的对比实验，可以看看。

所以，BN究竟应该放在激活的前面还是后面？以及，BN与其他变量，如激活函数、初始化方法、dropout等，如何组合才是最优？**可能只有直觉和经验性的指导意见，具体问题的具体答案可能还是得实验说了算（微笑）。**

## BN层为什么有效？

BN层的有效性已有目共睹，但为什么有效可能还需要进一步研究，这里有一些解释，

- **BN层让损失函数更平滑**。论文[**How Does Batch Normalization Help Optimization**](https://arxiv.org/abs/1805.11604)中，通过分析训练过程中每步梯度方向上步长变化引起的损失变化范围、梯度幅值的变化范围、光滑度的变化，认为添**加BN层后，损失函数的landscape(loss surface)变得更平滑，相比高低不平上下起伏的loss surface，平滑loss surface的梯度预测性更好，可以选取较大的步长。**如下图所示，

  ![https://arxiv.org/abs/1805.11604](https://s2.ax1x.com/2019/12/04/Q16lhn.png)

- **BN更有利于梯度下降**。论文[**An empirical analysis of the optimization of deep network loss surfaces**](https://arxiv.org/abs/1612.04010)中，绘制了VGG和NIN网络在有无BN层的情况下，loss surface的差异，包含初始点位置以及不同优化算法最终收敛到的local minima位置，如下图所示。**没有BN层的，其loss surface存在较大的高原，有BN层的则没有高原，而是山峰，因此更容易下降。**

  ![https://arxiv.org/abs/1612.04010](https://s2.ax1x.com/2019/12/04/Q1cAUJ.png)

- 这里再提供一个直觉上的理解，没有BN层的情况下，网络没办法直接控制每层输入的分布，其分布前面层的权重共同决定，或者说分布的均值和方差“隐藏”在前面层的每个权重中，网络若想调整其分布，需要通过复杂的反向传播过程调整前面的每个权重实现，**BN层的存在相当于将分布的均值和方差从权重中剥离了出来，只需调整$\gamma$和$\beta$两个参数就可以直接调整分布，让分布和权重的配合变得更加容易。**

这里多说一句，论文[**How Does Batch Normalization Help Optimization**](https://arxiv.org/abs/1805.11604)中对比了标准VGG以及加了BN层的VGG每层分布随训练过程的变化，发现两者并无明显差异，认为BatchNorm并没有改善 **Internal Covariate Shift**。**但这里有个问题是**，两者的训练都可以收敛，对于不能收敛或者训练过程十分震荡的模型呢，其分布变化是怎样的？我也不知道，没做过实验（微笑）。

以上。

# 参考

- [arxiv-Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [arxiv-How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604)
- [arxiv-An empirical analysis of the optimization of deep network loss surfaces](https://arxiv.org/abs/1612.04010)
- [talk-How does Batch Normalization Help Optimization?](https://www.microsoft.com/en-us/research/video/how-does-batch-normalization-help-optimization/)
- [How does Batch Normalization Help Optimization?](http://gradientscience.org/batchnorm/)
- [Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
- [Batch Normalization — What the hey?](https://gab41.lab41.org/batch-normalization-what-the-hey-d480039a9e3b)
- [Why Does Batch Normalization Work?](https://abay.tech/blog/2018/07/01/why-does-batch-normalization-work/)
- [An Intuitive Explanation of Why Batch Normalization Really Works](https://mlexplained.com/2018/01/10/an-intuitive-explanation-of-why-batch-normalization-really-works-normalization-in-deep-learning-part-1/)
- [Normalization in Gradient`s Point of View](https://towardsdatascience.com/normalization-in-gradient-s-point-of-view-manual-back-prop-in-tf-2197dfa3497e)
- [Intro to Optimization in Deep Learning: Busting the Myth About Batch Normalization](https://blog.paperspace.com/busting-the-myths-about-batch-normalization/)







