---

title: 从AlexNet(2012)开始
mathjax: true
date: 2019-08-30 13:35:53
tags:
- CNN
categories:
- backbone网络
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)



# 写在前面

本文重点在于回顾深度神经网络在CV领域的**First Blood**——**AlexNet**，AlexNet是首个在大规模图像识别问题取得突破性进展的深度神经网络，相比基于SIFT+FVs、稀疏编码的传统方法，性能提升了10多个百分点（error rate 26.2% → 15.3%，ILSVRC-2012），并由此开启了深度神经网络血洗CV各领域的开端，如下图所示（SuperVision即AlexNet）。

![ImageNet Classification error throughout years and groups](https://s2.ax1x.com/2019/09/02/nPLg74.png)

截止本文时间2019年9月2日，AlexNet论文的引用量达45305，论文作者Alex Krizhevsky、Ilya Sutskever和“深度学习之父”Geoff Hinton。

![citations](https://s2.ax1x.com/2019/09/02/nPX9R1.png)

# 网络结构

AlexNet的原始网络结构如下，可以参见caffe的网络定义[bvlc_alexnet](https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt)，pytorch等也给出了变种实现，见[torchvision/models/alexnet.py](https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py)。

![AlexNet architecture](https://s2.ax1x.com/2019/09/02/nPv5Ie.png)



整个网络大体由5个卷积层和3个全连接层组成，受限于当时的计算资源，网络通过2块GTX580 3GB的GPU训练，上图可见，整个网络上下一分为二，各用1块GPU训练（在caffe中通过group层实现），每个GPU放一半的神经元，网络中第3个卷积层和3个全连接层跨GPU连接。**与使用单个GPU和50%神经元的网络相比，这个双GPU方案的Top1和Top5错误率分别降低了1.7%和1.2%。**



每层的配置如下，第一个卷积层的kernel size为11，stride为4：

![AlexNet architecture](https://s2.ax1x.com/2019/09/05/nmUy2d.png)

# 创新点

为了获得最佳表现，论文中综合应用了很多技术，有些后来已成为通用的标准做法。

- **使用ReLU作为激活函数**，作为non-saturating非线性激活函数有效避免了梯度消失问题，同时与tanh（saturating非线性激活函数）相比，训练速度提升了数倍（CIFAR-10上训练达到25%错误率速度快了6倍）。

- **多GPU训练**，实际上相当于**增加了网络的宽度**，如上节所述，Top1和Top5错误率比单GPU网络分别降低了1.7%和1.2%。

- 提出了**LRN（Local Response Normalization）层**，使用相邻$n$个特征图上同位置的值对当前值进行归一化，公式如下。**LRN被认为没有太大效果，已不被后来者采用。**
  $$
  b_{x, y}^{i}=a_{x, y}^{i} /\left(k+\alpha \sum_{j=\max (0, i-n / 2)}^{\min (N-1, i+n / 2)}\left(a_{x, y}^{j}\right)^{2}\right)^{\beta}
  $$

- 使用**Overlapping Max-Pooling**，如上节图中，Pooling层的kernel size $z=3$，stride $s=2$，$z > s$，与$s=z=2$相比，Top1和Top5错误率分别下降了0.4%和0.3%。

- **通过Data Augmentation数据增广降低过拟合，提高预测准确度**

  - 训练阶段，**通过生成大量训练数据来降低过拟合**，生成数据的方式有2种，
    - 第1种方式从$256\times 256$图像中随机裁剪+左右翻转出$224\times 224$的图像，将训练数据扩大了2048倍；
    - 第2种方式对每张训练图像RGB通道做数值扰动，扰动量通过对整个训练集的RGB像素进行PCA获得，扰动量为$[P_1, P_2, P_3] [\alpha_{1} \lambda_{1}, \alpha_{2} \lambda_{2}, \alpha_{3} \lambda_{3}]^{T}$，其中，$P_i$和 $\lambda_{i}$为RGB像素协方差矩阵的特征向量和特征值，$\alpha_{i}$为0均值0.1标准差的高斯随机值。
  - 预测阶段，从待预测$256\times 256$图中上下左右中间crop + 左右翻转得到10张$224\times 224$的图像，逐一输入网，络对输出结果取平均，来提升预测阶段的准确率，相当于**数据层面的集成学习**。

- 对前2个全连接层使用**Dropout**技术，训练时每次随机让50%的神经元输出为0，以此来降低过拟合，预测时将权重乘以0.5。这样可以强迫网络学习到更鲁棒的特征，也可以从集成学习的视角理解，预测阶段相当于对随机到的所有模型求了个期望。关于Dropout的更多信息可参见论文[《Improving neural networks by preventing co-adaptation of feature detectors》](https://arxiv.org/abs/1207.0580)

  ![dropout](https://s2.ax1x.com/2019/09/10/nNy2XF.gif)

- batchsize 128，SGD Momentum 0.9，weight decay 0.0005，initial learning rate 0.01 停滞时divide by 10，

$$
\begin{aligned} v_{i+1} & :=0.9 \cdot v_{i}-0.0005 \cdot \epsilon \cdot w_{i}-\epsilon \cdot\left\langle\left.\frac{\partial L}{\partial w}\right|_{w_{i}}\right\rangle_{D_{i}} \\ w_{i+1} & :=w_{i}+v_{i+1} \end{aligned}
$$

# 其他有意思的点

回顾AlexNet论文，发现论文中提及了很多有意思的点，有些仅仅是一笔带过，但是可能启发了后面大量的工作，翻回来看才发现“祖师爷”早有预兆。

- **finetune**，在一个库上训练，在另一个库上finetune

  ![AlexNet finetune](https://s2.ax1x.com/2019/09/10/nNr4u4.png)

- **权重可视化**，仅可视化第1个卷积层的96个卷积核权重，发现网络学到了频率方向性的特征，更有意思的是，GPU1上的48个卷积核是颜色无关的，GPU2上的是颜色相关的。

  ![96 Convolutional Kernels](https://s2.ax1x.com/2019/09/10/nN6U9x.png)

- 匹配与检索，使用最后一个全连接层的输出作为特征，通过欧氏距离可计算图像间的特征相似度，可做匹配，提出可以通过auto-encoder进一步压缩获取到short binary code，可用于检索，如下图所示，检索与最左边一列特征最近的图像

  ![retrieve images](https://s2.ax1x.com/2019/09/10/nN2Wgx.png)

- **深度十分重要**，增加深度可以进一步提升性能，当前性能只是受限于计算资源和训练时间（微笑）

  ![depth is important](https://s2.ax1x.com/2019/09/10/nNRRoQ.png)

  ![depth is important](https://s2.ax1x.com/2019/09/10/nNR7LT.png)

- 在ILSVRC 2012上做的报告展示了使用AlexNet做**detection**的结果，如下

  ![AlexNet Localization](https://s2.ax1x.com/2019/09/10/nNhFGd.png)



不愧是开创性工作的paper，给这含金量跪了。

# 参考

- [paper: ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [slides LSVRC 2012: ImageNet Classification with Deep Convolutional Neural Networks](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf)
- [slides 2015: ImageNet Classification with Deep Convolutional Neural Networks](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
- [cs231n_2017_lecture9](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf)
- [Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/)

