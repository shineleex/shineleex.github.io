---
title: ZFNet(2013)及可视化的开端
mathjax: false
date: 2019-09-21 15:23:05
tags:
categories:
- backbone网络
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)



# 写在前面

ZFNet出自论文[《 Visualizing and Understanding Convolutional Networks》](https://arxiv.org/abs/1311.2901)，作者Matthew D. Zeiler和Rob Fergus——显然ZFNet是以两位作者名字的首字母命名的，截止20190911，论文引用量为4207。ZFNet通常被认为是**ILSVRC 2013**的冠军方法，但实际上ZFNet排在第3名，前两名分别是Clarifai和NUS，不过Clarifai和ZFNet都出自Matthew D. Zeiler之手，见[ILSVRC2013 results](http://www.image-net.org/challenges/LSVRC/2013/results.php#cls)。



ZFNet(2013)在AlexNet(2012)的基础上，性能再次提升，如下图所示，图片来自[cs231n_2019_lecture09](http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture09.pdf)。

![ImageNet Winners](https://s2.ax1x.com/2019/09/11/naIoOs.png)



论文最大的贡献有2个：

- 提出了**ZFNet**，一种比AlexNet性能更好的网络架构
- 提出了一种**特征可视化**的方法，并据此来分析和理解网络



本文将围绕上述2点展开，先介绍网络架构，再介绍特征可视化的方法。



# 网络架构与动机

ZFNet的网络架构如下

![ZFNet Architecture](https://s2.ax1x.com/2019/09/11/ndSmPH.png)

ZFNet的网络架构是在AlexNet基础上修改而来，与AlexNet相比，差异不大：

- 第1个卷积层，kernel size从11减小为7，将stride从4减小为2（这将导致feature map增大1倍）
- 为了让后续feature map的尺寸保持一致，第2个卷积层的stride从1变为2

仅这2项修改，就获得了几个点的性能提升。所以，重要的是**为什么这样修改？这样修改的动机是什么？**文中这样叙述：

![ZFNet Architecture Selection](https://s2.ax1x.com/2019/09/11/ndpQfJ.png)

通过对AlexNet的特征进行可视化，文章作者发现第2层出现了**aliasing**。在数字信号处理中，**aliasing是指在采样频率过低时出现的不同信号混淆的现象**，作者认为这是第1个卷积层stride过大引起的，为了解决这个问题，可以**提高采样频率**，所以将stride从4调整为2，与之相应的将kernel size也缩小（可以认为stride变小了，kernel没有必要看那么大范围了），这样修改前后，特征的变化情况如下图所示，第1层呈现了更多更具区分力的特征，第二2层的特征也更加清晰，没有aliasing现象。更多关于aliasing的内容，可以参见[Nyquist–Shannon sampling theorem](https://wiki2.org/En/Nyquist%E2%80%93Shannon_sampling_theorem)和[Aliasing](https://wiki2.org/en/Aliasing)。

![stride 2 vs 4, kernel size 7x7 vs 11x11](https://s2.ax1x.com/2019/09/11/ndAFEt.png)



这就引出了另外一个问题，如何将特征可视化？正如论文标题Visualizing and Understanding Convolutional Networks所显示的那样，**与提出一个性能更好的网络结构相比，这篇论文更大的贡献在于提出一种将卷积神经网络深层特征可视化的方法**。



# 特征可视化

在博文《卷积神经万络之卷积计算、作用与思想》 [博客园](https://www.cnblogs.com/shine-lee/p/9932226.html) | [CSDN](https://blog.csdn.net/blogshinelee/article/details/83858851) | [blog.shinelee.me](https://blog.shinelee.me/2018/11-08-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B9%8B%E5%8D%B7%E7%A7%AF%E8%AE%A1%E7%AE%97%E3%80%81%E4%BD%9C%E7%94%A8%E4%B8%8E%E6%80%9D%E6%83%B3.html) 中，我们讲到**卷积神经网络通过逐层卷积将原始像素空间逐层映射到特征空间，深层feature map上每个位置的值都代表与某种模式的相似程度，但因为其位于特征空间，不利于人眼直接观察对应的模式，为了便于观察理解，需要将其映射回像素空间**，“从群众中来，到群众中去”，论文[《 Visualizing and Understanding Convolutional Networks》](https://arxiv.org/abs/1311.2901)就重点介绍了如何“到群众中去”。



**可视化操作，针对的是已经训练好的网络，或者训练过程中的网络快照，可视化操作不会改变网络的权重，只是用于分析和理解在给定输入图像时网络观察到了什么样的特征，以及训练过程中特征发生了什么变化。**



下面这张图截自[论文同款talk](https://www.bilibili.com/video/av67354448/)：

![Projecting Back](https://s2.ax1x.com/2019/09/11/nd0MLQ.png)



给定1张输入图像，先前向传播，得到每一层的feature map，如果想可视化第$i$层学到的特征，保留该层feature map的最大值，将其他位置和其他feature map置0，将其反向映射回原始输入所在的像素空间。对于一般的卷积神经网络，前向传播时不断经历 input image→conv → rectification → pooling →……，可视化时，则从某一层的feature map开始，依次反向经历 unpooling → rectification → deconv → …… → input space，如下图所示，上方对应更深层，下方对应更浅层，前向传播过程在右半侧从下至上，特征可视化过程在左半侧从上至下：

![deconvnet](https://s2.ax1x.com/2019/09/11/ndsbNj.png)

可视化时每一层的操作如下：

- **Unpooling**：在前向传播时，**记录相应max pooling层每个最大值来自的位置**，在unpooling时，根据来自上层的map直接填在相应位置上，如上图所示，Max Locations “Switches”是一个与pooling层输入等大小的二值map，标记了每个局部极值的位置。
- **Rectification**：因为使用的ReLU激活函数，前向传播时只将正值原封不动输出，负值置0，**“反激活”过程与激活过程没什么分别**，直接将来自上层的map通过ReLU。
- **Deconvolution**：可能称为transposed convolution更合适，卷积操作output map的尺寸一般小于等于input map的尺寸，transposed convolution可以将尺寸恢复到与输入相同，相当于上采样过程，该操作的做法是，与convolution共享同样的卷积核，但需要将其**左右上下翻转**（即中心对称），然后作用在来自上层的feature map进行卷积，结果继续向下传递。关于Deconvolution的更细致介绍，可以参见博文《一文搞懂 deconvolution、transposed convolution、sub-­pixel or fractional convolution》 [博客园]((https://www.cnblogs.com/shine-lee/p/11559825.html)) | [CSDN](https://blog.csdn.net/firelx/article/details/101078452) | [blog.shinelee.me](https://blog.shinelee.me/2019/09-16-%E4%B8%80%E6%96%87%E6%90%9E%E6%87%82deconvolution%E3%80%81transposed-convolutionon%E3%80%81sub%C2%ADpixel-or-fractional-convolution.html)。

不断经历上述过程，将特征映射回输入所在的像素空间，就可以呈现出人眼可以理解的特征。给定不同的输入图像，看看每一层关注到最显著的特征是什么，如下图所示：

![Visualization of features in a fully trained model](https://s1.ax1x.com/2018/11/18/izLYCQ.png)



# 其他

除了网络架构和可视化方法，论文中还有其他一些值得留意的点，限于篇幅就不展开了，这里仅做记录，详细内容可以读一读论文：

- **Occlusion Sensitivity** 实验：使用一个灰色的小块，遮挡输入图像的不同区域，观察对正类输出概率的影响，以此来分析哪个区域对分类结果的影响最大，即对当前输入图像，网络最关注哪个区域。结果发现，feature map最强响应可视化后对应的区域影响最大。
- **Feature Generalization**：在ImageNet上预训练，固定权重，然后迁移到其他库上（Caltech-101、Caltech-256），重新训练最后的softmax classifier，只需要很少的样本就能快速收敛，且性能不错。
- **Feature Analysis**：对训练好的网络，基于每一层的特征单独训练SVM或Softmax分类器，来评估不同层特征的区分能力，发现越深层的特征区分能力越强。



以上。



# 参考

- [paper: Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
- [talk: Visualizing and Understanding Deep Neural Networks by Matt Zeiler](https://www.bilibili.com/video/av67354448/)
- [Aliasing](http://users.wfu.edu/matthews/misc/DigPhotog/alias/)
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)