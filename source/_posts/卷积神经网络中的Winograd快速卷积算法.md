---
title: 卷积神经网络中的Winograd快速卷积算法
mathjax: true
date: 2019-05-22 16:06:01
tags:
- CNN
categories:
- 深度学习
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

# 写在前面
随便翻一翻流行的推理框架（加速器），如[NCNN](https://github.com/Tencent/ncnn)、[NNPACK](https://github.com/Maratyszcza/NNPACK)等，可以看到，对于卷积层，大家不约而同地采用了Winograd快速卷积算法，该算法出自CVPR 2016的一篇 paper：[Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/abs/1509.09308)。

本文将尝试揭开Winograd算法的神秘面纱。

# 问题定义
将一维卷积运算定义为$F(m, r)$，$m$为Output Size，$r$为Filter Size，则输入信号的长度为$m+r-1$，卷积运算是对应位置相乘然后求和，**输入信号每个位置至少要参与1次乘法**，所以乘法数量最少与输入信号长度相同，记为

$$
\mu(F(m, r))=m+r-1
$$

在行列上分别进行一维卷积运算，可得到二维卷积，记为$F(m\times n, r\times s)$，输出为$m\times n$，卷积核为$r\times s$，则输入信号为$(m+r-1)(n+s-1)$，乘法数量至少为

$$
\begin{aligned} \mu(F(m \times n, r \times s)) &=\mu(F(m, r)) \mu(F(n, s)) \\ &=(m+r-1)(n+s-1) \end{aligned}
$$

若是直接按滑动窗口方式计算卷积，一维时需要$m\times r$次乘法，二维时需要$m\times n \times r \times s$次乘法，**远大于上面计算的最少乘法次数**。

使用Winograd算法计算卷积快在哪里？一言以蔽之：**快在减少了乘法的数量**，将乘法数量减少至$m+r-1$或$(m+r-1)(n+s-1)$。

怎么减少的？请看下面的例子。

# 一个例子 F(2, 3)
先以1维卷积为例，输入信号为$d=\left[ \begin{array}{llll}{d_{0}} & {d_{1}} & {d_{2}} & {d_{3}}\end{array}\right]^{T}$，卷积核为$g=\left[ \begin{array}{lll}{g_{0}} & {g_{1}} & {g_{2}}\end{array}\right]^{T}$，则卷积可写成如下矩阵乘法形式：

$$
F(2, 3) = \left[ \begin{array}{lll}{d_{0}} & {d_{1}} & {d_{2}} \\ {d_{1}} & {d_{2}} & {d_{3}}\end{array}\right] \left[ \begin{array}{l}{g_{0}} \\ {g_{1}} \\ {g_{2}}\end{array}\right]=\left[ \begin{array}{c}{r_0} \\ {r_1}\end{array}\right]
$$

如果是一般的矩阵乘法，则需要**6次乘法和4次加法**，如下：

$$
\begin{array}{l}{r_{0}=\left(d_{0} \cdot g_{0}\right)+\left(d_{1} \cdot g_{1}\right)+\left(d_{2} \cdot g_{2}\right)} \\ {r_{1}=\left(d_{1} \cdot g_{0}\right)+\left(d_{2} \cdot g_{1}\right)+\left(d_{3} \cdot g_{2}\right)}\end{array}
$$

但是，卷积运算中输入信号转换成的矩阵不是任意矩阵，其中**有规律地分布着大量的重复元素**，比如第1行和第2行的$d_1$和$d_2$，卷积转换成的矩阵乘法比一般矩阵乘法的问题域更小，这就让优化存在了可能。

Winograd是怎么做的呢？

$$
F(2,3)=\left[ \begin{array}{lll}{d_{0}} & {d_{1}} & {d_{2}} \\ {d_{1}} & {d_{2}} & {d_{3}}\end{array}\right] \left[ \begin{array}{l}{g_{0}} \\ {g_{1}} \\ {g_{2}}\end{array}\right]=\left[ \begin{array}{c}{m_{1}+m_{2}+m_{3}} \\ {m_{2}-m_{3}-m_{4}}\end{array}\right]
$$

其中，

$$
\begin{array}{ll}{m_{1}=\left(d_{0}-d_{2}\right) g_{0}} & {m_{2}=\left(d_{1}+d_{2}\right) \frac{g_{0}+g_{1}+g_{2}}{2}} \\ {m_{4}=\left(d_{1}-d_{3}\right) g_{2}} & {m_{3}=\left(d_{2}-d_{1}\right) \frac{g_{0}-g_{1}+g_{2}}{2}}\end{array}
$$

乍看上去，为了计算$\begin{array}{l}{r_{0}=m_1 + m_2 + m_3 } \\ {r_{1}=m_2 - m_3 - m_4}\end{array}$，需要的运算次数分别为：
- 输入信号$d$上：4次加法（减法）
- ~~卷积核$g$上：3次加法（$g_1+g_2$中间结果可保留），2次乘法（除法）~~
- 输出$m$上：4次乘法，4次加法

**在神经网络的推理阶段，卷积核上的元素是固定的**，因此$g$**上的运算可以提前算好**，**预测阶段只需计算一次**，可以忽略，所以一共所需的运算次数为$d$与$m$上的运算次数之和，**即4次乘法和8次加法**。

与直接运算的6次乘法和4次加法相比，乘法次数减少，加法次数增加。在计算机中，乘法一般比加法慢，通过减少减法次数，增加少量加法，可以实现加速。

# 1D winograd
上一节中的计算过程写成矩阵形式如下：
$$
Y=A^{T}\left[(G g) \odot\left(B^{T} d\right)\right]
$$

 其中，$\odot$为element-wise multiplication（Hadamard product）对应位置相乘，

$$
B^{T}=\left[ \begin{array}{cccc}{1} & {0} & {-1} & {0} \\ {0} & {1} & {1} & {0} \\ {0} & {-1} & {1} & {0} \\ {0} & {1} & {0} & {-1}\end{array}\right]
$$

$$
G=\left[ \begin{array}{ccc}{1} & {0} & {0} \\ {\frac{1}{2}} & {\frac{1}{2}} & {\frac{1}{2}} \\ {\frac{1}{2}} & {-\frac{1}{2}} & {\frac{1}{2}} \\ {0} & {0} & {1}\end{array}\right]
$$

$$
A^{T}=\left[ \begin{array}{llll}{1} & {1} & {1} & {0} \\ {0} & {1} & {-1} & {-1}\end{array}\right]
$$

$$
g=\left[ \begin{array}{lll}{g_{0}} & {g_{1}} & {g_{2}}\end{array}\right]^{T}
$$

$$
d=\left[ \begin{array}{llll}{d_{0}} & {d_{1}} & {d_{2}} & {d_{3}}\end{array}\right]^{T}
$$

- $g$：卷积核
- $d$：输入信号
- $G$：Filter transform矩阵，尺寸$(m+r-1)\times r$
- $B^T$：Input transform矩阵，尺寸$(m+r-1)\times (m+r-1)$
- $A^T$：Output transform矩阵，尺寸$m \times (m+r-1)$

整个计算过程在逻辑上可以分为4步：
- Input transform
- Filter transform
- Hadamar product
- Output transform

**注意，这里写成矩阵形式，并不意味着实现时要调用矩阵运算的接口，一般直接手写计算过程速度会更快，写成矩阵只是为了数学形式。**

# 1D to 2D，F(2, 3) to F(2x2, 3x3)
上面只是看了1D的一个例子，2D怎么做呢？

论文中一句话带过：
>  A minimal 1D algorithm F(m, r) is **nested with itself** to obtain a minimal 2D algorithm,F(m×m, r×r).

$$
Y=A^{T}\left[\left[G g G^{T}\right] \odot\left[B^{T} d B\right]\right] A
$$

其中，$g$为$r \times r$ Filter，$d$为$(m+r-1)\times (m+r-1)$的image tile。

问题是：怎么**nested with itself**？

这里继续上面的例子$F(2, 3)$，扩展到2D，$F(2\times 2, 3 \times 3)$，先写成矩阵乘法，见下图，图片来自[SlideShare](https://www.slideshare.net/embeddedvision/even-faster-cnns-exploring-the-new-class-of-winograd-algorithms-a-presentation-from-arm?from_action=save)，注意数学符号的变化，

![nested 1D winograd algorithm](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zMi5heDF4LmNvbS8yMDE5LzA1LzIyL1ZwQkZjNi5wbmc)
将卷积核的元素拉成一列，将输入信号每个滑动窗口中的元素拉成一行。注意图中红线划分成的分块矩阵，**每个子矩阵中重复元素的位置与一维时相同，同时重复的子矩阵也和一维时相同**，如下所示
![nested 1D winograd algorithm](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zMi5heDF4LmNvbS8yMDE5LzA1LzIyL1ZwRHhZOS5wbmc)
令$D_0 = [k_0, k_1, k_2, k_3]^T$，即窗口中的第0行元素，$D_1 \ D_2 \ D_3$表示第1、2、3行；$W_0=[w_0, w_1, w_2]^T$，

$$\begin{aligned}
\left[ \begin{array}{c}{r_0} \\ {r_1} \\ {r_2} \\ {r_3}\end{array}\right] &=
\left[ \begin{array}{c}{R_0} \\ {R_1}\end{array}\right] = 
\left[ \begin{array}{c}{K_0 W_0 + K_1 W_1 + K_2 W_2} \\ {K_1 W_0 + K_2 W_1 + K_3 W_2} \end{array} \right] \\
&= \left[ \begin{array}{c} {A^{T}\left[(G W_0) \odot\left(B^{T} D_0 \right)\right] + A^{T}\left[(G W_1) \odot\left(B^{T} D_1 \right)\right] + A^{T}\left[(G W_2) \odot\left(B^{T} D_2 \right)\right]} \\ {A^{T}\left[(G W_0) \odot\left(B^{T} D_1 \right)\right] + A^{T}\left[(G W_1) \odot\left(B^{T} D_2 \right)\right] + A^{T}\left[(G W_2) \odot\left(B^{T} D_3 \right)\right]} \end{array} \right] \\
\\
&=A^{T}\left[\left[G [W_0 \ W_1 \ W_2 ] G^{T}\right] \odot\left[B^{T} [d_0 \ d_1 \ d_2 \ d_3] B\right]\right]A \\
\\
&=A^{T}\left[\left[G g G^{T}\right] \odot\left[B^{T} d B\right]\right] A
\end{aligned}
$$

卷积运算为对应位置相乘再相加，上式中，$A^{T}\left[(G W_0) \odot\left(B^{T} D_0 \right)\right]$表示长度为4的$D_0$与长度为3的$W_0$卷积结果，结果为长度为2的列向量，其中，$(G W_0)$和$(B^{T} D_0)$均为长度为4的列向量，进一步地，$\left[(G W_0) \odot\left(B^{T} D_0 \right)+ (G W_1) \odot\left(B^{T} D_1 \right) + (G W_2) \odot\left(B^{T} D_2 \right)\right]$可以看成3对长度为4的列向量两两对应位置相乘再相加，结果为长度为4的列向量，也可以看成是4组长度为3的行向量的点积运算，同样，$\left[(G W_0) \odot\left(B^{T} D_1 \right)+ (G W_1) \odot\left(B^{T} D_2 \right) + (G W_2) \odot\left(B^{T} D_3 \right)\right]$也是4组长度为3的行向量的内积运算，考虑两者的重叠部分$(B^T D_1)$和$(B^T D_2)$，恰好相当于$G [W_0 \ W_1 \ W_2 ]$的每一行在$B^{T} [d_0 \ d_1 \ d_2 \ d_3]$的对应行上进行1维卷积，上面我们已经进行了列向量卷积的Winograd推导，行向量的卷积只需将所有左乘的变换矩阵转置后变成右乘就可以了，至此，上面的推导结果就不难得出了。

所谓的**nested with itself**如下图所示，
![nested 1D winograd algorithm](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zMi5heDF4LmNvbS8yMDE5LzA1LzIyL1ZweGRpRC5wbmc)

此时，Winograd算法的乘法次数为16（上图$4\times 4$），而直接卷积的乘法次数为36，**降低了2.25倍的乘法计算复杂度**。

# 卷积神经网络中的Winograd
要将Winograd应用在卷积神经网络中，还需要回答下面两个问题：
- 上面我们仅仅是针对一个小的image tile，但是在卷积神经网络中，feature map的尺寸可能很大，难道我们要实现$F(224, 3)$吗？
- 在卷积神经网络中，feature map是3维的，卷积核也是3维的，3D的winograd该怎么做？

第一个问题，在实践中，会将input feature map切分成一个个等大小有重叠的tile，在每个tile上面进行winograd卷积。

第二个问题，3维卷积，相当于逐层做2维卷积，然后将每层对应位置的结果相加，下面我们会看到多个卷积核时更巧妙的做法。

这里直接贴上论文中的算法流程：
[![convnet layer winograd algorithm](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zMi5heDF4LmNvbS8yMDE5LzA1LzIyL1ZwelJuMS5wbmc)](https://imgchr.com/i/VpzRn1)
整体仍可分为4步，
- Input transform
- Filter transform
- Batched-GEMM（批量矩阵乘法）
- Output transform

算法流程可视化如下，图片出自论文[Sparse Winograd Convolutional neural networks on small-scale systolic arrays](https://www.researchgate.net/publication/328091476_Sparse_Winograd_Convolutional_neural_networks_on_small-scale_systolic_arrays/figures)，与算法对应着仔细推敲还是挺直观的。
![An overview of Winograd convolution layer](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zMi5heDF4LmNvbS8yMDE5LzA0LzI5L0UxcVNvUi5wbmc)
注意图中的Matrix Multiplication，对应3维卷积中逐channel卷积后的对应位置求和，相当于$(m+r-1)^2$个矩阵乘积，参与乘积的矩阵尺寸分别为$\lceil H / m\rceil\lceil W / m\rceil \times C$和$C \times K$，把Channel那一维消掉。

# 总结
- Winograd算法通过减少乘法次数来实现提速，但是加法的数量会相应增加，同时需要额外的transform计算以及存储transform矩阵，随着卷积核和tile的尺寸增大，就需要考虑加法、transform和存储的代价，而且tile越大，transform矩阵越大，计算精度的损失会进一步增加，所以一般Winograd只适用于较小的卷积核和tile（对大尺寸的卷积核，可使用FFT加速），在目前流行的网络中，小尺寸卷积核是主流，典型实现如$F(6\times 6, 3\times 3)$、$F(4\times 4, 3\times 3)$、$F(2\times 2, 3\times 3)$等，可参见[NCNN](https://github.com/Tencent/ncnn/tree/master/src/layer/arm)、[FeatherCNN](https://github.com/Tencent/FeatherCNN/blob/booster/src/booster/arm/winograd_kernels_F63.cpp)、[ARM-ComputeLibrary](https://github.com/ARM-software/ComputeLibrary/tree/master/src/core/NEON/kernels/convolution/winograd/transforms)等源码实现。
- 就卷积而言，Winograd算法和FFT类似，都是先通过线性变换将input和filter映射到新的空间，在那个空间里简单运算后，再映射回原空间。
- 与im2col+GEMM+col2im相比，winograd在划分时使用了更大的tile，就划分方式而言，$F(1\times 1, r\times r)$与im2col相同。


# 参考
- [arxiv: Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/abs/1509.09308)
- [video: Fast Algorithms for Convolutional Neural Networks by Andrew Lavin and Scott Gray](https://www.bilibili.com/video/av50718398)
- [video: Even Faster CNNs Exploring the New Class of Winograd Algorithms](https://www.bilibili.com/video/av53072685)
- [arxiv: Sparse Winograd Convolutional neural networks on small-scale systolic arrays](https://arxiv.org/abs/1810.01973)
- [ARM-software/ComputeLibrary](https://github.com/ARM-software/ComputeLibrary)