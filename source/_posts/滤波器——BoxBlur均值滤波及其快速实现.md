---
title: 滤波器——BoxBlur均值滤波及其快速实现
mathjax: true
date: 2018-07-13 10:20:57
tags: 
- 滤波器
categories: 
- 传统计算机视觉
---


# 动机：卷积核、滤波器、卷积、相关
在数字图像处理的语境里，图像一般是二维或三维的矩阵，卷积核（kernel）和滤波器（filter）通常指代同一事物，即**对图像进行卷积或相关操作时使用的小矩阵**，尺寸通常较小，常见的有3\*3、5\*5、7\*7等。卷积操作相当于对滤波器旋转180度后的相关操作，如下图所示，但很多滤波器是中心对称的，而且两者运算上可以等价，所以很多时候不太区分。

![相关运算与卷积运算](https://s1.ax1x.com/2018/11/18/FSSAsJ.png)


设计不同的滤波器，可以达到去噪（denoising）、平滑（smoothing）、模糊（blurring）、锐化（sharpening）、浮雕（embossing）、边缘检测（edge detection）等目的。在空域中直接进行卷积操作（滑动窗口），需要4层循环嵌套，复杂度达到$O(m^2*n^2)$，$m$为图像尺寸，$n$为滤波器尺寸，随着图像或卷积尺寸增大，复杂度以平方快速增长，因此需要一些快速实现方式，尤其是在计算资源并不充足的嵌入式等端上。


# Box Blur
均值滤波器可能是最基本最常见的滤波器了，一个3\*3的均值滤波器如1所示，使用该滤波器对图像进行滤波，相当于对图像中的每一个像素使用其周围的像素进行平均。均值滤波器用途广泛，除最直接的平滑操作外，还可近似实现其他滤波操作，比如**带通滤波**和**高斯平滑**等。因为均值滤波器在频域近似为一个低通滤波器，因此两个不同半径的均值滤波器滤波结果的差值可近似带通滤波器；根据[中心极限定理](https://wiki2.org/en/Central_limit_theorem)，多次Box Blur的结果可近似高斯平滑。**应用得越广泛就越需要仔细优化，可以采用均值滤波器来近似实现其他滤波器的一个前提就是均值滤波可以更高效。** 
$$
\frac{1}{9}
 \left[
 \begin{matrix}
   1 & 1 & 1 \\
   1 & 1 & 1 \\
   1 & 1 & 1
  \end{matrix}
  \right] \tag{1}
$$

直接实现四层循环的均值滤波复杂度为$O(m^2*n^2)$，可以利用均值滤波器所有权重都相同等性质实现快速滤波。

## 行列分解实现
可将卷积核分解为列向量和行向量的相乘，如2所示，对图像进行2D的均值滤波，等价于先逐行进行平均然后逐列平均，复杂度可由$O(m^2*n^2)$ 降至$O(m^2*2n)$ 。这样实现的前提是卷积核可分解，换句话说，可分解的卷积核均可考虑这样优化，比如高斯滤波等。
$$
\frac{1}{9}
 \left[
 \begin{matrix}
   1 & 1 & 1 \\
   1 & 1 & 1 \\
   1 & 1 & 1
  \end{matrix}
  \right] = \frac{1}{3}
  \left[
  \begin{matrix}
   1  \\
   1  \\
   1 
  \end{matrix}
  \right]
  \cdot
   \frac{1}{3}
   \left[
  \begin{matrix}
   1  & 1 & 1 \\
  \end{matrix}
  \right] \tag{2}
$$

## 类“队列”实现
行列分解后，相当于在行上和列上进行1D滑动窗口均值滤波。在1D窗口滑动过程中，相邻窗口有大量元素是重叠的，比如下图中，8、5、10和5、10、7其中5和10就是重叠的。整个滑动过程可以看成是不断进出“队列”的过程，窗口每向右移动1个像素，相当于最左侧的像素出队列，最右侧的像素进队列，当前像素的滤波结果为当前队列内元素之和然后平均，而前后一直驻留在队列中的元素则不需要重复加和，通过避免重复计算来实现提速。

![1D滑动窗口](https://s1.ax1x.com/2018/11/18/FSS1Qe.png)


因此，计算第$i+1$ 个窗口的和$S[i+1]$可以通过第$i$ 个窗口的和$S[i]$与最左$x[i-r]$最右$x[i+r+1$的元素得到，$r$ 为滤波器半径，如下：
$$S[i+1] = S[i] + x[i+r+1] - x[i-r]$$
这样，我们得到了与滤波器尺寸无关的算法，算法复杂度进一步降低至$O(m^2)$ 。

此外，我们也可考虑2D的滑动窗口，如下图所示：

![2D滑动窗口](https://s1.ax1x.com/2018/11/18/FSStot.png)

同样利用相邻滑动窗口内的重叠元素，计算以元素$(i, j)$ 为中心的窗口元素之和$S[i, j]$ 如下，其中$C[i, j]$为窗口内以$(i, j)$为中心的半径为$r$的列和，
$$S[i, j] = \sum_{k=-r}^{+r} C[i, j+k]$$

窗口向右移动时，
$$S[i, j+1] = S[i, j] + C[i, j+r+1] - C[i, j- r]$$

窗口向下移动时，
$$C[i+1, j] = C[i, j] + C[i+r+1, j] - C[i-r-1, j]$$

滤波结果为：
$$x'[i, j] = \frac{1}{(2r+1)^2} S[i, j]$$

## 积分图
如果需要得到多个不同半径的均值滤波结果时，使用积分图（[Summed-area table](https://wiki2.org/en/Summed-area_table)）可能是个好办法。积分图中$(x, y)$ 位置$I(x, y)$的值等于原图中该位置左上角所有像素之和，累加和包不包含这个像素自身所在的行和列与具体实现有关，这里沿用[Wiki](https://wiki2.org/en/Summed-area_table)上的表述方式包含（[Opencv为不包含](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#integral)），计算方式如下：
$$I(x, y) = \sum_{x'\le x, \ y' \le y}i(x', y')$$

积分图可通过单趟遍历快速实现，有了积分图就可以计算任意尺寸box内元素之和，仅需2次减法和1次加法常数次运算，如下：

![积分图](https://s1.ax1x.com/2018/11/18/FSSUFP.png)


这样，当需要不同尺寸均的值滤波结果时，使用积分图的运算时间是一样的。


## 指令级优化
除了以上优化方法，还可采用指令级优化。对每一个像素位置求均值是在该像素的邻域范围内进行的，同一行上的像素位于连续的内存区域，对像素施加的都是近乎相同的操作——加法或减法，因此时宜采用[SIMD](https://wiki2.org/en/SIMD)指令，如MMX、SSE、AVX、NEON等，同时载入多个数据、同时对多个数据进行相同的操作，一些实现方式可参见 参考资料，这里不再详述。**需要注意的是，指令级的优化意味着兼容性、可扩展性的损失，如果代码尚未稳定，则不建议采用**。

# 一些优缺点和总结

这里简单分析下各种方法的优缺点：

- **类“队列”实现**：不能实现in-place操作，如果内存空间不足，可缓存一个窗口高度图像宽度的内存块，在缓存块操作后再写回原图。
- **积分图方法**：需要较大的内存来存储积分图，好处是积分图仅需求取一次，后面所有尺寸的Box Blur均可使用，而且求各处的滤波结果互不依赖，方便并行化。

**基本上所有的优化方式的出发点都是减少不必要的重复计算，本文所介绍的几种方法在其他滤波操作的优化中也常被采用。**以上仅为算法思路介绍，具体实现时可能要进一步考虑内存访问的时间、边界处理等细节，不再赘述。

# 参考
- [Box Blur](https://wiki2.org/en/Box_blur)
- [Fast Image Convolutions](https://web.archive.org/web/20060718054020/http://www.acm.uiuc.edu/siggraph/workshops/wjarosz_convolution_2001.pdf)
- [Tips & Tricks: Fast Image Filtering Algorithms](https://www.academia.edu/5661387/Tips_and_Tricks_Fast_Image_Filtering_Algorithms)
- [APPROXIMATING A GAUSSIAN USING A BOX FILTER](http://nghiaho.com/?p=1159)
- [Filter primitive ‘feGaussianBlur’](https://www.w3.org/TR/SVG11/filters.html#feGaussianBlurElement)
- [SSE图像算法优化系列十三：超高速BoxBlur算法的实现和优化（Opencv的速度的五倍）](https://www.cnblogs.com/Imageshop/p/8302990.html)