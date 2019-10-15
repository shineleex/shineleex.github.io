---
title: im2col：将卷积运算转为矩阵相乘
mathjax: true
date: 2019-04-26 18:03:05
tags:
- CNN
categories:
- 深度学习基础
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

# im2col实现
如何将卷积运算转为矩阵相乘？直接看下面这张图，以下图片来自论文[High Performance Convolutional Neural Networks for Document Processing](https://hal.inria.fr/inria-00112631/)：

![im2col](https://s2.ax1x.com/2019/04/26/EmXKun.png)
上图为3D卷积的传统计算方式与矩阵乘法计算方式的对比，传统卷积运算是将卷积核以滑动窗口的方式在输入图上滑动，当前窗口内对应元素相乘然后求和得到结果，一个窗口一个结果。**相乘然后求和恰好也是向量内积的计算方式**，所以可以将每个窗口内的元素拉成向量，通过向量内积进行运算，多个窗口的向量放在一起就成了矩阵，每个卷积核也拉成向量，多个卷积核的向量排在一起也成了矩阵，于是，卷积运算转化成了矩阵运算。

下图为转化后的矩阵尺寸，padding为0：
![EmzaRO.png](https://s2.ax1x.com/2019/04/26/EmzaRO.png)
代码上怎么实现呢？这里参看一下[SeetaFaceEngine/FaceIdentification/src/conv_net.cpp](https://github.com/seetaface/SeetaFaceEngine/blob/master/FaceIdentification/src/conv_net.cpp#L74)  中的代码，与上面的图片对照着看比较直观。

```cpp
int dst_h = (src_h - kernel_h) / stride_h_ + 1; // int src_h = input->height(); int kernel_h = weight->height();
int dst_w = (src_w - kernel_w) / stride_w_ + 1; // int src_w = input->width(); int kernel_w = weight->width();
int end_h = src_h - kernel_h + 1;
int end_w = src_w - kernel_w + 1;
int dst_size = dst_h * dst_w;
int kernel_size = src_channels * kernel_h * kernel_w;

const int src_num_offset = src_channels * src_h * src_w; // int src_channels = input->channels();
float* const dst_head = new float[src_num * dst_size * dst_channels];
float* const mat_head = new float[dst_size * kernel_size];

const float* src_data = input->data().get();
float* dst_data = dst_head;
int didx = 0;

for (int sn = 0; sn < src_num; ++sn) {
  float* mat_data = mat_head;
  for (int sh = 0; sh < end_h; sh += stride_h_) {
    for (int sw = 0; sw < end_w; sw += stride_w_) {
      for (int sc = 0; sc < src_channels; ++sc) {
        int src_off = (sc * src_h + sh) * src_w + sw;
        for (int hidx = 0; hidx < kernel_h; ++hidx) {
          memcpy(mat_data, src_data + src_off,
                  sizeof(float) * kernel_w);
          mat_data += kernel_w;
          src_off += src_w;
        }
      } // for sc
    } // for sw
  } // for sh
  src_data += src_num_offset;

  const float* weight_head = weight->data().get();
  // int dst_channels = weight->num();
  matrix_procuct(mat_head, weight_head, dst_data, dst_size, dst_channels, 
    kernel_size, true, false);
    
  dst_data += dst_channels * dst_size;
} // for sn
```
`src_num `个输入，每个尺寸为 `src_channels * src_h * src_w`，卷积核尺寸为`kernel_size = src_channels * kernel_h * kernel_w`，将每个输入转化为二维矩阵，尺寸为`(dst_h * dst_w) * (kernel_size)`，可以看到最内层循环在逐行拷贝当前窗口内的元素，窗口大小与卷积核大小相同，一次拷贝`kernel_w`个元素，一个窗口内要拷贝`src_channels*kernel_h`次，因此一个窗口共拷贝了`kernel_size`个元素，共拷贝`dst_h * dst_w`个窗口，因此输入对应的二维矩阵尺寸为`(dst_h * dst_w) * (kernel_size)`。对于卷积核，有`dst_channels= weight->num();`个卷积核，因为是行有先存储，卷积核对应的二维矩阵尺寸为`dst_channels*(kernel_size)`。**逻辑上虽然为矩阵乘法，实现时两个矩阵逐行内积即可**。

# 优缺点分析
将卷积运算转化为矩阵乘法，从乘法和加法的运算次数上看，两者没什么差别，但是转化成矩阵后，运算时需要的数据被存在连续的内存上，这样访问速度大大提升（cache），同时，矩阵乘法有很多库提供了高效的实现方法，像BLAS、MKL等，转化成矩阵运算后可以通过这些库进行加速。

缺点呢？这是一种空间换时间的方法，消耗了更多的内存——转化的过程中数据被冗余存储。


# 参考
- [在 Caffe 中如何计算卷积？](https://www.zhihu.com/question/28385679)
- [High Performance Convolutional Neural Networks for Document Processing](https://hal.inria.fr/inria-00112631/)
- [algorithm convolution / filter and cross-correlation and implementation](https://bzdww.com/article/182442/)
- [Convolution in Caffe: a memo](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo)