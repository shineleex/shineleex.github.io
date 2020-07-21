---
title: Arctan的快速近似算法
mathjax: true
date: 2020-07-17 12:02:19
tags:
categories:
- 算法
---



博客：[博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee) | [blog](https://blog.shinelee.me/)

# 写在前面

如果$arctan$的计算成为了瓶颈，那么是时候对其进行优化了。

![Arctangent.png](https://gitee.com/shinelee/ImgBed/raw/master/2020/Arctan快速近似算法/Arctangent.png)

$arctan$的近似计算本质上是在所需精度范围内对$arctan$曲线进行拟合，比较直接的想法是**泰勒展开**，
$$
\arctan (x)=x-\frac{x^{3}}{3}+\frac{x^{5}}{5}-\frac{x^{7}}{7}+\ldots
$$
根据需要的精度，确定展开多少项，但$arctan$的泰勒展开在$x$接近1时，收敛较慢，并不高效。

另一个直接的想法是**查表**，根据所需精度，正切值定点化后，将其对应的角度保存成表，计算时，根据最近的正切值查表，一般需要较大的内存空间。

需要注意的是，$arctan(x)$返回的是$(-\pi/2, \pi/2)$， $arctan2(y, x)$返回的范围是$(-\pi, \pi ]$，因为后者可以根据$x$和$y$的正负确定位于哪个象限。实际上，只需近似或存储$[0, \pi/4]$即可（即八象限中的第一象限），若输入向量$(x, y)$，根据$x$和$y$的正负和大小关系，可以折算到所有的八个象限。

此外，[CORDIC（**CO**ordinate **R**otation **DI**gital **C**omputer）](https://wiki2.org/en/CORDIC)算法也是个选择，仅涉及移位和加法操作，但仍需要迭代。

# Arctan快速近似计算

这里，罗列paper 《Efficient Approximations for the Arctangent Function 》中的7种近似算法，这些近似算法通过[Lagrange interpolation](https://wiki2.org/en/Lagrange_polynomial)和minimax optimization techniques得到，最大近似误差和所需计算如下所示，

![error-and-computational-workload.png](https://gitee.com/shinelee/ImgBed/raw/master/2020/Arctan快速近似算法/error-and-computational-workload.png)

从上到下依次为，

- **线性近似**，最大近似误差 $0.07 \ rad = 4^{\circ}$，

$$
\arctan (x) \approx \frac{\pi}{4} x, \quad-1 \leq x \leq 1
$$

- **二阶近似**，最大近似误差 $0.0053 \ rad = 0.3^{\circ}$，

$$
\arctan (x) \approx \frac{\pi}{4} x+0.285 x(1-|x|), \quad-1 \leq x \leq 1
$$

- 搜索更佳的系数，最大近似误差 $0.0038 \ rad = 0.22^{\circ}$，

$$
\arctan (x) \approx \frac{\pi}{4} x+0.273 x(1-|x|), \quad-1 \leq x \leq 1
$$

- $\alpha x^{3}+\beta x$形式的**三阶近似**，最大近似误差 $0.005 \ rad = 0.29^{\circ}$，

$$
\arctan (x) \approx \frac{\pi}{4} x+x\left(0.186982-0.191942 x^{2}\right), \quad-1 \leq x \leq 1
$$

- $x(x-1)(\alpha x-\beta)$形式的**三阶近似**，最大近似误差 $0.0015 \ rad = 0.086^{\circ}$，

$$
\arctan (x) \approx \frac{\pi}{4} x-x(|x|-1)(0.2447+0.0663|x|), \quad-1 \leq x \leq 1
$$

- $x /\left(1+\beta x^{2}\right)$形式的近似，最大近似误差 $0.0047 \ rad = 0.27^{\circ}$，

$$
\arctan (x) \approx \frac{x}{1+0.28086 x^{2}}, \quad-1 \leq x \leq 1
$$

- 另一个近似，最大近似误差 $0.0049 \ rad = 0.28^{\circ}$，

$$
\arctan (x) \approx \frac{x}{1+0.28125 x^{2}}, \quad-1 \leq x \leq 1
$$

实际使用时，可先定点化，按需选取。

以上。

# 参考

- [Streamlining Digital Signal Processing: A Tricks of the Trade Guidebook](https://ieeexplore.ieee.org/book/6241055)
- [FAST APPROXIMATE ARCTAN/ATAN FUNCTION](http://nghiaho.com/?p=997)

