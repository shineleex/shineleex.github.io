---
title: OpenCV各版本差异与演化，1.x To 4.0
mathjax: false
date: 2018-10-31 17:44:51
tags:
- opencv
categories:
- 库与框架
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

# 写在前面

最近因项目需要，得把OpenCV捡起来，登录[OpenCV官网](https://opencv.org/)，竟然发现release了[4.0.0-beata版本](https://opencv.org/releases.html)，所以借此机会，查阅资料，了解下OpenCV各版本的差异及其演化过程，形成了以下几点认识：
1. **新版本的产生是为了顺应当下的需要**，通过版本更新，接纳新技术和新方法，支持新兴编程语言接口，使用新的指令集，优化性能，解决固有问题等
2. **新技术新方法会优先加入到新的大版本中**，即使新的技术方法可以在旧版本中实现，但为了推动用户向新版本迁移，仍会优先加入到新版本中（这条看着与第1条差不多，实际意义是不同的）
3. **新版本不可避免地会带有旧版本的痕迹**，毕竟新版本是从旧版本基础上“生长”出来的，新老版本间能看到比较明显的过渡痕迹，同时出于降低迁移成本的考虑，需要（部分）向前兼容

因此，如果新版本已经稳定，且需要从头开始新项目，先考虑拥抱新版本。若碰到问题，可到旧版本的资料中找找答案。但这并不绝对，具体情况还得具体分析。

下面分析下各版本的差异以及演化路径。

# OpenCV版本差异与演化，1.x To 4.0
![OpenCV](https://s1.ax1x.com/2018/11/18/izLDET.png)

## OpenCV 1.x

OpenCV 最初基于**C语言**开发，API也都是基于C的，面临内存管理、指针等C语言固有的麻烦。

**2006年10月1.0**发布时，部分使用了C++，同时支持Python，其中已经有了random trees、boosted trees、neural nets等机器学习方法，完善对图形界面的支持。

**2008年10月1.1pre1**发布，使用 VS2005构建，Python bindings支持**Python 2.6**，Linux下支持**Octave bindings**，在这一版本中加入了SURF、RANSAC、Fast approximate nearest neighbor search等，Face Detection (cvHaarDetectObjects)也变得更快。

## OpenCV 2.x

当C++流行起来，OpenCV 2.x发布，其尽量使用C++而不是C，但是为了向前兼容，仍保留了对C API的支持。从2010年开始，2.x决定不再频繁支持和更新C API，而是focus在**C++ API**，C API仅作备份。

**2009年9月2.0 beta**发布，主要使用**CMake构建**，加入了很多新特征、描述子等，如FAST、LBP等。

**2010年4月2.1**版本，加入了Grabcut等，可以使用**SSE/SSE2…**指令集。

**2010年10月2.2**版本发布，OpenCV的模块变成了大家熟悉的模样，像**opencv_imgproc**、**opencv_features2d**等，同时有了**opencv_contrib**用于放置尚未成熟的代码，**opencv_gpu**放置使用CUDA加速的OpenCV函数。

**2011年6月起的2.3.x**版本、**2012年4月起的2.4.x**版本，一面增加新方法，一面修复bug，同时加强对GPU、Java  for Android、 OpenCL、并行化的支持等等，OpenCV愈加稳定完善，值得注意的是 SIFT和SURF从2.4开始被放到了**nonfree** 模块（因为专利）。

考虑到过渡，OpenCV 2.4.x仍在维护，不过以后可能仅做bug修复和效率提升，不再增加新功能——鼓励向3.x迁移。

## OpenCV 3.x

随着3.x的发布，1.x的C API将被淘汰不再被支持，以后C API可能通过C++源代码自动生成。3.x与2.x不完全兼容，与2.x相比，主要的不同之处在于OpenCV 3.x 的大部分方法都使用了**OpenCL加速**。

**2014年8月3.0 alpha**发布，除大部分方法都使用OpenCL加速外，3.x默认包含以及使用[IPP](https://wiki2.org/en/Integrated_Performance_Primitives)，同时，matlab bindings、Face Recognition、SIFT、SURF、 text detector、motion templates & simple flow 等都移到了**opencv_contrib**下（opencv_contrib不仅存放了尚未稳定的代码，同时也存放了涉及专利保护的技术实现），大量涌现的新方法也包含在其中。

**2017年8月3.3**版本，**2017年12月开始的3.4.x**版本，opencv_dnn从opencv_contrib移至opencv，同时OpenCV开始支持C++ 11构建，之后明显感到对**神经网络**的支持在加强，opencv_dnn被持续改进和扩充。

## OpenCV 4.0

**2018年10月4.0.0**发布，OpenCV开始需要支持**C++11**的编译器才能编译，同时对几百个基础函数使用 **"wide universal intrinsics"**重写，这些内联函数可以根据目标平台和编译选项映射为SSE2、 SSE4、 AVX2、NEON 或者 VSX 内联函数，获得性能提升。此外，还加入了QR code的检测和识别，以及Kinect Fusion algorithm，DNN也在持续改善和扩充。

# 总结
这些年来，计算机视觉领域的新技术新方法不断涌现，指令集、编程语言和并行化技术越发先进，OpenCV也在紧跟时代的脚步，不断吸收完善自身。本文仅对OpenCV的演化过程仅总结了部分要点，详细可参见 OpenCV 在 github上的ChangeLog。

# 参考
- [OpenCV ChangeLog 1.0 – 2.1](https://github.com/opencv/opencv/wiki/ChangeLog_v10-v21)
- [OpenCV Change Logs](https://github.com/opencv/opencv/wiki/ChangeLog)
- [Why there are two versions of OpenCV 3.x and 2.4.xx ?](http://answers.opencv.org/question/92583/why-there-are-two-versions-of-opencv-3x-and-24xx/)
- [what is the difference between OpenCV 2.4.11 and 3.0.0](https://stackoverflow.com/questions/29579801/what-is-the-difference-between-opencv-2-4-11-and-3-0-0)
- [Where did SIFT and SURF go in OpenCV 3?](https://www.pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/)
- [opencv_contrib](https://github.com/opencv/opencv_contrib)


