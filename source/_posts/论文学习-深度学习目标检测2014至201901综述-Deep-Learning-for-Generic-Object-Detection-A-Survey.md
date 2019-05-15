---
title: 论文学习-深度学习目标检测2014至201901综述-Deep Learning for Generic Object Detection A Survey
mathjax: true
date: 2019-02-14 18:19:17
tags:
categories:
- 目标检测
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

[toc]
# 写在前面

paper：https://arxiv.org/abs/1809.02165
github：https://github.com/hoya012/deep_learning_object_detection，A paper list of object detection using deep learning

这篇综述对深度学习目标检测2014至201901取得的进展进行了总结，包括：

> More than 250 key contributions are included in this survey, covering many aspects of generic object detection research: **leading detection frameworks** and fundamental subprob-lems including **object feature representation**, **object proposal generation**, **context information modeling** and **training strategies**; **evaluation issues**, **specifically benchmark datasets**, **evaluation metrics**, and **state of the art performance**.

本文的主要目的在于摘录paper中的一些重要图表和结论，作为系统学习的索引，不做详细的展开。

下面两张图来自github，分别为paper list和performance table，红色为作者认为必读的paper。
![目标检测DCNN paper list](https://s2.ax1x.com/2019/02/13/k0rACF.png)
![performance table](https://s2.ax1x.com/2019/02/14/kBHdW4.png)

# 目标检测任务与挑战

**目标检测任务的输入是一张图像，输出是图像中的物体位置和类别**，如下图所示，位置可通过Bounding Box描述，也可描述为像素的集合。
![通用目标检测任务](https://s2.ax1x.com/2019/02/13/k0w3Ox.png)
为了确定图片中物体的位置和类别，要面临很多挑战，一个好的检测器要做到**定位准确**、**分类准确**还要**效率高**，需要对光照、形变、尺度、视角、尺寸、姿态、遮挡、模糊、噪声等情况鲁棒，需要能容忍可能存在的较大的类内差异，又能区分开较小的类间差异，同时还要保证高效。
![目标检测任务的挑战](https://s2.ax1x.com/2019/02/13/k0rWV0.png)
![目标检测任务的挑战](https://s2.ax1x.com/2019/02/13/k0sEIf.png)

# 目标检测方法汇总

在2012年前，目标检测方法主要是人工特征工程+分类器，2012年后主要是基于DCNN的方法，如下图所示：
![目标检测Milestones](https://s2.ax1x.com/2019/02/13/k0Dk6A.png)
![DCNN目标检测](https://s2.ax1x.com/2019/02/13/k0DahF.png)

目标检测的框架可以分成2类：

 1. **Two stage detection framework**：含region proposal，先获取ROI，然后对ROI进行识别和回归bounding box，以RCNN系列方法为代表。
 2. **One stage detection framework**：不含region proposal，将全图grid化，对每个grid进行识别和回归，以YOLO系列方法为代表。

Pipeline对比与演化如下：
![目标检测方法Pipeline对比与演化](https://s2.ax1x.com/2019/02/13/k0WF1K.jpg)
**主干网络、检测框架设计、大规模高质量的数据集**是决定检测性能的3个最重要的因素，决定了学到特征的好坏以及特征使用的好坏。

# 基础子问题
这一节谈论的重点包括：基于DCNN的特征表示、候选区生成、上下文信息、训练策略等。

## 基于DCNN的特征表示
### 主干网络（network backbone）
ILSVRC（ImageNet Large Scale Visual Recognition Competition）极大促进了DCNN architecture的改进，在计算机视觉的各种任务中，往往将这些经典网络作为主干网络（backbone），再在其上做各种文章，常用在目标检测任务中的DCNN architectures如下：
![DCNN architectures](https://s2.ax1x.com/2019/02/13/k0fnrF.png)
### Methods For Improving Object Representation
物体在图像中的尺寸是未知的，图片中的不同物体尺寸也可能是不同的，而DCNN越深层的感受野越大，因此只在某一层上进行预测显然是难以达到最优的，一个自然的想法是利用不同层提取到的信息进行预测，称之为**multiscale object detection**，可分成3类：

1. Detecting with combined features of multiple CNN layers
2. Detecting at multiple CNN layers; 
3. Combinations of the above two methods

直接看图比较直观：
![ION和HyperNet](https://s2.ax1x.com/2019/02/14/kBo0SA.png)
![RFB 与 ZIP](https://s2.ax1x.com/2019/02/14/kB3OuF.jpg)
尝试对几何变形进行建模也是改善Object Representation的一个方向，方法包括结合Deformable Part based Models (DPMs)的方法、Deformable Convolutional Networks (DCN)方法等。
![改善DCNN特征表示的方法](https://s2.ax1x.com/2019/02/14/kB8eUA.jpg)
## Context Modeling 
上下文信息可以分为3类：

 1. **Semantic context**: The likelihood of an object to be found in some scenes but not in others;
 2. **Spatial context**: The likelihood of finding an object in some position and not others with respect to other objects in the scene; 
 3. **Scale context**: Objects have a limited set of sizes relative to other objects in the scene. 

DCNN通过学习不同抽象层级的特征可能已经隐式地使用了contextual information，因此目前的state-of-art目标检测方法并没有显式地利用contextual information，但近来也有一些显式利用contextual information的DCNN方法，可分为2类：Global context和Local context。

![context information](https://s2.ax1x.com/2019/02/14/kBc7PP.png)

![Local Context](https://s2.ax1x.com/2019/02/14/kBcb28.png)
感觉可以在某种程度上看成是数据层面的集成学习。

## Detection Proposal Methods

Two stage detection framework需要生成ROI。

生成ROI的方法，可以分为**Bounding Box Proposal Methods**和**Object Segment Proposal Methods**，前者回归出Bounding Box来描述ROI，后者通过分割得到像素集合来描述ROI。
![object proposal methods](https://s2.ax1x.com/2019/02/14/kBgvQO.png)
![Region Proposal Network](https://s2.ax1x.com/2019/02/14/kB2peH.png)
## Other Special Issues 
通过data augmentation tricks（数据增广）可以得到更鲁棒的特征表示，可以看成是数据层面上的集成学习，考虑到物体尺度可大可小的问题，scaling是使用最多的数据增广方法。
![representative methods for training strategies and class imbalance handling](https://s2.ax1x.com/2019/02/14/kB56jP.png)
# Datasets and Performance Evaluation
![popular databases for object recognition](https://s2.ax1x.com/2019/02/14/kB5fAg.png)

![example images](https://s2.ax1x.com/2019/02/14/kB5XEF.png)
![Statistics of commonly used object detection datasets](https://s2.ax1x.com/2019/02/14/kB5zC9.png)
![metrics](https://s2.ax1x.com/2019/02/14/kBIpg1.png)
以上。