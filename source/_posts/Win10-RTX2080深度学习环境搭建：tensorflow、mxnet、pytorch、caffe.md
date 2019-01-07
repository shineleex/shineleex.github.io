---
title: Win10+RTX2080深度学习环境搭建：tensorflow、mxnet、pytorch、caffe
mathjax: false
date: 2018-12-26 17:42:49
tags:
categories:
- 库与框架
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

GPU为**RTX2080**，系统为更新到最新版本的**Win10**。

# 准备工作

- 安装**VS2015**，到官网地址[older-download](https://visualstudio.microsoft.com/vs/older-downloads/)下载安装
- 安装**Matlab**，笔者安装的是Matlab2017b
- 安装**Anaconda3-4.4.0-Windows-x86_64.exe**（到[anaconda archive](https://repo.anaconda.com/archive/)下载），笔者曾下载并安装了最新版的Anaconda3-2018.12-Windows-x86_64.exe，在使用conda安装包时发生SSLError错误，据[github issue](https://github.com/conda/conda/issues/6064)所说是最新版win10和最新版anaconda有冲突，4.4版本没有这个问题，4.4对应的**python版本为3.6**
- 安装**CUDA 10.0**，到[cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)根据自己的平台下载安装，安装好后可以看到环境变量多了CUDA_PATH、CUDA_PATH_10_0等，以及path中也多了bin和libnvvp路径
- 安装**CUDNN 7.4**，到[cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)根据CUDA版本下载安装，下载下来后将其中的所有文件（bin、include、lib文件夹）拷贝到cuda的目录，合并对应的文件夹

# 设置conda国内镜像源

为conda设置**国内镜像源**，默认国外的镜像源会比较慢。
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --set show_channel_urls yes
```
这时目录"C:\Users\用户名"下的.condarc文件内容变为

```bash
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
show_channel_urls: true
```
如果有default可以删除。

# conda 深度学习环境

```bash
# 创建conda环境
conda create -n py36DL python=3.6
# 更新pip
pip install --upgrade pip
# 若报错
easy_install pip
# 激活环境
activate py36DL
# 安装ipython，后面用于测试库是否安装成功
conda install ipython
```

# tensorflow、mxnet、pytorch安装

注意：**下面的所有安装都是在激活了的py36DL环境中进行的**。

## tensorflow
笔者通过[官网](https://tensorflow.google.cn/install/pip)、通过conda、通过豆瓣镜像源安装tensorflow在import时都会失败，报“ImportError: DLL load failed: 找不到指定的模块”的错误，最终成功的安装方式如下：

到[fo40225/tensorflow-windows-wheel](https://github.com/fo40225/tensorflow-windows-wheel)找到对应的版本下载whl，笔者下载的是**tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl**的avx2版本（有两个压缩包，解压出whl文件），如果安装不成功的话可以试试sse2版本，这里神奇的地方是该whl文件应该是在cuda100cudnn73avx2下编译的，但是我的环境是cuda100和cudnn74，竟然也是可以安装成功的，**笔者久经磨难，喜极而泣**。

下载下来后通过pip安装

```bash
# 切换到whl目录
pip install tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl

# 进入ipython验证
import tensorflow as tf
tf.VERSION
# '1.12.0'
```

## mxnet

mxnet安装比较简单，这里直接通过豆瓣镜像源用pip安装。
```bash
pip install -i https://pypi.doubanio.com/simple/ mxnet-cu100

# 进入ipython验证
import mxnet
mxnet.__version__
# '1.3.1'
```
mxnet的[官网](http://mxnet.incubator.apache.org/install/index.html?platform=Windows&language=Python&processor=GPU)显示支持到cu92，实际已经有了cu100版本。


## pytorch

pytorch的安装也很简单，在[官网](https://pytorch.org/get-started/locally/)，选择pip、Python3.6、CUDA10.0，显示

```bash
pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-win_amd64.whl
pip3 install torchvision
```
这里先把链接https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-win_amd64.whl拷贝到IDM下载whl文件，然后离线安装

```bash
# 切换到whl路径
pip install torch-1.0.0-cp36-cp36m-win_amd64.whl
pip install torchvision

#进入ipython验证
import torch
torch.__version
# '1.0.0'
```

# Caffe安装

笔者使用的是[happynear/caffe-windows](https://github.com/happynear/caffe-windows)版本的caffe，下载解压，同时下载第三方库拷贝到项目windows/thirdparty/文件夹，Copy .\windows\CommonSettings.props.example to .\windows\CommonSettings.props，打开Caffe.sln，根据github上的README修改配置文件.\windows\CommonSettings.props，编译成功后再参考README配置python和matlab，**注意使用时需要将thirdparty/bin目录添加到path环境变量，否则运行时会报错**。

修改环境变量后以管理员运行CMD，运行
```bash
echo %path%
```
立即生效，不用重启系统。

## 配置文件修改

主要修改项如下：
- **UseCuDNN**：true
- **CudaVersion**：10.0
- **PythonSupport**：true
- **MatlabSupport**：true
- **CudaArchitecture**：compute_50,sm_50;compute_52,sm_52;compute_60,sm_60;compute_61,sm_61;compute_70,sm_70;compute_75,sm_75;
- **CuDnnPath**：C:\Program Files\NVIDIA GPU Computing Toolkit\CUDNN-100。需替换为你的cudnn7.4的路径
- **PythonDir**：D:\Anaconda3\envs\py36DL\。替换为你的python路径
- **MatlabDir**：D:\Program Files\MATLAB\R2017b\。替换为你的matlab路径

根据[wiki](https://wiki2.org/en/CUDA)，RTX2080的Compute capability (version)为7.5，目前只有CUDA10.0支持7.5，因此CudaArchitecture中如果加入compute_75,sm_75;的话需要CUDA为10.0，否则会报错。如果装的是CUDA9.2，在不加compute_75,sm_75;的情况下也是可以编译成功的。


## 编译时常见错误

- **将警告视为错误**
在报错的工程上右键，选择 属性→C/C++→将警告视为错误，改为否，生成项目。要是某个项目文件报这个错的话，也可以在相应文件上右键，进行同样操作。

- **The Windows SDK version 10.0.10586.0 was not found**
在报错的工程上右键，选择 重定SDK版本目标，选择 目标平台版本（默认就一项 8.1），点击确定，生成项目。

## 运行时错误

- **Check failed: error == cudaSuccess (74 vs. 0) misaligned address**
根据[cuDNN bug in Caffe with "group" in conv layer (misaligned address) #5729](https://github.com/BVLC/caffe/issues/5729#issuecomment-378519210)，在 cudnn_conv_layer.cpp 文件`void CuDNNConvolutionLayer<Dtype>::Reshape`函数`size_t total_max_workspace = ...`代码前加入对齐代码如下
> this is a bug of Caffe, I solved it by modifying cudnn_conv_layer.cpp and aligning the address to be multiples of 32.
> You can insert tow lines of code before size_t total_max_workspace = ... as follow:
```cpp
size_t m=32;
max_workspace = (max_workspace + m-1) / m * m; //align address to be multiples of m
```
> BTW, I think there is another bug, these lines should be put in else block:
```cpp
for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
  workspace[g] = reinterpret_cast<char *>(workspaceData)+g*max_workspace;
}
```

# 参考
- [CUDA wiki](https://wiki2.org/en/CUDA)
- [happynear/caffe-windows](https://github.com/happynear/caffe-windows)
- [cuDNN bug in Caffe with "group" in conv layer (misaligned address) #5729](https://github.com/BVLC/caffe/issues/5729#issuecomment-378519210)


