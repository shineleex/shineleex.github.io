---
title: Caffe源码理解2：SyncedMemory CPU和GPU间的数据同步
mathjax: true
date: 2018-12-01 16:50:49
tags:
- caffe
categories:
- 库与框架
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

# 写在前面
在Caffe源码理解1中介绍了`Blob`类，其中的数据成员有

```cpp
shared_ptr<SyncedMemory> data_;
shared_ptr<SyncedMemory> diff_;
```
`std::shared_ptr` 是共享对象所有权的智能指针，当最后一个占有对象的`shared_ptr`被销毁或再赋值时，对象会被自动销毁并释放内存，见[cppreference.com](https://zh.cppreference.com/w/cpp/memory/shared_ptr)。而`shared_ptr`所指向的`SyncedMemory`即是本文要讲述的重点。

在Caffe中，`SyncedMemory`有如下两个特点：

- **屏蔽了CPU和GPU上的内存管理以及数据同步细节**
- **通过惰性内存分配与同步，提高效率以及节省内存**

背后是怎么实现的？希望通过这篇文章可以将以上两点讲清楚。

# 成员变量的含义及作用

`SyncedMemory`的数据成员如下：
```cpp
enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
void* cpu_ptr_; // CPU侧数据指针
void* gpu_ptr_; // GPU侧数据指针
size_t size_; // 数据所占用的内存大小
SyncedHead head_; // 指示再近一次数据更新发生在哪一侧，在调用另一侧数据时需要将该侧数据同步过去
bool own_cpu_data_; // 指示cpu_ptr_是否为对象内部调用CaffeMallocHost分配的CPU内存
bool cpu_malloc_use_cuda_; // 指示是否使用cudaMallocHost分配页锁定内存，系统malloc分配的是可分页内存，前者更快
bool own_gpu_data_; // 指示gpu_ptr_是否为对象内部调用cudaMalloc分配的GPU内存
int device_; // GPU设备号
```

`cpu_ptr_`和`gpu_ptr_`所指向的数据空间有两种来源，一种是**对象内部自己分配的**，一种是**外部指定的**，为了区分这两种情况，于是有了`own_cpu_data_`和`own_gpu_data_`，当为`true`时表示是对象内部自己分配的，因此需要对象自己负责释放（析构函数），如果是外部指定的，则由外部负责释放，即**谁分配谁负责释放**。

外部指定数据时需调用`set_cpu_data`和`set_gpu_data`，代码如下：

```cpp
void SyncedMemory::set_cpu_data(void* data) {
  check_device(); 
  CHECK(data);
  if (own_cpu_data_) { // 如果自己分配过内存，先释放，换外部指定数据
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data; // 直接指向外部数据
  head_ = HEAD_AT_CPU; // 指示CPU侧更新了数据
  own_cpu_data_ = false; // 指示数据来源于外部
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) { // 如果自己分配过内存，先释放，换外部指定数据
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data; // 直接指向外部数据
  head_ = HEAD_AT_GPU; // 指示GPU侧更新了数据
  own_gpu_data_ = false; // 指示数据来源于外部
#else
  NO_GPU;
#endif
}
```

# 构造与析构

在`SyncedMemory`构造函数中，获取GPU设备（如果使用了GPU的话），注意构造时`head_ = UNINITIALIZED`，**初始化成员变量，但并没有真正的分配内存**。

```cpp
// 构造
SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

// 析构
SyncedMemory::~SyncedMemory() {
  check_device(); // 校验当前GPU设备以及gpu_ptr_所指向的设备是不是构造时获取的GPU设备
  if (cpu_ptr_ && own_cpu_data_) { // 自己分配的空间自己负责释放
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) { // 自己分配的空间自己负责释放
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

// 释放CPU内存
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}

```

但是，在析构函数中，却释放了CPU和GPU的数据指针，那么是什么时候分配的内存呢？这就要提到，Caffe官网中说的“**在需要时分配内存**” ，以及“**在需要时同步CPU和GPU**”，这样做是为了**提高效率**、**节省内存**。

> Blobs conceal the computational and mental overhead of mixed CPU/GPU operation by synchronizing from the CPU host to the GPU device as needed. Memory on the host and device is allocated on demand (lazily) for efficient memory usage.

具体怎么实现的？我们接着往下看。

# 内存同步管理

`SyncedMemory`成员函数如下：

```cpp
const void* cpu_data(); // to_cpu(); return (const void*)cpu_ptr_; 返回CPU const指针
void set_cpu_data(void* data);
const void* gpu_data(); // to_gpu(); return (const void*)gpu_ptr_; 返回GPU const指针
void set_gpu_data(void* data);
void* mutable_cpu_data(); // to_cpu(); head_ = HEAD_AT_CPU; return cpu_ptr_; 
void* mutable_gpu_data(); // to_gpu(); head_ = HEAD_AT_GPU; return gpu_ptr_;
enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
SyncedHead head() { return head_; }
size_t size() { return size_; }
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
```

其中，`cpu_data()`和`gpu_data()`返回`const`指针只读不写，`mutable_cpu_data()`和`mutable_gpu_data()`返回可写指针，它们4个在获取数据指针时均调用了`to_cpu()`或`to_gpu()`，两者内部逻辑一样，**内存分配发生在第一次访问某一侧数据时分配该侧内存**，如果不曾访问过则不分配内存，以此按需分配来节省内存。同时，用`head_`来指示**最近一次数据更新发生在哪一侧，仅在调用另一侧数据时才将该侧数据同步过去，如果访问的仍是该侧，则不会发生同步，当两侧已同步都是最新时，即**`head_=SYNCED`，**访问任何一侧都不会发生数据同步**。下面以`to_cpu()`为例，

```cpp
inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
  case UNINITIALIZED: // 如果未分配过内存（构造函数后就是这个状态）
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_); // to_CPU时为CPU分配内存
    caffe_memset(size_, 0, cpu_ptr_); // 数据清零
    head_ = HEAD_AT_CPU; // 指示CPU更新了数据
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU: // 如果GPU侧更新过数据，则同步到CPU
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) { // 如果CPU侧没分配过内存，分配内存
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_); // 数据同步
    head_ = SYNCED; // 指示CPU和GPU数据已同步一致
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU: // 如果CPU数据是最新的，不操作
  case SYNCED: // 如果CPU和GPU数据都是最新的，不操作
    break;
  }
}

// 分配CPU内存
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size)); // cuda malloc
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}
```

下面看一下`head_`状态是如何转换的，如下图所示：

![head_状态转换](https://s1.ax1x.com/2018/12/01/FnY2OU.png)

若以`X`指代CPU或GPU，`Y`指代GPU或CPU，**需要注意的是**，如果`HEAD_AT_X`表明`X`侧为最新数据，调用`mutable_Y_data()`时，在`to_Y()`内部会将`X`侧数据同步至`Y`，**会暂时将状态置为**`SYNCED`，但退出`to_Y()`后**最终仍会将状态置为**`HEAD_AT_Y`，如`mutable_cpu_data()`代码所示，

```cpp
void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

```

不管之前是何种状态，只要调用了`mutable_Y_data()`，则`head_`就为`HEAD_AT_Y`。背后的思想是，无论当前是`HEAD_AT_X`还是`SYNCED`，**只要调用了**`mutable_Y_data()`，**就意味着调用者可能会修改**`Y`**侧数据**，**所以认为接下来**`Y`**侧数据是最新的**，因此将其置为`HEAD_AT_Y`。

至此，就可以理解Caffe官网上提供的何时发生内存同步的例子，以及为什么建议不修改数据时要调用`const`函数，不要调用`mutable`函数了。

```cpp
// Assuming that data are on the CPU initially, and we have a blob.
const Dtype* foo;
Dtype* bar;
foo = blob.gpu_data(); // data copied cpu->gpu.
foo = blob.cpu_data(); // no data copied since both have up-to-date contents.
bar = blob.mutable_gpu_data(); // no data copied.
// ... some operations ...
bar = blob.mutable_gpu_data(); // no data copied when we are still on GPU.
foo = blob.cpu_data(); // data copied gpu->cpu, since the gpu side has modified the data
foo = blob.gpu_data(); // no data copied since both have up-to-date contents
bar = blob.mutable_cpu_data(); // still no data copied.
bar = blob.mutable_gpu_data(); // data copied cpu->gpu.
bar = blob.mutable_cpu_data(); // data copied gpu->cpu.
```

> A rule of thumb is, always use the const call if you do not want to change the values, and never store the pointers in your own object. Every time you work on a blob, call the functions to get the pointers, as the SyncedMem will need this to figure out when to copy data.

以上。

# 参考
- [Blobs, Layers, and Nets: anatomy of a Caffe model](http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html)
- [cudaMallocHost函数详解](https://blog.csdn.net/dcrmg/article/details/54975432)

