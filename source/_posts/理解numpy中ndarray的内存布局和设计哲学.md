---
title: 理解numpy中ndarray的内存结构
mathjax: false
date: 2020-02-10 22:14
tags:
categories:
- coding
---



博客：[博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee) | [blog](https://blog.shinelee.me/)





本文的主要目的在于理解`numpy.ndarray`的内存结构及其背后的设计哲学。

# ndarray是什么

> NumPy provides an **N-dimensional array** type, the [ndarray](https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.ndarray.html#arrays-ndarray), which describes a collection of “items” of **the same type**. The items can be [indexed](https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.indexing.html#arrays-indexing) using for example N integers.
>
> —— from https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.html

ndarray是numpy中的**多维数组**，数组中的元素具有**相同的类型**，且可以被**索引**。

如下所示：

```python
>>> import numpy as np
>>> a = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> type(a)
<class 'numpy.ndarray'>
>>> a.dtype   
dtype('int32')
>>> a[1,2]
6
>>> a[:,1:3]
array([[ 1,  2],
       [ 5,  6],
       [ 9, 10]])

>>> a.ndim    
2
>>> a.shape   
(3, 4)        
>>> a.strides 
(16, 4)       
```

注：`np.array`并不是类，而是用于创建`np.ndarray`对象的其中一个函数，numpy中多维数组的类为`np.ndarray`。

# ndarray的设计哲学

**ndarray的设计哲学在于数据存储与其解释方式的分离，或者说`copy`和`view`的分离，让尽可能多的操作发生在解释方式上（`view`上），而尽量少地操作实际存储数据的内存区域。**

如下所示，像`reshape`操作返回的新对象`b`，`a`和`b`的`shape`不同，但是两者共享同一个数据block，`c=b.T`，`c`是`b`的转置，但两者仍共享同一个数据block，数据并没有发生变化，发生变化的只是数据的解释方式。

```python
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> b = a.reshape(4, 3)
>>> b
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])

# reshape操作产生的是view视图，只是对数据的解释方式发生变化，数据物理地址相同
>>> a.ctypes.data
80831392
>>> b.ctypes.data
80831392
>>> id(a) == id(b)
false

# 数据在内存中连续存储
>>> from ctypes import string_at
>>> string_at(b.ctypes.data, b.nbytes).hex()
'000000000100000002000000030000000400000005000000060000000700000008000000090000000a0000000b000000'

# b的转置c，c仍共享相同的数据block，只改变了数据的解释方式，“以列优先的方式解释行优先的存储”
>>> c = b.T
>>> c
array([[ 0,  3,  6,  9],
       [ 1,  4,  7, 10],
       [ 2,  4,  8, 11]])
>>> c.ctypes.data
80831392
>>> string_at(c.ctypes.data, c.nbytes).hex()
'000000000100000002000000030000000400000005000000060000000700000008000000090000000a0000000b000000'
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

# copy会复制一份新的数据，其物理地址位于不同的区域
>>> c = b.copy()
>>> c
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
>>> c.ctypes.data
80831456
>>> string_at(c.ctypes.data, c.nbytes).hex()
'000000000100000002000000030000000400000005000000060000000700000008000000090000000a0000000b000000'

# slice操作产生的也是view视图，仍指向原来数据block中的物理地址
>>> d = b[1:3, :]
>>> d
array([[3, 4, 5],
       [6, 7, 8]])
>>> d.ctypes.data
80831404
>>> print('data buff address from {0} to {1}'.format(b.ctypes.data, b.ctypes.data + b.nbytes))
data buff address from 80831392 to 80831440

```



> **副本是一个数据的完整的拷贝**，如果我们对副本进行修改，它不会影响到原始数据，物理内存不在同一位置。
>
> **视图是数据的一个别称或引用**，通过该别称或引用亦便可访问、操作原有数据，但**原有数据不会产生拷贝**。如果我们对视图进行修改，它会影响到原始数据，物理内存在同一位置。
>
> **视图一般发生在：**
>
> - 1、numpy 的切片操作返回原数据的视图。
> - 2、调用 ndarray 的 view() 函数产生一个视图。
>
> **副本一般发生在：**
>
> - Python 序列的切片操作，调用deepCopy()函数。
> - 调用 ndarray 的 copy() 函数产生一个副本。
>
> —— from [NumPy 副本和视图](https://www.runoob.com/numpy/numpy-copies-and-views.html)

`view`机制的好处显而易见，**省内存，同时速度快**。

# ndarray的内存布局

> NumPy arrays consist of two major components, **the raw array data** (from now on, referred to as the data buffer), and **the information about the raw array data**. The data buffer is typically what people think of as arrays in C or Fortran, **a contiguous (and fixed) block of memory containing fixed sized data items**. NumPy also contains a significant set of data that describes **how to interpret the data in the data buffer**.
>
> —— from [NumPy internals](https://docs.scipy.org/doc/numpy-1.17.0/reference/internals.html)

ndarray的内存布局**示意图**如下：

![https://stackoverflow.com/questions/57262885/how-is-the-memory-allocated-for-numpy-arrays-in-python](https://s2.ax1x.com/2020/02/07/1gICjI.png)

可大致划分成2部分——对应设计哲学中的数据部分和解释方式：

- **raw array data**：为一个连续的memory block，存储着原始数据，类似C或Fortran中的数组，连续存储
- **metadata**：是对上面内存块的解释方式

metadata都包含哪些信息呢？

- `dtype`：**数据类型**，指示了每个数据占用多少个字节，这几个字节怎么解释，比如`int32`、`float32`等；
- `ndim`：有多少维；
- `shape`：每维上的数量；
- `strides`：**维间距**，即到达当前维下一个相邻数据需要前进的字节数，因考虑内存对齐，不一定为每个数据占用字节数的整数倍；

上面4个信息构成了`ndarray`的**indexing schema**，即**如何索引到指定位置的数据，以及这个数据该怎么解释**。

除此之外的信息还有：字节序（大端小端）、读写权限、C-order（行优先存储） or Fortran-order（列优先存储）等，如下所示，

```python
>>> a.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  WRITEBACKIFCOPY : False
  UPDATEIFCOPY : False
```

`ndarray`的底层是C和Fortran实现，上面的属性可以在其源码中找到对应，具体可见[PyArrayObject](https://docs.scipy.org/doc/numpy-1.17.0/reference/c-api.types-and-structures.html#c.PyArrayObject)和[PyArray_Descr](https://docs.scipy.org/doc/numpy-1.17.0/reference/c-api.types-and-structures.html#c.PyArray_Descr)等结构体。

# 为什么可以这样设计

为什么`ndarray`可以这样设计？

因为`ndarray`是为矩阵运算服务的，**`ndarray`中的所有数据都是同一种类型**，比如`int32`、`float64`等，每个数据占用的字节数相同、解释方式也相同，所以可以稠密地排列在一起，在取出时根据`dtype`现copy一份数据组装成`scalar`对象输出。这样极大地节省了空间，`scalar`对象中除了数据之外的域没必要重复存储，同时因为连续内存的原因，可以按秩访问，速度也要快得多。

![https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.html](https://s2.ax1x.com/2020/02/10/15qu7D.png)

```python
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> a[1,1]
5
>>> i,j = a[1,1], a[1,1]

# i和j为不同的对象，访问一次就“组装一个”对象
>>> id(i)
102575536
>>> id(j)
102575584
>>> a[1,1] = 4
>>> i
5
>>> j
5
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  4,  6,  7],
       [ 8,  9, 10, 11]])

# isinstance(val, np.generic) will return True if val is an array scalar object. Alternatively, what kind of array scalar is present can be determined using other members of the data type hierarchy.
>> isinstance(i, np.generic)
True
```

这里，可以将`ndarray`与python中的`list`对比一下，`list`可以容纳不同类型的对象，像`string`、`int`、`tuple`等都可以放在一个`list`里，所以`list`中存放的是对象的引用，再通过引用找到具体的对象，这些对象所在的物理地址并不是连续的，如下所示

![https://jakevdp.github.io/PythonDataScienceHandbook/02.01-understanding-data-types.html](https://s2.ax1x.com/2020/02/10/15j5dK.png)

所以相对`ndarray`，`list`访问到数据需要多跳转1次，`list`只能做到对对象引用的按秩访问，对具体的数据并不是按秩访问，所以效率上`ndarray`比`list`要快得多，空间上，因为`ndarray`只把数据紧密存储，而`list`需要把每个对象的所有域值都存下来，所以`ndarray`比`list`要更省空间。



# 小结

下面小结一下：

- `ndarray`的设计哲学在于**数据与其解释方式的分离，让绝大部分多维数组操作只发生在解释方式上**；
- `ndarray`中的**数据在物理内存上连续存储，在读取时根据`dtype`现组装成对象输出，可以按秩访问，效率高省空间**；
- 之所以能这样实现，在于`ndarray`是为矩阵运算服务的，所有**数据单元都是同种类型**。

# 参考

- [Array objects](https://docs.scipy.org/doc/numpy-1.17.0/reference/arrays.html)
- [NumPy internals](https://docs.scipy.org/doc/numpy-1.17.0/reference/internals.html)
- [NumPy C Code Explanations](https://docs.scipy.org/doc/numpy-1.17.0/reference/internals.code-explanations.html)
- [Python Types and C-Structures](https://docs.scipy.org/doc/numpy-1.17.0/reference/c-api.types-and-structures.html)
- [How is the memory allocated for numpy arrays in python?](https://stackoverflow.com/questions/57262885/how-is-the-memory-allocated-for-numpy-arrays-in-python)
- [NumPy 副本和视图](https://www.runoob.com/numpy/numpy-copies-and-views.html)