---
title: python中如何查看指定内存地址的内容
mathjax: false
date: 2020-02-08 22:45:03
tags:
categories:
- coding
---



python中一般并不需要查看内存内容，但作为从C/C++过来的人，有的时候还是想看看内存，有时是为了验证内容是否与预期一致，有时是为了探究下内存布局。

```python
from sys import getsizeof 
from ctypes import string_at

'''
getsizeof(...)
    getsizeof(object, default) -> int
    Return the size of object in bytes.
    
string_at(ptr, size=-1)
    string_at(addr[, size]) -> string
    Return the string at addr.
'''

```

`getsizeof`用于获取对象占用的内存大小，`string_at`用于获取指定地址、指定字节长度的内容，因为返回的对象类型是`bytes`，可以调用`hex()`函数转换成16进制查看。

对`int`对象的内存内容如下，首先通过函数`id`获取对象的内存地址。

```python
i = 100
type(i)
# int
s = string_at(id(i), getsizeof(i))
type(s)
# bytes
s
# b'>\x00\x00\x00\x00\x00\x00\x00\xa0\x99\xfd\x1d\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00d\x00\x00\x00'
s.hex()
# '3e00000000000000a099fd1d00000000010000000000000064000000'
```

如果对`int`对象的内存布局不熟悉，可能看不出什么。

再举一个`numpy`的例子。

```python
>>> import numpy as np
>>> a = np.arange(12).reshape(3,4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

>>> a.data
<memory at 0x00000000062483A8>
>>> m = a.data
>>> type(m)
memoryview
>>> m.hex()
'000000000100000002000000030000000400000005000000060000000700000008000000090000000a0000000b000000'

>>> a.ctypes.data
68393696
>>> string_at(a.ctypes.data, a.nbytes).hex()
'000000000100000002000000030000000400000005000000060000000700000008000000090000000a0000000b000000'


```

上面展示的两个例子，一个是通过`memoryview`对象查看，另一个是通过`string_at`查看。不是所有对象都支持`memoryview`，

> *class* `memoryview`(*obj*)
>
> Create a [`memoryview`](https://docs.python.org/3/library/stdtypes.html#memoryview) that references *obj*. *obj* must support the buffer protocol. Built-in objects that support the buffer protocol include [`bytes`](https://docs.python.org/3/library/stdtypes.html#bytes) and [`bytearray`](https://docs.python.org/3/library/stdtypes.html#bytearray).
>
> —— from https://docs.python.org/3/library/stdtypes.html#memoryview

但`string_at`，

> `ctypes.string_at`(*address*, *size=-1*)
>
> This function returns the C string starting at memory address *address* as a bytes object. If size is specified, it is used as size, otherwise the string is assumed to be zero-terminated.
>
> —— from https://docs.python.org/3/library/ctypes.html?highlight=string_at#ctypes.string_at

