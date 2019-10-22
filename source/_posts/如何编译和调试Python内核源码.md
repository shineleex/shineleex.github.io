---
title: 如何编译和调试Python内核源码？
mathjax: false
date: 2019-10-16 09:57:26
tags:
categories:
- Python百问
---



博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

# 写在前面

**如果对Python源码感兴趣，那“窥探”其实现的最佳方式就是调教它，不，调试它。**

# 获取源代码

Python的官方默认实现为CPython，即C语言实现（主要指解释器的实现，其他实现见[Other Interpreter Implementations](https://devguide.python.org/#other-interpreter-implementations )）。CPython的源代码可以从官网[pyhton.org]( https://www.python.org/downloads/ )或者[ github.com/python/cpython ](https://github.com/python/cpython)获取，目前最新的稳定版本为3.8.0，于2019.10.14发布。这里，从官网 https://www.python.org/downloads/release/python-380/ 下载源码压缩包，如下图所示，

![python source code](https://s2.ax1x.com/2019/10/16/KiV44H.png)

# 源代码的组织

解压后，**目录结构**如下

```bash
{ Python-3.8.0 }  » tree -d -L 1 .
.
├── Doc 		# rst(reStructuredText)格式官方文档，用其生成https://docs.python.org/
├── Grammar		# Python的EBNF(Extended Backus–Naur form)语法定义文件
├── Include		# .h 头文件
├── Lib			# .py 纯Python实现的标准库
├── m4			# ？
├── Mac			# Mac-specific code，支持MacOS
├── Misc		# Things that do not belong elsewhere.
├── Modules		# C实现的标准库，内含.c .asm .macros .h
├── Objects		# 内置数据类型实现
├── Parser		# Python语法分析器源码
├── PC			# Windows-specific code，支持Windows
├── PCbuild		# Windows生成文件，for MSVC
├── Programs	# main函数文件，用于生成可执行文件，如python.exe的入口文件
├── Python		# CPython解释器源码
└── Tools		# 独立工具代码，used to maintain Python
```

CPython的源码组织结构如下，摘抄自[CPython Source Code Layout]( https://devguide.python.org/exploring/#cpython-source-code-layout )，

<img src="https://s2.ax1x.com/2019/10/16/KiG3Zj.png" alt="CPython Source Code Layout" style="zoom:80%;" />

源码文件分门别类存放，而且，无论是py实现的标准库、c实现的标准库、内置数据类型还是内置函数，在`Lib/test/`和`Doc/library/`目录下都有与之对应的test_x.py测试文件和rst文档文件（对于内置数据类型和函数，其文档集中保存在stdtypes.rst和functions.rst）。比如，内置类型`int`位于`Objects/longobject.c`文件中。

下面正式开始编译CPython。

# windows下编译CPython

据[Compile and build on Windows](https://devguide.python.org/setup/#windows )，Python3.6及之后的版本可以使用VS2017编译，安装VS2017时，记得勾选 **Python development** 和 **Python native development tools**，有备无患。

安装好VS2017后，双击`PCbuild/pcbuild.sln`，打开解决方案。因为我们的关注点仅在Python内核和解释器部分，所以仅编译python和pythoncore，其他模块暂时忽略，具体地，

- 切换到debug win32
- 右键解决方案→属性→配置属性
- 仅勾选项目python和pythoncore
- 确定

<img src="https://s2.ax1x.com/2019/10/16/KiDRqx.png" alt="vs2017 python build configuration" style="zoom:80%;" />

此时再“生成解决方案”，生成目录为`PCbuild/win32`，内容如下，含解释器python_d.exe和内核python38_d.dll，

![PCbuild build dir](https://s2.ax1x.com/2019/10/16/KirNfe.png)

接下来，将项目python设为启动项目（默认状态即是启动项目），点击**调试**，运行得到如下控制台，可以像平时使用python一样，与之交互。

![python38_d debug console](https://s2.ax1x.com/2019/10/16/KisW4O.png)

如果想生成全部模块，需要运行` PCbuild\get_externals.bat `下载依赖，再编译，具体可参见[Build CPython on Windows](https://cpython-core-tutorial.readthedocs.io/en/latest/build_cpython_windows.html )。

# 调试CPython

**只要程序能运行起来，一切就好办了。凭借“宇宙最强IDE”，我们可以任性地设断点调试甚至修改代码。**

`F5`重新启动调试，弹出控制台。在上面我们知道`int`类型位于`Objects/longobject.c`文件，打开文件，简单浏览后在函数`PyObject * PyLong_FromLong(long ival)`入口处打个断点。然后，在弹出的控制台中输入`a = 1`来创建`int`对象，回车，程序停在了断点处，查看变量`ival`的值为1——恰为我们输入的数值，**这个函数会跟根据输入的C long int创建一个`int`对象，返回对象指针**。

![debug int](https://s2.ax1x.com/2019/10/16/Ki48iR.png)

再来看看函数调用堆栈，如下图所示，

![call stack](https://s2.ax1x.com/2019/10/16/Ki4qyT.png)

调用顺序从下至上，从中可以推断出，

- 从python_d.exe的入口main运行起来后，进入python38_d.dll
- 从标准输入stdin中读取键入的字符串
- 解析字符串，建立了**语法树AST**（abstract syntax tree）
- 解析语法树中的节点，判断字符为number，将字符串转化为C long int
- 由C long int创建Python的`int`对象

继续运行，弹出的控制台中光标前出现`<<<`，等待输入。这时如果我们点击调试中的停止按钮（全部中断），会发现程序停在`Parser/myreadline.c`文件`_PyOS_WindowsConsoleReadline`函数中的`ReadConsoleW`一行，

```c
if (!ReadConsoleW(hStdIn, &wbuf[total_read], wbuflen - total_read, &n_read, NULL)) {
    err = GetLastError();
    goto exit;
}
```

`ReadConsoleW`为WINAPI，详见[ReadConsole function](https://docs.microsoft.com/en-us/windows/console/readconsole )，其等待并读取控制台的输入，读取的字符保存在`wbuf`中。如果有输入，则进入上面的流程，解析→建立语法树→……

# 小结

至此，我们揭开了Python面纱的一角——**不过是一个可运行、可调试的程序而已**（微笑）。

# 参考

- [Directory structure](https://devguide.python.org/setup/#directory-structure)
- [ reStructuredText ]( http://docutils.sourceforge.net/rst.html )
- [ Extended Backus–Naur form ]( https://wiki2.org/en/EBNF )
- [Exploring CPython’s Internals]( https://devguide.python.org/exploring/ )
- [Compile and build on Windows](https://devguide.python.org/setup/#windows )

