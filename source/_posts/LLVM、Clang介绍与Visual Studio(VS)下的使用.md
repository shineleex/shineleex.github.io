---
title: LLVM、Clang介绍与Visual Studio(VS)下的使用
date: 2018-06-27
tags: [编译器, 开发环境]
categories: [coding]
---

# 写在前面

因项目需要，想在VS中检测内存越界，检索到了[AddressSanitizer][1]，然后发现了Clang，又进一步找到了[LLVM][2]。借此，总结一下查阅到的相关内容。

# 编译与链接

在正式开始之前，插播一段编译器和链接器。
**[编译器][2]**主要用于将源代码从高级语言翻译成低级语言（汇编语言、目标代码、机器码），输出目标文件。
**[链接器][3]**主要用于将一个或多个目标文件（obj）与库文件（lib）合并成一个可执行文件（exe）或者库文件（lib、dll等）。

![编译与链接](http://p48vt5kn0.bkt.clouddn.com/blog/180404/3gKjGLG2m2.gif)

编译器可分为前端（front end）和后端（back end），两者以中间代码（IR，Intermediate Representation）为分界。
![编译器的前端和后端](http://p48vt5kn0.bkt.clouddn.com/blog/180404/fgjhCd423h.png?imageslim)

也可划分成前端、中端和后端，这里的**中端主要完成对IR的优化工作**，输出仍为IR。

# LLVM与Clang等

![LLVM_Logo](http://p48vt5kn0.bkt.clouddn.com/blog/180404/66cJ256gg1.png?imageslim)

[LLVM][4]全称为Low Level Virtual Machine，按wiki的说法，它是“a collection of **modular and reusable compiler** and **toolchain technologies**”，起初只支持C/C++，现已支持多种语言。提及LLVM可能指代的是LLVM project/infrastructure（框架，编译器各个环节对应项目的集合）、An LLVM-based compiler、LLVM libraries（库）、LLVM core（编译器的后端）、The LLVM IR，具体如下：

>  
- **The LLVM project/infrastructure**: This is an umbrella for several projects that, together, form **a complete compiler**: frontends, backends, optimizers, assemblers, linkers, libc++, compiler-rt, and a JIT engine. The word "LLVM" has this meaning, for example, in the following sentence: "LLVM is comprised of several projects".
- **An LLVM-based compiler**: This is **a compiler built partially or completely with the LLVM infrastructure**. For example, a compiler might use LLVM for the frontend and backend but use GCC and GNU system libraries to perform the final link. LLVM has this meaning in the following sentence, for example: "I used LLVM to compile C programs to a MIPS platform".
- **LLVM libraries**: This is the **reusable code portion** of the LLVM infrastructure. For example, LLVM has this meaning in the sentence: "My project uses LLVM to generate code through its Just-in-Time compilation framework".
- **LLVM core**: The **optimizations** that happen at the intermediate language level and the backend algorithms form the LLVM core where the project started. LLVM has this meaning in the following sentence: "LLVM and Clang are two different projects".
- **The LLVM IR**: This is the LLVM compiler **intermediate representation**. LLVM has this meaning when used in sentences such as "I built a frontend that translates my own language to LLVM".
——[What exactly is LLVM](https://stackoverflow.com/questions/2354725/what-exactly-is-llvm)

![LLVM Compiler Infrastructure](http://p48vt5kn0.bkt.clouddn.com/blog/180404/c6B5Ac33jm.png?imageslim)

而[Clang][5]呢？Clang是a C language family frontend for LLVM，是C-like语言的编译器前端，支持C, C++, Objective C/C++, OpenCL C等。后端使用LLVM，现已兼容GCC——[Clang.LLVM](https://clang.llvm.org/)。

# Visual Studio 下使用LLVM与Clang

官网上提供的方式

> 
To use the LLVM toolchain from Visual Studio, select a project in Solution Explorer, open its Property Page (Alt+F7 by default), and in the "General" section of "**Configuration Properties**" change "**Platform Toolset**" to "LLVM-vs2012", "LLVM-vs2013", etc.
Alternatively, invoke MSBuild with /p:PlatformToolset=LLVM-vs2013 to try out the toolchain without modifying the project files.
——[LLVM builds](http://llvm.org/builds/)

在VS2010中，设置如下，设置完成后，发现调试时遇到断点不停，需要将**调试信息格式**设置为“**程序设置库/Zi**”而不是“**用于编辑并继续的程序设置库/ZI**”。

![VS2010-LLVM](http://p48vt5kn0.bkt.clouddn.com/blog/180404/blekBDI6mg.png?imageslim)

![VS2010-LLVM-ZI](http://p48vt5kn0.bkt.clouddn.com/blog/180404/0mD4BjfJF2.png?imageslim)


[1]: https://github.com/google/sanitizers
[2]: https://wiki2.org/en/Compiler
[3]: https://wiki2.org/en/Linker_(computing)
[4]: http://llvm.org/
[5]: https://wiki2.org/en/Clang
