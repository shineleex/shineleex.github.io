---
title: Protocol Buffers（1）：序列化、编译与使用
mathjax: false
date: 2019-04-13 16:48:01
tags:
- protobuf
categories:
- 库与框架
---

Protocol Buffers docs：https://developers.google.com/protocol-buffers/docs/overview
github：https://github.com/protocolbuffers/protobuf

# 序列化与反序列化
有些时候，我们希望给数据结构或对象拍个“快照”，或者保存成文件，或者传输给其他应用程序。比如，在神经网络训练过程中，我们会将不同阶段的网络权重以模型文件的形式保存下来，如果训练意外终止，可以重新载入模型文件将模型复原，继续训练。

将数据结构或对象以某种格式转化为字节流的过程，称之为**序列化（Serialization）**，目的是把当前的状态保存下来，在需要时复原数据结构或对象（序列化时不包含与对象相关联的函数，所以后面只提数据结构）。**反序列化（Deserialization）**，是序列化的逆过程，读取字节流，根据约定的格式协议，将数据结构复原。如下图所示，图片来自[geeksforgeeks](https://www.geeksforgeeks.org/serialization-in-java/)

![Serialization and Deserialization](https://s2.ax1x.com/2019/04/12/AbMBTA.png)

在介绍具体技术之前，我们先在脑海里分析下序列化和反序列化的过程：

- 代码运行过程中，数据结构和对象位于内存，其中的各项数据成员可能彼此紧邻，也可能分布在并不连续的各个内存区域，比如指针指向的内存块等；
- **文件中字节是顺序存储的**，要想将数据结构保存成文件，就需要把所有的数据成员平铺开（flatten），然后串接在一起；
- 直接串接可能是不行的，因为**字节流中没有天然的分界**，所以在序列化时需要按照某种约定的格式（协议），以便在反序列化时知道“**从哪里到哪里是哪个数据成员**”，因此格式可能需要约定：指代数据成员的标识、起始位置、终止位置、长度、分隔符等
- 由上可见，**格式协议是最重要的，它直接决定了序列化和反序列化的效率、字节流的大小和可读性等**

# Protocol Buffers概览

本文的主角**Protocol Buffers**，简称**Protobuf**，是谷歌开源的一项序列化技术，用官方语言介绍就是：

> **What are protocol buffers?**  
> Protocol buffers are Google's **language-neutral**, **platform-neutral**, extensible mechanism for serializing structured data – think XML, but **smaller**, **faster**, and **simpler**. 
> **You define how you want your data to be structured once**, then you can use **special generated source code** to easily write and read your structured data to and from a variety of data streams and using a variety of languages.

跨语言，跨平台，相比XML和JSON **更小、更快、更容易**，因为XML、JSON为了可阅读、自解释被设计成字符文本形式，所以体积更大，在编码解码上也更麻烦，而Protobuf序列化为binary stream，体积更小，**但是丧失了可读性——后面我们将看到可读性可以通过另一种方式得到保证。**至于上面的"**You define how you want your data to be structured once**"该怎么理解？参看下图，图片素材来自 [Protocol Buffers官网首页](https://developers.google.com/protocol-buffers/)。

![Protocol Buffers Example](https://s2.ax1x.com/2019/04/12/Ab5jDx.png)
首先是proto文件，在其中定义我们想要序列化的数据结构，如上图中的`message Person`，通过Protobuf提供的protoc.exe生成编解码代码文件（C++语言是.cc和.h），其中定义了类`Person`，类的各个成员变量与proto文件中的定义保持一致。序列化时，定义`Person`对象，对其成员变量赋值，调用序列化成员函数，将对象保存到文件。反序列化时，读入文件，将`Person`对象复原，读取相应的数据成员。

proto文件仅定义了数据的结构（name、id、email），具体的数据内容（1234、"John Doe"、"jdoe@example.com"）保存在序列化生成的文件中，通过简单的思考可知，序列化后的文件里应该会存在一些辅助信息用来将数据内容与数据结构对应起来，以便在反序列化时将数据内容赋值给对应的成员。

流程如下：
![Protocol Buffers Pipeline](https://s2.ax1x.com/2019/04/12/AbJX2n.png)

对Protobuf有了大致的了解后，我们来看看如何编译和使用Protobuf。

# Protocol Buffers C++ 编译
在 [github release](https://github.com/protocolbuffers/protobuf/releases) 下载对应版本的源码，参见 [cmake/README.md](https://github.com/protocolbuffers/protobuf/blob/master/cmake/README.md)查看如何通过源码编译，笔者使用的是VS2015，通过如下指令编译：

```bash
# 源码位于protobuf-3.7.1目录，cd protobuf-3.7.1/cmake
mkdir build
cd build
mkdir solution
cd solution
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=../../../../install ../.. -Dprotobuf_BUILD_TESTS=OFF
```
运行上面指令，会在solution目录下生成vs解决方案，编译整个解决方案，其中的INSTALL工程会生成install文件夹（位于protobuf-3.7.1/../install），内含3个子文件夹：

> - **bin** - that contains protobuf **protoc.exe** compiler;
> - **include** - that contains **C++ headers** and protobuf *.proto files;
> - **lib** - that contains linking **libraries** and CMake configuration files for protobuf package.

通过上面3个文件夹，我们就可以完成序列化和反序列化工作。

# Protocol Buffers C++ 使用
下面通过一个例子说明怎么使用Protobuf。

新建proto文件example.proto，添加内容如下：
```
package example;

message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;
}
```

每个filed的格式：
`required/optional/repeated FieldType FieldName = FieldNumber（a unique number in current message）`

> -  **Field Numbers** are used to identify your fields in the message binary format.
> - **required**: a well-formed message must have exactly one of this field.
> - **optional**: a well-formed message can have zero or one of this field (but not more than one).
> - **repeated**: this field can be repeated any number of times (including zero) in a well-formed message. The order of the repeated values will be preserved.

将example.proto文件复制到bin目录，运行如下指令：

```bash
protoc.exe example.proto --cpp_out=./
```

`--cpp_out`指定了生成cpp代码文件的目录，也可通过`--java_out`、`--python_out`等来指定其他语言代码生成的目录。上面指令会在当前目录下生成example.pb.cc和example.pb.h两个文件，其中命名空间`example`下定义了`Person`类，该类继承自`public ::google::protobuf::Message`，`Person`的数据成员含有`name_`、`id_`、`email_`，以及对应的`set`、`has`等成员函数。

接下来，在vs中新建一个测试工程，
- 将include目录添加到 附加包含目录，
- 将lib目录添加到 附加库目录，将lib文件添加到 附加依赖项，
- 将生成example.pb.cc 和 example.pb.h也添加到工程，
- 新建main.cpp，`#include "example.pb.h"`

添加如下内容：

```cpp
#include "example.pb.h"

int main()
{
    // Set data
    example::Person msg;
    msg.set_id(1234);
    msg.set_name("John Doe");
    msg.set_email("jdoe@example.com");

    // Serialization
    fstream output("./Person.bin", ios::out | ios::binary);
    msg.SerializePartialToOstream(&output);
    output.close();

    // Deserialization
    example::Person msg1;
    fstream input("./Person.bin", ios::in | ios::binary);
    msg1.ParseFromIstream(&input);
    input.close();

    // Get data
    cout << msg1.id() << endl; // 1234
    cout << msg1.name() << endl; // John Doe
    cout << msg1.email() << endl; // jdoe@example.com

    return 0;
}
```
上面代码将对象保存到Person.bin文件，在反序列化恢复对象。Person.bin文件内容如下：
![Person binary stream](https://s2.ax1x.com/2019/04/13/ALkZHx.png)

还是能看出一些规律的，字符串前1个字节表示的整数与字符串的长度相同，这是偶然吗？如果字符串很长，比如600个字符，超出1个字节能表示的范围怎么办？其他字节又是什么含义？

**这些问题，比如关于Protobuf是如何编码的，以及生成的cc和h文件代码细节，留到后面的文章介绍。**

# Protocol Buffers的可读性
二进制文件虽然体积更小，但其可读性无疑是差的，XML和JSON的优势之一就是可读性，**可读意味着可编辑、可人工校验**，Protobuf是不是就不能做到了呢？

并不是的，让我们继续在`main`函数中添加如下代码：

```cpp
#include "google/protobuf/io/zero_copy_stream_impl.h"

int main()
{
    // ……
    
    // Serialization to text file
    fw.open("./Person.txt", ios::out | ios::binary);
    google::protobuf::io::OstreamOutputStream *output = new google::protobuf::io::OstreamOutputStream(&fw);
    google::protobuf::TextFormat::Print(msg, output);
    delete output;
    fw.close();

    // Deserialization from text file
    example::Person msg2;
    fr.open("./Person.txt", ios::in | ios::binary);
    google::protobuf::io::IstreamInputStream input(&fr);
    google::protobuf::TextFormat::Parse(&input, &msg2);
    fr.close();

    // Get data
    cout << msg2.id() << endl; // 1234
    cout << msg2.name() << endl; // John Doe
    cout << msg2.email() << endl; // jdoe@example.com
}
```

这段代码是将对象保存成文本文件，再复原。打开文件Person.txt，其内容如下：

```json
name: "John Doe"
id: 1234
email: "jdoe@example.com"
```

和JSON是不是很像，也是类似的**key-value**对。

有了文本文件我们就可以直接阅读、校验和修改序列化后的数据，并且自如地在二进制文件和文本文件间转换，比如修改文本文件、恢复成对象、再导出二进制文件。

相信通过这篇文章，你已经对Protocol Buffer有了初步的了解，后续文章将深入介绍Protobuf的编解码和源码细节。

以上。

# 参考
- [5 Reasons to Use Protocol Buffers Instead of JSON For Your Next Service](https://codeclimate.com/blog/choose-protocol-buffers/)
- [eishay/jvm-serializers](https://github.com/eishay/jvm-serializers/wiki)
- [What is serialization?](https://stackoverflow.com/questions/633402/what-is-serialization)
- [Protocol buffers](https://developers.google.com/protocol-buffers/)
- [Protocol Buffer - A Walk Through For Beginners](https://www.c-sharpcorner.com/article/protocol-buffer-a-beginners-walk-through-moving-beyond-xml-and-json/)
- [google protocol buffers vs json vs XML [closed]](https://stackoverflow.com/questions/14028293/google-protocol-buffers-vs-json-vs-xml)