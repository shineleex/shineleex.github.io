---
title: 远程桌面MATLAB启动失败问题解决
mathjax: false
date: 2019-12-14 15:53:25
tags:
categories:
- 工具
---



博客：[博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee) | [blog](https://blog.shinelee.me/)



远程桌面打开MATLAB会报错，解决办法，打开matlab的licenses路径，如`matlab/R2017b/licenses/`，路径下存有license文件，如`license_standalone.lic`（可能为其他名字），打开文件，在每行如下位置添加`TS_OK`。

![matlab license](https://s2.ax1x.com/2019/12/14/QRUpnK.png)

行数较多，可通过执行如下脚本自动添加，**注意，执行前先备份**

```python
license_path = './license_standalone.lic'

with open(license_path, 'r') as fr:
    lines = fr.readlines()

with open(license_path, 'w') as fw:
    for i, line in enumerate(lines):
        if (line[-2] == '\\'):
            continue
        lines[i] = line[:-2] + ' TS_OK\n'
    fw.writelines(lines)
```



# 参考

- [MATLAB远程桌面不可启动——解决方法](https://blog.csdn.net/u011631889/article/details/90409014)