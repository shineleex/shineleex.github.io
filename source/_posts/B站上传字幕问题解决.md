---
title: B站上传字幕问题解决
mathjax: false
date: 2019-05-23 13:58:33
tags:
- 脚本
categories:
- 工具
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

B站上传字幕时，如果srt文件中出现如下空行，则会报错，仅上传了空行前的部分
![srt file](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9zMi5heDF4LmNvbS8yMDE5LzA1LzIzL1ZDZm0xVS5wbmc?x-oss-process=image/format,png)
于是写了个python脚本，如下：

```python
import pysrt
import glob

srt_files = glob.glob('./*.srt')

for f in srt_files:
    subs = pysrt.open(f)
    for sub in subs:
        if sub.text == '':
            sub.text = ' '
    subs.save(f, encoding='utf-8')
```

解析srt文本，对象化为`subs`，判断当前字幕的文本是不是空串，如果是空串，变为空格，再保存文件。

这样上传字幕就不会出问题了。

pysrt github地址：https://github.com/byroot/pysrt

通过`pip install pysrt`安装。