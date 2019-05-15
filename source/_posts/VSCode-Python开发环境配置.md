---
title: VSCode Python开发环境配置
mathjax: false
date: 2019-01-07 17:14:09
tags:
- vscode
categories: 
- coding
---

博客：[blog.shinelee.me](https://blog.shinelee.me/) | [博客园](https://www.cnblogs.com/shine-lee/) | [CSDN](https://blog.csdn.net/blogshinelee)

# 准备工作

- **安装anaconda**，[官网](https://www.anaconda.com/)下载安装，笔者安装在"D:\Anaconda3"
安装好之后，查看环境变量path中是否有如下路径，没有的话添加进去
```powershell
D:\Anaconda3
D:\Anaconda3\Scripts
```
- **安装git**，[官网](https://git-scm.com/)下载安装，默认安装路径"C:\Program Files\Git"
- **安装VSCode**，[官网](https://code.visualstudio.com/)下载安装


# VSCode初步

查看[Visual Studio Code Tips and Tricks](https://code.visualstudio.com/docs/getstarted/tips-and-tricks)，快速熟悉VSCode。

## 用户界面
了解VSCode用户界面，如下图所示，随便点一点，还是比较一目了然的。
![vscode-userinterface](https://code.visualstudio.com/assets/docs/getstarted/userinterface/hero.png)

## 快捷键

Windows下的默认快捷键如下图所示，**万能Ctrl+Shift+P**。也可以 文件→首选项→键盘快捷方式，**自定义快捷键绑定**。
![vscode 快捷键](https://s2.ax1x.com/2019/01/05/F7uvTO.png)

# 安装扩展
如图搜索并安装相应扩展
![vscode-python 扩展](https://s2.ax1x.com/2019/01/05/F7n7sP.png)

- 安装**Chinese（Simplified）**中文简体语言包，参看官方文档[Display Language](https://code.visualstudio.com/docs/getstarted/locales)设置显示语言
- 安装**Python**扩展，如果前面安装的anaconda的路径已经加入到path环境变量中，这里跟着提示操作就可以，vscode会自动找到系统python的位置，调试时如果发现提示pylint没有安装，可以通过`pip`或者`conda`安装，参看[Linting Python in Visual Studio Code](https://code.visualstudio.com/docs/python/linting)
- 安装**Jupyter**、**Path Intellisense**、**vscode-python-docstring**等扩展，直接参看扩展说明以及[Working with Jupyter Notebooks in Visual Studio Code](https://code.visualstudio.com/docs/python/jupyter-support)即可，都很直观
- 安装**Settings Sync**，用于同步配置，将配置保存到github gist，参看扩展说明一步步操作即可，快捷键Shift + Alt + U上传配置

直接阅读扩展说明，即可知道每个扩展的用途。

安装好Python扩展后，按Ctrl+Shift+P，输入python→选择解析器，会显示所有环境（conda、venv等），可以选择任何一个作为解析器，如下图所示：

![vscode-python conda环境选择](https://code.visualstudio.com/assets/docs/python/environments/interpreters-list.png)

# 配置文件与内置终端设置

对于编辑器、窗口以及扩展等，VSCode都提供了默认配置，用户也可**自定义配置**，具体操作如下。

依次点击 文件→首选项→设置，或者直接`Ctrl+,`打开配置界面，通过右上角的按钮切换到 配置文件（见下图），左侧为默认配置，右侧为用户自定义配置，也可为当前工作区专门配置（会在当前文件夹下创建.vscode/settings.json文件）。

内置终端修改：默认内置终端为powershell，这里改为git bash。在左侧的默认配置项上点击“铅笔”图标可以将当前项复制到右侧进行修改，这里将**内置终端**修改为**git bash**，修改"terminal.integrated.shell.windows"和"terminal.integrated.shellArgs.windows"，如下图所示。
![vscode-settings](https://s2.ax1x.com/2019/01/05/F7Quyn.png)

修改完之后重启VSCode，会发现内置终端变成了bash，就可以使用`ll`等命令、运行sh脚本了，如下图所示。
![F7l7E6.png](https://s2.ax1x.com/2019/01/05/F7l7E6.png)

但是还存在一个问题，cmd激活conda环境的命令是`activate envname`，bash激活conda环境的命令为`source activate envname`，vscode在调试python时会自动调用`activate envname`来激活相应的环境，将默认终端换为bash后，会导致**环境激活不成功**，修改方法是在bash的配置文件中为`source activate`设置别名，具体如下：

- 打开"C:\Program Files\Git\etc\bash.bashrc"
- 在文件末尾加入如下两行：
```bash
alias activate=". $(which activate)"
alias deactivate=". $(which deactivate)"
```
重启vscode就可以了。

# 高级调试配置

即launch.json文件，在调试时，通常需要指定命令行参数或者临时环境变量等，这些都可以在launch.json文件中设置，具体怎么做呢？

高级调试配置需要通过VSCode打开文件夹，而不是直接打开文件，具体做法是：

- 在待调试文件所在的文件夹**右键**，选择 **open with code**
- **调试→添加配置**，会在当前文件夹下生成.vscode文件夹以及**.vscode/launch.json**文件（与工作去设置文件是同一文件夹）

打开launch.json文件，默认配置如下

```json
{
    "name": "Python: Current File (Integrated Terminal)",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal"
},
```

默认调试当前文件，默认调试终端为Integrated Terminal，即在vscode内置终端中调试。也可指定要launch的文件，直接修改上面"program"的值，将${file}替换为要调试的文件。

此外，还可添加其他配置项，常用的配置选项如下：
- `env`：指定环境变量
- `envFile`：指定环境变量定义文件，参见[Environment variable definitions file](https://code.visualstudio.com/docs/python/environments#_environment-variable-definitions-file)查看文件格式
- `args`：指定命令行参数

比如这样
```json
"env": {
    "CUDA_VISIBLE_DEVICES": "0"
},
"args": [
    "--port", "1593"
]
```
其他的配置项可参见[Set configuration options](https://code.visualstudio.com/docs/python/debugging#_set-configuration-options)。

# 小结
使用高效率生产力工具等于珍惜生命！现在可以愉快地coding了！
![vscode debug](https://s2.ax1x.com/2019/01/07/FbHlMq.png)

# 参考
- https://code.visualstudio.com/docs
- [Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
