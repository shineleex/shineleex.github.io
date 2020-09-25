



- 用Pythonic方式来思考

  - Python3中的字符类型
    - bytes：字节流；`open('path', 'rb')`
    - str：Unicode；`open('path', 'r')`
  - 表达式如果变得复杂，就应该考虑将其拆解成小块，并命名为函数。不要一味追求过于紧凑的写法
  - 切片，考虑`itertools.islide`
  - 优先使用列表推导，而不是`map`和`filter`
  - 大列表 用生成器表达式，用`it = ()`做推导，用`next(it)`获取下一个
  - `enumerate()`把迭代器包装成生成器
  - `zip`同时迭代多个迭代器（不等长，会提前终止），可使用`itertools.zip_longest`
  - 异常处理：`try/except/else/finally`，`raise`

- 函数

  - 尽量用异常来表示特殊情况，而不要返回`None`

  - 闭包

    - 小心作用域bug，`nonlocal`，可改用**仿函数**（辅助类`__call__`）

  - 考虑用**生成器**改写直接返回列表的函数

  - `*args`令函数接受数量可变的位置参数

  - 用`None`作为可变类型参数的形式上的默认值

    > default参数的默认值仅在模块加载时评估一次，凡是以默认形式调用函数的代码，共享统一份可变参数，请设置为`None`，若设置为`[]`或`{}`可能产生奇怪的行为

  - 参数列表中`*`表示位置参数就此终结，后面的参数只能采用关键字参数来指定

- 类与继承

  - `namedtuple`
  - 用函数做hook（作为函数参数），可以为函数、带闭包的函数、仿函数（类重构了`__call__`）
  - `@classmethod`，第一个参数为`cls`，表示类本身，可引用类属性，可通过`cls`实例化对象返回，可提供比`__init__`更复杂的构造器操作
  - 用`super().__init__()`来初始化超类，超类初始化的顺序由`mro`（method resolution order）决定
  - 多用`public`属性，少用`private`属性（会被编译器重命名，多重继承时可能埋bug），多用`protected`
  - 编写自制的容器时，可以从`collections.abc`模块的抽象基类中继承，简单的话可以直接从`list`、`dict`继承

- 元类与属性

  - `@property`，实现`getter`和`setter`
  - 

