---
title: 德布鲁因序列与indexing 1
mathjax: true
date: 2020-07-14 10:32:36
tags:
categories:
- 数据结构与算法
---



# 写在前面

在数值计算中，为了控制精度以及避免越界，需要严格控制数值的范围，有时需要知道二进制表示中"left-most 1"或"right-most 1”的位置，这篇文章就来介绍一下通过**德布鲁因序列（De Bruijn sequence）**来快速定位的方法。

# 标记left-most 1与right-most 1

对于一个二进制数$v$，如何仅保留最低位或最高位的1?

最低位的1，即right-most 1，其特点是这一位右侧均为0，可通过`v & -v`或者`v & ((~v)+1)`来标记最低位的1。

比如`0101 1010`，取反后为`1010 0101`，再加1为`1010 0110`，与后为`0000 0010`。

最高位的1，即left-most 1，其特点是这一位左侧均为0，可通过下面来标记最高位的1。

```cpp
uint32_t keepHighestBit( uint32_t n )
{
    n |= (n >>  1);
    n |= (n >>  2);
    n |= (n >>  4);
    n |= (n >>  8);
    n |= (n >> 16);
    return n - (n >> 1);
}
```

前5行移位将最高位1右侧的所有位均置为1，`n-(n >> 1)`再将他们清0。

至此，我们已经得到了一个二进制的“one hot”表示，只有1位为1，它标记了最高位或最低位1的位置。

# 确定位置

假设，得到的“one hot”表示为`0000 0100 0000 0000`，如何确定1在哪一位呢？

比较直接的想法是通过移位计数，不断右移，并计数，直到最低位为1。

有没有更好的方法？

令得到的“one hot”表示为`h`，对于`uint32`，`h`只有32种，我们希望找到的这32种one hot表示与$0\sim 31$的映射关系，即$f(h) \rightarrow 0\sim 31$。

- **查表**：以`h`对应的`uint32`数为下标，构建数组，通过查表方式得到，但`h`最大为$2^{31}$，直接构建数组不现实
- **哈希**：再增加一层映射，$f(g(h)) \rightarrow 0\sim 31$，即找到一个hash函数$g$，先将$h$映射到$0 \sim 31$，再通过查表$0\sim 31 \rightarrow 0\sim 31$，但一般哈希会涉及到取余操作，还要考虑不要有碰撞

对这个特殊问题，可以使用 德布鲁因序列——可视为一种特殊的哈希，不需要取余，且绝不会发生碰撞。

# 德布鲁因序列（De Bruijn sequence）

先看一个德布鲁因序列的例子，令字符集$A = \{0, 1\}$，字符有$k=2$种，子串长度$n=2$，则所有可能的子串有$\{00, 01, 10, 11\}$，则循环序列$0011$是一个德布鲁因序列，$0011$的所有连续子串恰好为$\{00, 01, 10, 11\}$，都出现且只出现一次，同样，循环序列$1001$也是一个德布鲁因序列。

![De_Bruijn_sequence.png](https://gitee.com/shinelee/ImgBed/raw/master/2020/德布鲁因序列/De_Bruijn_sequence.png)

可见，**德布鲁因序列并不唯一，且是个循环序列，长度恰好为$k^n$，与所有可能子串的数量相同**。

wiki上的定义如下，

> In [combinatorial](https://wiki2.org/en/Combinatorics) [mathematics](https://wiki2.org/en/Mathematics), a **de Bruijn sequence** of order $n$ on a size-$k$ [alphabet](https://wiki2.org/en/Alphabet_(computer_science)) *A* is a [cyclic sequence](https://wiki2.org/en/Cyclic_sequence) in which every possible length-$n$ [string](https://wiki2.org/en/String_(computer_science)#Formal_theory) on $A$ occurs exactly once as a [substring](https://wiki2.org/en/Substring) (i.e., as a *contiguous* [subsequence](https://wiki2.org/en/Subsequence)). Such a sequence is denoted by $B(k, n)$ and has length $k^n$, which is also the number of distinct strings of length $n$ on $A$.
>
> ——from wiki [De Bruijn sequence](https://wiki2.org/en/De_Bruijn_sequence)

再举一个$B(2, 4)$的例子，序列长度为$2^4=16$，如下
$$
0 0 0 0 1 1 1 1 0 1 1 0 0 1 0 1
$$
其所有循环子串如下，

![B-2-4.png](https://gitee.com/shinelee/ImgBed/raw/master/2020/德布鲁因序列/B-2-4.png)

每个位置的子串均不相同，所有子串对应着$0\sim 2^n-1$范围的整数，恰好形成了$2^n$个位置与$2^n$个数的映射。

# 德布鲁因序列的使用

将`h`与德布鲁因序列相乘，相当于左移操作，把某位置的子串移到了最左端，再将该子串右移至最右，即仅保留该子串，可知道该子串是什么，因为序列中每个子串的位置都是唯一的，根据映射关系可知道该子串的位置，相当于知道了`h`。为此需要建立 子串与位置 对应关系的检索表。

```cpp
unsigned int v;   
int r;           
static const int MultiplyDeBruijnBitPosition[32] = 
{
  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
  31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
};
r = MultiplyDeBruijnBitPosition[((uint32_t)((v & -v) * 0x077CB531U)) >> 27];
// The index of the LSB in v is stored in r

//return the index of the most significant bit set from a 32 bit unsigned integer
uint8_t highestBitIndex( uint32_t b )
{
    static const uint32_t deBruijnMagic = 0x06EB14F9;
    static const uint8_t deBruijnTable[32] = {
         0,  1, 16,  2, 29, 17,  3, 22, 30, 20, 18, 11, 13,  4,  7, 23,
        31, 15, 28, 21, 19, 10, 12,  6, 14, 27,  9,  5, 26,  8, 25, 24,
    };
    return deBruijnTable[(keepHighestBit(b) * deBruijnMagic) >> 27];
}
```

因为德布鲁因序列是循环序列，而左移操作会自动在最低位填0，所以习惯将全0子串放在序列的最高位，这样比较方便，不需要特殊处理。

# 德布鲁因序列的生成与索引表的构建

德布鲁因序列可以通过构建德布鲁因图得到，图中每条**哈密顿路径（Hamiltonian path）**都对应一个德布鲁因序列，

![De_Bruijn_binary_graph.svg.png](https://gitee.com/shinelee/ImgBed/raw/master/2020/德布鲁因序列/De_Bruijn_binary_graph.svg.png)

数量共有
$$
\frac{(k !)^{k^{n-1}}}{k^{n}}
$$
具体生成方式和证明可查看[De Bruijn sequence](https://wiki2.org/en/De_Bruijn_sequence)和[神奇的德布鲁因序列](https://halfrost.com/go_s2_de_bruijn/)。

保存子串与位置映射关系的检索表可通过如下方式生成，其中`debruijn32`为德布鲁因序列对应的`uint32`正整数。

```cpp
uint8 index32[32] = {0};
void setup( void )
{	
	int i;
	for(i=0; i<32; i++)
		index32[ (debruijn32 << i) >> 27 ] = i;
}
```



# 参考

- [De Bruijn sequence](https://wiki2.org/en/De_Bruijn_sequence)
- [神奇的德布鲁因序列](https://halfrost.com/go_s2_de_bruijn/)
- [De Bruijn Sequences for Fun and Profit](https://www.slideshare.net/alekbr/de-bruijn-sequences-for-fun-and-profit)
- [Bit mathematics cookbook](https://bisqwit.iki.fi/story/howto/bitmath/)