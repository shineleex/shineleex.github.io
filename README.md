# shineleex.github.io

## 20191213更新-解决latex渲染问题
参考https://segmentfault.com/q/1010000018176184
> npm un hexo-renderer-marked --save
npm i hexo-renderer-kramed --save


进入 node_modules\kramed\lib\rules\inline.js
> //escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
  escape: /^\\([`*\[\]()#$+\-.!_>])/,
//  em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
  em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,

## 解决hexo的MarkDown渲染器与MathJax的冲突
编辑 node_modules\marked\lib\marked.js
```js
escape: /^\\([\\`*{}\[\]()# +\-.!_>])/,
// 替换为
escape: /^\\([`*\[\]()# +\-.!_>])/,
// 取消对\\,\{,\}的转义


em: /^\b_((?:[^_]|__)+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
// 替换为
em:/^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
// 取消对对斜体标记 _ 的转义
```


