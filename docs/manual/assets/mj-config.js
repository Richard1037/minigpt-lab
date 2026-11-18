/* MathJax 配置必须在 mathjax 脚本之前设好。
   单独放一个文件，是为了让 md_to_html.py 的模板里不出现一堆转义大括号。 */
window.MathJax = {
  tex: { inlineMath: [['$', '$']], displayMath: [['$$', '$$']], processEscapes: true },
  svg: { fontCache: 'global' },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] },
  startup: { typeset: true }
};
