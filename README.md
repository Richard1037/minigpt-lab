# miniGPT Lab

用 Python 和 NumPy，从张量运算一步步手写一个能训练、保存和续写文本的小型 GPT。课程网站在本机运行，包含公式讲解、可视化、代码编辑与公开测评。

## 启动

需要 Python 3.10 或以上版本。在仓库根目录执行：

```sh
python -m pip install -r requirements.txt
python webapp/pilot_server.py
```

打开 [miniGPT Lab](http://127.0.0.1:8124/)。从顶部目录进入课程或训练实验室。训练实验室可用英文字符语料训练并逐字符查看生成结果，训练会调用当前课程实现。

## 目录

- `webapp/` 包含网站、生成的课程资料和离线 KaTeX 资源。
- `exercises/` 包含本地练习、测评与训练实验的连接代码。
- `solutions/` 包含课程参考实现。
- `scripts/` 与 `tests/` 包含课程构建和验证工具。
- `docs/` 包含课程说明、[项目决策摘要](docs/对话与项目决策摘要.md)和[旧版手册](docs/旧版动手手册.md)。

## 开发检查

```sh
node --test tests/*.cjs
python -m unittest discover -s tests -p "test_learning_evolution.py"
```

编辑器草稿和成绩保存在当前浏览器。网站的本地运行时、临时目录、运行日志及个人备份由 Git 忽略。
