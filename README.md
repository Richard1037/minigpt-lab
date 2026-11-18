<picture>
  <img src="webapp/minigpt-lab.svg" alt="miniGPT Lab 标志" width="72" height="72">
</picture>

# miniGPT Lab

用 **Python + NumPy** 从张量运算开始，亲手实现一个能训练、保存和续写文本的小型 GPT。整个学习环境在本机运行，CPU 即可使用。课程把公式、交互图示、代码编辑和公开测评放在同一条学习路径上。

## 快速开始

需要 Python 3.10 或更新版本。在终端执行：

```sh
git clone https://github.com/Richard1037/minigpt-lab.git
cd minigpt-lab
python -m pip install -r requirements.txt
python webapp/pilot_server.py
```

打开 [本地 miniGPT Lab](http://127.0.0.1:8124/)，点顶部「目录」选择任务。第一次学习可以从「整体架构」开始，想先体验训练可以直接进入「训练实验室」。

## 课程路线

| 部分 | 你会理解或实现什么 |
| --- | --- |
| 01 整体架构 | 走一遍文本编码、模型打分、损失和自回归生成的数据流。 |
| 02 NumPy 与张量 | 练习形状、索引、广播、归约、矩阵乘法和拆合注意力头。 |
| 04 完整基础层 | 实现 Parameter、Module、Linear、Embedding、GELU、RMSNorm 和 MLP，复用它们组装分支。 |
| 05 概率与损失 | 实现稳定 Softmax 和交叉熵，并把梯度传回分数。 |
| 06 注意力与 Transformer | 依次实现因果注意力、多头投影和 Pre-Norm Block。 |
| 07 组装 GPT | 把字符编号、位置嵌入、Block 堆叠和共享输出权重接起来。 |
| 08 优化与训练 | 用交叉熵、全局梯度裁剪和 AdamW 实现训练闭环。 |
| 09 生成与保存 | 用温度和 top-k 采样完成滑窗续写，使用检查点工具保存与恢复。 |

第 03 部分的标量网络路线暂缓，相关资料保留在旧版手册。

每节先解释模块为什么存在、输入输出表示什么，再展示运算和实现任务。推导梯度时会就地列出对应的前向公式。动手清单和关键公式放在编辑器之前，样例和参考代码可按需展开。

## 试一次英文续写

1. 进入「训练实验室」，在常驻侧栏的「训练语料」中载入样例，或粘贴自己的英文文本。
2. 输入一个续写开头，例如 `the cat sits`。可以加到三条，同一次训练所得的模型会分别续写。
3. 点「训练并续写」。先看训练和验证损失，再逐字符查看结果及每一步实际使用的采样概率。

侧栏可以设定模型规模、训练步数、学习率和采样参数。更换自填语料时，默认用同一语料的末尾 20% 验证。训练结果还提供前向激活 RMS、裁剪前梯度范数和实际参数更新幅度，帮助定位变化发生在哪个环节。

当前网页实验由平台调度并调用仓库里的课程组件。独立训练程序的组装与网页接入是最后的个人实践，参考入口见 [train_my_gpt_reference.py](solutions/train_my_gpt_reference.py)。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| [webapp/](webapp/) | 本地网站、训练实验室、课程数据及离线 KaTeX 资源 |
| [exercises/](exercises/) | 学习者实现、公开测评和训练实验调度 |
| [solutions/](solutions/) | 模块参考实现及训练、续训和生成的参考程序 |
| [scripts/](scripts/) | 课程构建工具 |
| [tests/](tests/) | 前端交互、计算行为及课程接线的回归检查 |
| [docs/](docs/) | 教学规范、[项目决策摘要](docs/对话与项目决策摘要.md)和[旧版手册](docs/旧版动手手册.md) |

## 开发检查

在仓库根目录运行。前端检查需要 Node.js。

```sh
node --test "tests/*.cjs"
python -m unittest discover -s tests -p "test_transformer_course.py"
python -m unittest discover -s tests -p "test_learning_evolution.py"
```

浏览器保存编辑器草稿、测评进度和实验历史。本地练习写回时会创建备份。运行环境、临时目录、日志和备份由 Git 忽略。
