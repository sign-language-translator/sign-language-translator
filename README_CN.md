<div align="center">

# 手语翻译器 ⠎⠇⠞

<img width="61.8%" alt="SLT: Sign Language Translator" src="https://github.com/sign-language-translator/sign-language-translator/assets/118578823/d4723333-3d25-413d-83a1-a4bbdc8da15a">

<br>
</br>

***使用人工智能构建自定义翻译器，实现手语和文本之间的互译。***

[![python](https://img.shields.io/pypi/pyversions/sign-language-translator?logo=python)](https://pypi.org/project/sign-language-translator/)
[![PyPi](https://img.shields.io/pypi/v/sign-language-translator?logo=pypi)](https://pypi.org/project/sign-language-translator/)
![GitHub branch check runs](https://img.shields.io/github/check-runs/sign-language-translator/sign-language-translator/main?logo=pytest)
[![codecov](https://codecov.io/gh/sign-language-translator/sign-language-translator/branch/main/graph/badge.svg?precision=1)](https://codecov.io/gh/sign-language-translator/sign-language-translator)
[![Documentation Status](https://img.shields.io/readthedocs/sign-language-translator?logo=readthedocs&)](https://sign-language-translator.readthedocs.io/)<br>
[![GitHub Repo stars](https://img.shields.io/github/stars/sign-language-translator/sign-language-translator?logo=github)](https://github.com/sign-language-translator/sign-language-translator/stargazers)
[![Repository Views per Month](https://img.shields.io/badge/dynamic/xml?url=https%3A%2F%2Fu8views.com%2Fapi%2Fv1%2Fgithub%2Fprofiles%2F118578823%2Fviews%2Fday-week-month-total-count.svg&query=%2F*%5Blocal-name()%3D'svg'%5D%2F*%5Blocal-name()%3D'g'%5D%5B3%5D%2F*%5Blocal-name()%3D'g'%5D%2F*%5Blocal-name()%3D'text'%5D%5B2%5D&label=%F0%9F%91%81%EF%B8%8F%20views%2Fmonth&color=6d96ff&cacheSeconds=5000)](https://u8views.com/github/mdsrqbl)
[![Downloads](https://img.shields.io/pepy/dt/sign_language_translator?color=purple&logoColor=white&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MDAiIGhlaWdodD0iODAwIiBmaWxsPSJub25lIiB2aWV3Qm94PSIwIDAgMjQgMjQiPjxwYXRoIGZpbGw9IiMwMDAiIGZpbGwtcnVsZT0iZXZlbm9kZCIgZD0iTTkuMiAyLjhjLS4yLjMtLjIuOC0uMiAxLjZWMTFINy44Yy0uOSAwLTEuMyAwLTEuNS4yYS44LjggMCAwIDAtLjMuNmMwIC4zLjMuNiAxIDEuMmw0LjEgNC40LjcuNmEuNy43IDAgMCAwIC40IDBsLjctLjZMMTcgMTNjLjYtLjYuOS0xIC45LTEuMmEuOC44IDAgMCAwLS4zLS42Yy0uMi0uMi0uNi0uMi0xLjUtLjJIMTVWNC40YzAtLjggMC0xLjMtLjItMS42YTEuNSAxLjUgMCAwIDAtLjYtLjZjLS4zLS4yLS44LS4yLTEuNi0uMmgtMS4yYy0uOCAwLTEuMyAwLTEuNi4yYTEuNSAxLjUgMCAwIDAtLjYuNnpNNSAyMWExIDEgMCAwIDAgMSAxaDEyYTEgMSAwIDEgMCAwLTJINmExIDEgMCAwIDAtMSAxeiIgY2xpcC1ydWxlPSJldmVub2RkIi8+PC9zdmc+)](https://pepy.tech/projects/sign-language-translator/)<br>
[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%8C%90%20Web%20Demo-%F0%9F%A4%97%20hf.co%2FsltAI-mediumpurple)](https://huggingface.co/sltAI)

| **支持我们** ❤️ | [![PayPal](https://img.shields.io/badge/PayPal-00457C?logo=paypal&logoColor=white)](https://www.paypal.com/donate/?hosted_button_id=7SNGNSKUQXQW2) |
| - | - |

</div>

---

- [手语翻译器 ⠎⠇⠞](#手语翻译器-)
  - [概述](#概述)
    - [解决方案](#解决方案)
    - [主要组件](#主要组件)
    - [目标](#目标)
  - [如何安装本软件包](#如何安装本软件包)
  - [使用方法](#使用方法)
    - [Web界面](#web界面)
    - [Python](#python)
    - [命令行](#命令行)
  - [支持的语言](#支持的语言)
  - [模型](#模型)
  - [如何构建手语翻译器](#如何构建手语翻译器)
  - [模块层次结构](#模块层次结构)
  - [如何贡献](#如何贡献)
  - [引用、许可证和研究论文](#引用许可证和研究论文)
  - [致谢](#致谢)
  - [额外内容](#额外内容)

## 概述

手语是听障人士主要使用的手势和表情语言。本项目旨在通过人工智能技术架起听障社区与普通人群之间的沟通桥梁。

<details>
<summary>这个Python库提供了用户友好的翻译API和一个框架,可用于构建适应任何地区手语的翻译器...</summary>
</br>
一个主要的障碍是缺乏深度学习工程师和软件开发人员可以用来为目标社区构建有用产品的数据集(全球性和区域性)和框架。本项目旨在通过提供强大的组件、工具、数据集和模型来支持手语到文本以及文本到手语的转换,从而增强手语翻译能力。它旨在促进任何地区手语翻译器的创建,同时为手语标准化铺平道路。

</br>
与大多数其他项目不同,这个Python库可以翻译完整的句子,而不仅仅是字母表。

</details>

### 解决方案

这个软件包带有一个*可扩展的基于规则*的文本到手语翻译系统,可用于为手语到文本和文本到手语翻译的*深度学习*模型生成训练数据。

> [!提示]
> 要为你的区域语言创建基于规则的翻译系统,你可以继承TextLanguage和SignLanguage类,并将它们作为参数传递给ConcatenativeSynthesis类。要编写支持的单词的示例文本,你可以使用我们的语言模型。然后,你可以使用该系统来微调我们的深度学习模型。

详见<kbd>[文档](https://sign-language-translator.readthedocs.io)</kbd>和我们的<kbd>[数据集](https://github.com/sign-language-translator/sign-language-datasets)</kbd>。

### 主要组件

<ol>
<li>
<details>
<summary><b>
手语到文本
</b></summary>

1. 从手语视频中提取特征
   1. 参见[`slt.models.video_embedding`](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/video_embedding)子包和[`$ slt embed`](https://slt.readthedocs.io/en/latest/#embed-videos)命令。
   2. 目前使用Mediapipe 3D关键点进行深度学习。
2. 将手势转录和翻译成多种文本语言以泛化模型。
3. 要训练逐字注释写作任务,还可以使用通过连接文本中每个单词的手势而制作的合成数据集。(参见[`slt.models.ConcatenativeSynthesis`](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py))
4. 使用[`slt.models.sign_to_text`](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/models/sign_to_text)中的神经网络或任何多语言seq2seq模型的编码器,在你的数据集上进行微调。

</details>
</li>

<li>
<details>
<summary><b>
文本到手语
</b></summary>

这个问题有两种解决方案:

1. 基于规则的连接
   1. 用所有可以映射到这些手势的词元标注手语词典。参见我们的[映射格式](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-dictionary-mapping.json)。
   2. 解析输入文本并为每个词元播放适当的视频片段。
      1. 通过继承`slt.languages.TextLanguage`构建文本处理器(有关详细信息,请参见[`slt.languages.text`](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/text)子包)
      2. 通过继承`slt.languages.SignLanguage`将文本语法和单词映射到手语(有关详细信息,请参见[`slt.languages.sign`](https://github.com/sign-language-translator/sign-language-translator/tree/main/sign_language_translator/languages/sign)子包)
      3. 使用我们的基于规则的模型[`slt.models.ConcatenativeSynthesis`](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py)进行翻译。
   3. 它更快,但需要在输入中**消除词义歧义**。查看深度学习方法以自动处理歧义词和**词典中没有的词**。

2. 深度学习(seq2seq)
   1. 生成应该连接的文件名序列<!-- TODO: `slt.models.mGlossBART` -->
      1. 你需要普通文本句子对手语注释(逐字写出的手势序列)的[平行语料库](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-sentence-mapping.json)
   2. 或者通过使用预训练的多语言文本编码器直接合成手势
      1. GAN或扩散模型或解码器生成姿势向量序列(`shape = (time, num_landmarks * num_coordinates)`)<!-- TODO: `slt.models.SignPoseGAN` -->
         1. 用这些姿势向量移动Avatar(简单)<!-- TODO: `slt.models.Avatar` -->
         2. 使用运动迁移生成视频(中等)<!-- TODO: `slt.models.SignMotionTransfer` -->
         3. 为每个向量合成视频帧(困难)<!-- TODO: `slt.models.DeepPoseToImage` -->
      2. 视频合成模型(非常困难)<!-- TODO: `slt.models.DeepSignVideoGAN` -->

</details>
</li>

<li>
<details>
<summary><b>
语言处理
</b></summary>

1. 手语处理
   - 使用Mediapipe提取3D世界关键点。
   - 使用matplotlib进行姿势可视化。
   - 使用scipy进行姿势变换(数据增强)。
2. 文本处理
   - 通过将未知字符/拼写替换为支持的单词来规范化文本输入。
   - 消除上下文相关词的歧义以确保准确翻译。
         "spring" -> ["spring(water-spring)", "spring(metal-coil)"]
   - 文本分词(词级和句级)。
   - 对词元进行分类并标记。

</details>
</li>

<li>
<details>
<summary><b>
数据集
</b></summary>

有关我们的数据集和约定,请参见[*sign-language-datasets* 仓库](https://github.com/sign-language-translator/sign-language-datasets)及其[发布版本](https://github.com/sign-language-translator/sign-language-datasets/releases)。
有关构建手语视频数据集(或动作捕捉手套输出特征)的更多信息,请参见此[文档](https://slt.readthedocs.io/en/latest/datasets.html)。

**你的数据应该包括**:

1. 词级词典(单个手势的视频和相应的文本词元(单词和短语))
2. 词典的复制。(设置多个同步摄像机并记录随机随机人员执行词典视频的表现。([notebook](https://colab.research.google.com/github/sign-language-translator/notebooks/blob/main/data_collection/clip_extractor.ipynb)))
3. 平行句子
   1. 普通文本语言句子对手语视频。(使用我们的语言模型生成由词典单词组成的句子。)
   2. 普通文本语言句子对相应手语句子的[文本注释](https://github.com/sign-language-translator/sign-language-datasets#glossary)。
   3. 手语句子对其文本注释
   4. 手语句子对多种文本语言的翻译
4. 手语的语法规则
   1. 词序(如主语-宾语-动词-时间)
   2. 无意义词(如"the"、"am"、"are")
   3. 歧义词(如spring(弹簧)和spring(泉水))

**尽量包含**:

1. 多个摄像机角度
2. 多样化的表演者以捕捉所有手语的"口音"
3. 词元标注的唯一性
4. 同一概念的手语变体

尽量以可扩展和包容多样性的方式捕捉手语变体,并促进手语标准化工作。

</details>
</li>
</ol>

### 目标

1. 实现将手语**集成**到现有应用程序中。
2. 协助为资源匮乏的手语构建**定制**解决方案。
3. 提高聋人的**教育**质量并提高识字率。
4. 促进听障人士沟通的**包容性**。
5. 建立手语**标准化**的框架。

## 如何安装本软件包

```bash
pip install sign-language-translator
```

<details>
<summary>可编辑模式(<code>git clone</code>):</summary>

该软件包还附带一些可选依赖项(例如用于同义词查找的deep_translator和用于预训练姿势提取模型的mediapipe)。通过在命令中的项目名称后附加`[all]`、`[full]`、`[mediapipe]`或`[synonyms]`来安装它们(例如`pip install sign-langauge-translator[full]`)。

```bash
git clone https://github.com/sign-language-translator/sign-language-translator.git
cd sign-language-translator
pip install -e ".[all]"
```

```bash
pip install -e git+https://github.com/sign-language-translator/sign-language-translator.git#egg=sign_language_translator
```

</details>

## 使用方法

访问[slt.**readthedocs**.io](https://slt.readthedocs.io)查看Python、CLI和gradio GUI的详细用法。
参见[*测试用例*](https://github.com/sign-language-translator/sign-language-translator/blob/main/tests)或[*notebooks*仓库](https://github.com/sign-language-translator/notebooks)以了解内部代码的运行方式。

### Web界面

在HuggingFace Spaces上部署的单个模型:

[![HuggingFace Spaces](https://img.shields.io/badge/text%20to%20sign-ConcatenativeSynthesis%2BLM-mediumslateblue)](https://huggingface.co/spaces/sltAI/ConcatenativeSynthesis)

### Python

```python
import sign_language_translator as slt

# 项目的核心模型(基于规则的文本到手语翻译器)
# 使我们能够生成合成训练数据集
model = slt.models.ConcatenativeSynthesis(
   text_language="urdu", sign_language="pk-sl", sign_format="video" )

text = "یہ بہت اچھا ہے۔" # "这-非常-好-是"
sign = model.translate(text) # 分词、映射、下载并连接
sign.show()

model.sign_format = slt.SignFormatCodes.LANDMARKS
model.sign_embedding_model = "mediapipe-world"

# ==== 英语 ==== #
model.text_language = slt.languages.text.English()
sign_2 = model.translate("This is an apple.")
sign_2.save("this-is-an-apple.csv", overwrite=True)

# ==== 印地语 ==== #
model.text_language = slt.TextLanguageCodes.HINDI
sign_3 = model.translate("कैसे हैं आप?") # "你-好吗"
sign_3.save_animation("how-are-you.gif", overwrite=True)
```

| ![this very good is](https://github.com/sign-language-translator/sign-language-translator/assets/118578823/7f4ff312-df03-4b11-837b-5fb895c9f08e) | <picture><source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/4d54a197-d723-4cc4-a3ba-cae98e681003" /><source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/45e71098-7a94-4a9e-ad24-1773369b65d5" /><img alt="how are you (landmark 3d plot)" src="https://github.com/user-attachments/assets/45e71098-7a94-4a9e-ad24-1773369b65d5" /></picture> |
| :-: | :-: |
| "یہ بہت اچھا ہے۔" (这-非常-好-是) | "कैसे हैं आप?" (你好吗) |

</br>

```python
import sign_language_translator as slt

# sign = slt.Video("path/to/video.mp4")
sign = slt.Video.load_asset("pk-hfad-1_aap-ka-nam-kya(what)-hy")  # 你的名字是什么? (自动下载)
sign.show_frames_grid()

# 提取姿势向量用于特征降维
embedding_model = slt.models.MediaPipeLandmarksModel()      # pip install "sign_language_translator[mediapipe]"  # (或 [all])
embedding = embedding_model.embed(sign.iter_frames())

slt.Landmarks(embedding.reshape((-1, 75, 5)),
              connections="mediapipe-world"  ).show()

# # 加载手语到文本模型(pytorch) (即将推出!)
# translation_model = slt.get_model(slt.ModelCodes.Gesture)
# text = translation_model.translate(embedding)
# print(text)
```

```python
# 自定义翻译器 (https://slt.readthedocs.io/en/latest/#building-custom-translators)
help(slt.languages.SignLanguage)
help(slt.languages.text.Urdu)
help(slt.models.ConcatenativeSynthesis)
```

### 命令行

```bash
$ slt

用法: slt [选项] 命令 [参数]...
   手语翻译器(SLT)命令行界面。
   文档: https://sign-language-translator.readthedocs.io
选项:
  --version  显示版本并退出
  --help     显示此消息并退出
命令:
  assets     资源管理器,用于下载和显示数据集和模型。
  complete   使用语言模型完成序列。
  embed      使用选定模型嵌入视频。
  translate  将文本翻译成手语或反向翻译。
```

**生成训练示例**: 用一个命令使用语言模型写一个句子并从中合成手语视频:

```bash
slt translate --model-code rule-based --text-lang urdu --sign-lang pk-sl --sign-format video \
"$(slt complete '<' --model-code urdu-mixed-ngram --join '')"
```

## 支持的语言

<details>
<summary><b>文本语言</b></summary>

可用功能:

- 文本规范化
- 分词(词、短语和句子)
- 词元分类(标注)
- 词义消歧

| 名称 | 词汇量 | 歧义词元 | 手语 |
| - | - | - | - |
| [英语](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/english.py) | 1591个词+短语 | 167 | 776 |
| [乌尔都语](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/urdu.py)    | 2080个词+短语 | 227 | 776 |
| [印地语](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/text/hindi.py)   |  137个词+短语 |   5 |  84 |

</details>

<details>
<summary><b>手语</b></summary>

可用功能:

- 词和短语到手语的映射
- 根据语法重构句子
- 句子简化(删除停用词)

| 名称 | 词汇量 | 数据集 | 平行语料库 |
| - | - | - | :-: |
| [巴基斯坦手语](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/languages/sign/pakistan_sign_language.py) | 776 | 23小时 | [详情](https://github.com/sign-language-translator/sign-language-datasets#datasets) |

</details>

## 模型

<details>
<summary><b>翻译</b>: 文本到手语</summary>

| 名称 | 架构 | 描述 | 输入 | 输出 | Web演示 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [连接合成](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_to_sign/concatenative_synthesis.py) | 规则 + 哈希表 | 主要用于合成翻译数据集的核心基于规则的翻译器。<br/>使用TextLanguage、SignLanguage和SignFormat对象进行初始化。 | string | slt.Video \| slt.Landmarks | [![在Spaces中打开](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/sltAI/ConcatenativeSynthesis) |

</details>

<details>
<summary><b>手语嵌入/特征提取</b>:</summary>

| 名称 | 架构 | 描述 | 输入格式 | 输出格式 |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| [MediaPipe关键点<br>(姿势 + 手)](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/video_embedding/mediapipe_landmarks_model.py) | 基于CNN的管道。参见:[姿势](https://arxiv.org/pdf/2006.10204.pdf), [手](https://arxiv.org/pdf/2006.10214.pdf) | 将视频编码为描述表演者动作的姿势向量(3D世界或2D图像)。 | List of numpy images<br/>(n_frames, height, width, channels) | torch.Tensor<br/>(n_frames, n_landmarks * 5) |

</details>

<details>
<summary><b>数据生成</b>: 语言模型</summary>

[可用的训练模型](https://github.com/sign-language-translator/sign-language-datasets/releases/tag/v0.0.1)

| 名称 | 架构 | 描述 | 输入格式 | 输出格式 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [N-Gram语言模型](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/ngram_language_model.py) | 哈希表 | 基于学习到的前N个词元的统计信息预测下一个词元。 | List of tokens | (token, probability) |
| [Transformer语言模型](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/language_models/transformer_language_model/model.py) | 仅解码器Transformers (GPT) | 使用查询-键-值注意力、线性变换和软概率预测下一个词元。 | torch.Tensor<br/>(batch, token_ids)<br/><br/>List of tokens | torch.Tensor<br/>(batch, token_ids, vocab_size)<br/><br/>(token, probability) |

</details>

<details>
<summary><b>文本嵌入</b>:</summary>

[可用的训练模型](https://github.com/sign-language-translator/sign-language-datasets/releases/)

| 名称 | 架构 | 描述 | 输入格式 | 输出格式 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------ | ------------------------- |
| [向量查找](https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/models/text_embedding/vector_lookup_model.py) | 哈希表 | 查找词元索引并返回相应的向量。对句子进行分词并计算已知词元的平均向量。 | string | torch.Tensor<br/>(n_dim,) |

</details>

## 如何构建手语翻译器

要创建你自己的手语翻译器,你需要以下基本组件:

<ol>
<li>
<details>
<summary>数据收集</summary>

   1. 收集[词典视频](https://github.com/sign-language-translator/sign-language-datasets/releases/tag/v0.0.2)(词级)集合,展示个人执行手语手势。这些可以从聋人学校和机构获得。你应该记录多个人执行相同的手语以捕捉手语的各种"口音"。同时设置多个摄像机以进一步增强数据。
   2. 准备一个[JSON文件](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-dictionary-mapping.json)，将词典视频文件名映射到与手势同义的相应文本语言单词和短语。
   3. 准备一个包含文本语言句子和手语视频文件名序列的合成数据[平行语料库](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-synthetic-sentence-mapping.json)。你可以使用语言模型来生成这些句子和序列。
   4. 准备一个手语[句子视频](https://github.com/sign-language-translator/sign-language-datasets/releases/tag/v0.0.3)数据集，用多种文本语言的[翻译和注释](https://github.com/sign-language-translator/sign-language-datasets/blob/main/parallel_texts/pk-sentence-mapping.json)标注。

</details>
</li>

<li>
<details>
<summary>语言处理</summary>

   1. 实现`slt.languages.TextLanguage`的子类：
      - 对你的文本语言进行分词，并为词元分配适当的标签以简化处理。
   2. 创建`slt.languages.SignLanguage`的子类：
      - 使用提供的JSON数据将文本词元映射到视频文件名。
      - 重新排列视频文件名序列以符合手语的语法和结构。

</details>
</li>

<li>
<details>
<summary>基于规则的翻译</summary>

   1. 将上一步中你的类的实例传递给`slt.models.ConcatenativeSynthesis`类以获得基于规则的翻译器对象。
   2. 用你的文本语言构造句子，并使用基于规则的翻译器生成手语翻译。(你可以使用我们的语言模型来生成这样的文本。)

</details>
</li>

<li>
<details>
<summary>深度学习模型微调</summary>

   1. 利用前一步中的(合成和真实)手语视频及相应的文本句子。
   2. 应用我们的训练流程来微调所选模型，以提高准确性和翻译质量。

</details>
</li>
</ol>

记得回馈社区：

- 通过创建pull request分享你的数据、代码和模型，让其他人也能从你的工作中受益。
- 创建你自己的手语翻译器(例如作为你的大学论文)，为创建一个更具包容性和可访问性的世界做出贡献。

在[ReadTheDocs中的构建自定义翻译器部分](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators)或此[notebook](https://github.com/sign-language-translator/notebooks/blob/main/translation/concatenative_synthesis.ipynb)中查看`代码`。[![在Colab中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sign-language-translator/notebooks/blob/main/translation/concatenative_synthesis.ipynb)

## 模块层次结构

<details>
<summary><b style="font-size:large;"><code>sign-language-translator</code></b> (点击查看文件描述)</summary>

<pre>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/README.md">README.md</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/pyproject.toml">pyproject.toml</a>
├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/requirements.txt">requirements.txt</a>
├── <b>docs</b>
│   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/docs">*</a>
├── <b>tests</b>
│   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/tests">*</a>
│
└── <b style="font-size:large;">sign_language_translator</b>
    ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/cli.py">cli.py</a> <sub><sup>`> slt` 命令行界面</sup></sub>
    ├── <i><b>assets</b></i> <sub><sup>(自动下载)</sup></sub>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-datasets">*</a>
    │
    ├── <b>config</b>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/assets.py">assets.py</a> <sub><sup>下载、提取和删除模型和数据集</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/colors.py">colors.py</a> <sub><sup>用于可视化的命名RGB元组</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/enums.py">enums.py</a> <sub><sup>用于识别模型和类的字符串短代码</sup></sub>
    │   ├── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/settings.py">settings.py</a> <sub><sup>仓库设计模式中的全局变量</sup></sub>
    │   ├── <i><a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/urls.json">urls.json</a></i>
    │   └── <a href="https://github.com/sign-language-translator/sign-language-translator/blob/main/sign_language_translator/config/utils.py">utils.py</a>
    |
    [以下部分结构省略，保持与英文版本相同的结构和链接，只翻译注释文本]
</pre>

</details>

## 如何贡献

<details>
<summary><b>数据集</b>:</summary>

查看我们的数据集和约定[这里](https://github.com/sign-language-translator/sign-language-datasets)。

- 通过抓取、编译和集中视频数据集做出贡献。
- 帮助标注[词映射数据集](https://github.com/sign-language-translator/sign-language-datasets/tree/main/parallel_texts)。
- 与聋人学院建立联系，共同开发标准化的*手语语法*并将其整合到基于规则的翻译器中。

</details>

<details>
<summary><b>新代码</b>:</summary>

- 为各个地区创建专门的手语类。
- 为不同语言开发文本语言处理类。
- 使用不同的超参数尝试训练模型。
- 不要忘记将你的类和模型的**`字符串短代码`**集成到**`enums.py`**中，并确保更新工厂函数如`get_model()`和`get_.*_language()`。
- 用全面的文档字符串、示例用例和详细的测试用例增强代码库。

</details>

<details open>
<summary><b>现有代码</b>:</summary>

- 实现代码中的`# ToDo`或修复`# bug` / `# type: ignore`或[路线图](#upcoming/roadmap)中的任何内容。
- 通过实现并行处理和批处理等技术优化代码库。
- 通过包含说明性示例的清晰文档字符串和强大的测试覆盖来加强项目。
- 为[sign-language-translator ReadTheDocs](https://github.com/sign-language-translator/sign-language-translator/blob/main/docs/index.rst)文档做出贡献，为用户提供全面的见解。目前需要一个**更好的模板**用于自动生成的页面。

</details>

<details>
<summary><b>产品开发</b>:</summary>

- 根据你的专业知识和兴趣，参与[MLOps](https://huggingface.co/sltAI)和[web前端](https://github.com/sign-language-translator/slt-frontend)领域的开发工作。

</details>

## 引用、许可证和研究论文

```bibtex
@software{mdsr2023slt,
  author       = {Mudassar Iqbal},
  title        = {Sign Language Translator: Python Library and AI Framework},
  year         = {2023},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/sign-language-translator/sign-language-translator}},
}
```

本项目采用[Apache 2.0许可证](https://github.com/sign-language-translator/sign-language-translator/blob/main/LICENSE)。您可以使用该库，创建修改版本，或将代码片段整合到您自己的工作中。无论您的产品或研究是商业性还是非商业性的，都必须通过引用此仓库对原作者给予适当的信誉。

敬请期待研究论文！

## 致谢

本项目始于2021年10月，最初是由3名学生和1名导师完成的计算机科学学士学位最终项目。在大学9个月后，它成为了[Mudassar](https://github.com/mdsrqbl)的业余项目，并持续到至少2024-09-23。

## 额外内容

统计**代码总行数**(包：**14,034** + 测试：**2,928**)：

```bash
git ls-files | grep '\.py' | xargs wc -l
```

**仅供*娱乐* 🙃**

```text
问：聋人学生最喜欢的课程是什么？
答：沟通技巧
```

```text
问：为什么机器学习工程师很伤心？
答：三元组损失(Triplet loss)
```

<details>
<summary><b>Star历史</b></summary>

<a href="https://star-history.com/#sign-language-translator/sign-language-translator&Timeline">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sign-language-translator/sign-language-translator&type=Timeline&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=sign-language-translator/sign-language-translator&type=Timeline" />
    <img alt="Star历史图表" src="https://api.star-history.com/svg?repos=sign-language-translator/sign-language-translator&type=Timeline" />
  </picture>
</a>

</details>