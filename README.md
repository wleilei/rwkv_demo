# The simple demo of RWKV, which is easy to understand the code.

# Description
The original code： [RWKV](https://github.com/BlinkDL/RWKV-LM)

本人小白学习记录

## 环境配置：

1. 为C++配置合适的VS生成工具，旧版本网址在[这](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/)，官网为2022版本，下载2022版本即可，安装界面要选C++开发工具。将环境变量中的Path添加路径（如："D:\VS2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"），环境变量通过”系统、高级系统设置、环境变量、系统变量下面的Path“找到。
2. cuda版本需为11.8版本（我的11.0版本不能运行），不然可能会报错，下载安装cuda版本在[这](https://developer.nvidia.com/cuda-toolkit-archive)
3. 注意".cu"文件和".cpp"不能含有中文，否则会报错。出现找不到python38.lib文件错误，直接找到libs文件夹，复制到虚拟环境中的Scripts目录下。还是找不到，将该文件复制一份，重新命名为python38_d.lib即可。

## 代码架构
整个pytorch代码在rwkv_demo_1.py中，rwkv_config.py为参数文件。 pytorch代码遵循“数据处理，模型定义，损失函数定义，优化器，训练”的架构进行设计，接下来将分别进行简要讲解。

### 数据处理
1. 数据集: [enwki8数据集](http://prize.hutter1.net/), 这是一个字节级的数据集，包含了Wikipedia XML转储的前1亿个字节。主要有以下原因：Wikipedia作为一个在线百科全书，涵盖了广泛的主题，包括科学、历史、技术、文化等，这使得数据集中包含了多样化的语言结构，从简单的单词和短句到复杂的段落和章节。对这样多样化的内容进行高效的压缩需要算法能够处理各种语言结构；Wikipedia中的文章和内容通常会有很高的冗余性，尤其是在不同页面之间。许多主题和定义在不同的文章中会反复出现。良好的压缩算法应该能够识别和利用这种冗余性，避免在压缩过程中重复存储相似的信息；Wikipedia中包含大量的专有名词、特定主题和内部链接。这些专有名词可能是独特的，不能在其他文本中找到，这增加了压缩算法的难度。此外，内部链接的存在意味着压缩算法必须能够有效地处理链接和引用；Wikipedia中的内容可能包含各种文本格式，如表格、列表、引用等。这些格式的存在使得压缩算法需要能够处理和表示不同类型的数据。
2. 数据处理：将1亿个字节去重后排序的字节，作为词表；若样本长度为8，则从数据集中随机抽取9个字节，前8个字节作为x，后8个字节作为y，如(x:[2,4,5,1,8,7,6,9], y:[4,5,1,8,7,6,9,5])，进而能够实现上一个词预测下一个词的序列生成任务。
3. nn.Embedding: 使用nn.Embedding将词表转化为词向量，下面是其简要源码分析：


```python
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        # 创建一个词嵌入矩阵作为模块的可训练参数
        # num_embeddings 为词表大小
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input_indices):
        # 根据输入的索引，从嵌入矩阵中查找对应的嵌入向量
        # input_indices 是一个整数张量，每个值对应词汇表中的一个标记的索引
        # 返回的结果是一个张量，其大小与 input_indices 一致，每个索引被替换为对应的嵌入向量
        # 然后词向量就可以通过训练进行更新
        return self.weight[input_indices]


