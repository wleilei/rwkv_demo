# The simple demo of RWKV, which is easy to understand the code.

# Description
The original code （v_4）： [RWKV](https://github.com/BlinkDL/RWKV-LM)

本人小白学习记录，希望复现RWKV学习和应用新知识

# 环境配置：

1. 为C++配置合适的VS生成工具，旧版本网址在[这](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/)，官网为2022版本，下载2022版本即可，安装界面要选C++开发工具。将环境变量中的Path添加路径（如："D:\VS2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"），环境变量通过”系统、高级系统设置、环境变量、系统变量下面的Path“找到。
2. cuda版本需为11.8版本（我的11.0版本不能运行），不然可能会报错，下载安装cuda版本在[这](https://developer.nvidia.com/cuda-toolkit-archive)
3. 注意".cu"文件和".cpp"不能含有中文，否则会报错。出现找不到python38.lib文件错误，直接找到libs文件夹，复制到虚拟环境中的Scripts目录下。还是找不到，将该文件复制一份，重新命名为python38_d.lib即可。
4. 将batch_size设为1，显存2G左右
5. 文件夹下面需要生成一个“ __init__.py ”文件，这样python可以调用自己定义的库，如：import rwkv_config

# 代码架构
整个pytorch代码在rwkv_demo_1.py中，rwkv_config_1.py为参数文件。 pytorch代码遵循“数据处理，模型定义，损失函数定义，优化器，训练”的架构进行设计，接下来将分别进行简要讲解。

## 数据处理
1. **数据集**: [enwki8数据集](http://prize.hutter1.net/), 这是一个字节级的数据集，包含了Wikipedia XML转储的前1亿个字节。主要有以下原因：Wikipedia作为一个在线百科全书，涵盖了广泛的主题，包括科学、历史、技术、文化等，这使得数据集中包含了多样化的语言结构，从简单的单词和短句到复杂的段落和章节。对这样多样化的内容进行高效的压缩需要算法能够处理各种语言结构；Wikipedia中的文章和内容通常会有很高的冗余性，尤其是在不同页面之间。许多主题和定义在不同的文章中会反复出现。良好的压缩算法应该能够识别和利用这种冗余性，避免在压缩过程中重复存储相似的信息；Wikipedia中包含大量的专有名词、特定主题和内部链接。这些专有名词可能是独特的，不能在其他文本中找到，这增加了压缩算法的难度。此外，内部链接的存在意味着压缩算法必须能够有效地处理链接和引用；Wikipedia中的内容可能包含各种文本格式，如表格、列表、引用等。这些格式的存在使得压缩算法需要能够处理和表示不同类型的数据。
2. **数据处理**：将1亿个字节去重后排序的字节，作为词表；若样本长度为8，则从数据集中随机抽取9个字节，前8个字节作为x，后8个字节作为y，如(x:[2,4,5,1,8,7,6,9], y:[4,5,1,8,7,6,9,5])，进而能够实现上一个词预测下一个词的序列生成任务。
3. **nn.Embedding**: 使用nn.Embedding将词表转化为词向量，下面是其简要源码分析：
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

```

## 模型设计

### 模型架构
1. **整体架构**：Embedding层、layernorm层、RWKV块（12块）、layernorm层、全连接层（输出维度为词表长度）。应用“ nn.Sequential(*[Block(i) for i in range(rwkv_config.n_layer)]) ”以顺序方式定义模型，其中“ * ”号为拆包方式。

2. **RWKV块架构**：layernorm层、time-mix块、残差、layernorm层、channel-mix块、残差

3. **time-mix块**：tokenshift、计算（w、u、k、v、sr）、计算wkv（cuda操作）、sr*wkv

4. **channnel-mix块**：tokenshift、计算rkv(操作简单，不赘述，直接看源码)

### 模型细节

1. **tokenshift实现**： 使用“ nn.ZeroPad2d((0, 0, 1, -1)) ”操作将文本左移一个字，然后加上原始文本，进而实现tokenshift操作。因为x的shape为（B,T,C），“ nn.ZeroPad2d((0, 0, 1, -1)) ”的参数是（左，右，上，下）维度上补0操作，其意义是左右不变，在dim2的上方增1行补0，在dim2的下方缩减1行，进而实现左移。

2. 。。。

### 简述Pytorch梯度更新
**如何应用梯度更新参数**
1. 假设深度学习模型公式抽象成：L(Y,X) = f(AX+B,Y)，X为样本数据，Y为样本标签，L为损失函数。优化目标是寻找损失函数的最小值。
2. 因为X和Y是模型数据固定不动，所以要通过变化A和B来寻找最小值，即训练参数。
3. 计算出A的梯度，有两种情况：若梯度为正，则缩小A来寻找最小值；若梯度为正，则放大A来寻找最小值。所以，一般梯度下降法的核心为：A = A - A梯度，又为防止陷入局部最小值，设计了不同的更新策略。
4. 为什么不用到二阶导来进行更新参数呢？一方面是，计算一阶导数相对于计算二阶导数来说是相对简单的，特别是在高维空间和复杂模型中，计算Hessian矩阵需要更大的计算开销；另一方面是，Hessian矩阵是函数的二阶导数信息，它描述了函数在某一点处的曲率和凹凸性质。在优化问题中，Hessian矩阵的本征值可以提供有关函数形状的信息。正定的Hessian矩阵表示函数在该点是一个局部最小值，负定的Hessian矩阵表示函数在该点是一个局部最大值，而不定的Hessian矩阵则意味着该点可能是一个鞍点（saddle point），进而导致更新不稳定。
5. 那么pytorch如何进行梯度反向传播的呢？通过求导的链式法则构建计算图来进行反向传播。我们知道，搭建神经网络模型，通常是上一层的输出作为下一层的输入，即h(x) = g(f(x))，其中f()为第一层，g()为第二层。h对x进行链式求导可得：dh = dg * df。所以梯度的反向传播流程为：计算出下一层的梯度后，在计算上一层的梯度中直接乘下一层得出得梯度即可，举个例子（或者看rwkv_demo_1.py中的WKV类）：
```python
import torch

class MySigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 在forward方法中定义sigmoid函数的前向传播
        sigmoid_output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(sigmoid_output)  # 保存sigmoid输出用于反向传播计算梯度
        return sigmoid_output

    @staticmethod
    def backward(ctx, grad_output):
        # 在backward方法中定义sigmoid函数的反向传播，计算输入张量的梯度
        sigmoid_output, = ctx.saved_tensors
        grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)  # sigmoid函数的导数形式
        return grad_input  # 返回input的梯度，注意：forward中输入几个参数，
                           # 则返回几个参数的梯度，参考rwkv_demo_1.py中WKV的实现

# 使用自定义的MySigmoidFunction
x = torch.tensor([1.0], requires_grad=True)
y = MySigmoidFunction.apply(x)
loss = y.sum()
loss.backward()

print(x.grad)  # 输出张量的梯度为tensor([0.1966])

```
6. 动态计算图和静态计算图
- 动态计算图：是在每次前向传播时根据实际输入数据动态生成计算图，PyTorch是一个支持动态计算图的框架。这使得动态计算图具有很高的灵活性和可解释性。每次前向传播时，计算图都会重新构建，能够方便地跟踪和理解计算过程，使得调试和错误排查更加容易。但是，在前向传播过程中需要保存大量的中间计算结果，耗费内存高，并且难以优化。

- 静态计算图：是一种在模型定义阶段就固定的计算图，TensorFlow是一个代表性的静态计算图框架。在运行时不需要重复构建计算图，这使得静态计算图在性能方面更加高效。并且，框架可以在编译阶段对整个计算图进行静态优化，从而提高计算效率。静态计算图的编译优化可以包括操作融合、内存优化、并行计算等，这些优化在每次运行时都可以直接受益。缺点是缺乏灵活性。

- 在pytorch中，使用torch.jit.script装饰器将模型的前向传播函数标记为脚本化，转化为静态计算图。参考rwkv_demo_1.py中的RWKV_TimeMix类。


### Pytorch调用C++部署模型
1. 使用“ #include <torch/extension.h> ”C++头文件(C++为.cpp、cuda为.cu)，该头文件包含了一些必要的声明和宏定义，用于在C++代码中编写PyTorch的扩展和自定义操作。该头文件提供了以下功能：
- 定义 torch::Tensor 类型
- 定义 PyTorch 张量的操作
- 提供 PyTorch 张量和C++数据类型之间的转换
- 定义创建PyTorch扩展的宏：通过使用这些宏，可以方便地将C++函数导出为Python接口，并使其能够被PyTorch识别并在Python环境中使用。

2. 使用“ PYBIND11_MODULE ”宏来创建PyTorch扩展模块，是PyTorch和pybind11库提供的一个宏。PyTorch使用pybind11库来提供C++接口，从而使得Python代码能够调用和使用C++代码。PYBIND11_MODULE宏是pybind11库的一部分，它简化了将C++函数导出为Python模块的过程。PYBIND11_MODULE(TORCH_EXTENSION_NAME, m): 这是PyTorch扩展的入口宏，其中，“ TORCH_EXTENSION_NAME ”是扩展模块的名称，它在编译时被设置为宏定义；“ m ”是一个pybind11::module对象，它用于将C++函数和Python接口进行绑定，其一般形式为 m.def("函数名"，具体 C++ 实现的函数指针, "文档", 参数列表)。

3. 使用TORCH_LIBRARY宏，可以将C++函数组织成一个PyTorch扩展库，使得这些函数在Python环境中可以以库的形式调用。“ TORCH_LIBRARY(wkv, m) ”，其中，“ wkv ”是扩展库的名称，在Python中导入库时使用，“ m ”定义也一样。另外，通过这样也让C++程序能够融入pytorch的即时编译（jit）中。

4. “ torch.utils.cpp_extension.load ”函数用于加载和编译C++扩展模块，并将其导入到Python中。
- name：{类型：字符串； 含义：扩展模块的名称； 作用：指定C++扩展模块的名称，它将在编译和加载时使用。}
- sources：{类型：列表； 含义：扩展模块的源文件路径列表； 作用：指定C++扩展模块的源文件路径列表。这些源文件将被编译为共享库，用于创建Python模块。}
- extra_cflags：{类型：列表或None； 含义：额外的C编译标志； 作用：指定额外的C编译标志。这些标志将传递给C编译器，用于控制编译的行为。}
- extra_cuda_cflags：{类型：列表或None； 含义：额外的CUDA编译标志； 作用：指定额外的CUDA编译标志。这些标志将传递给CUDA编译器，用于控制编译的行为。}
- with_cuda：{类型：布尔值或None； 含义：是否编译包含CUDA代码的扩展； 作用：如果设置为True，将编译包含CUDA代码的扩展。如果设置为False，将不编译CUDA代码。如果为None，将根据系统环境自动决定是否启用CUDA。}
- verbose：{类型：布尔值； 含义：是否打印编译详细信息； 作用：如果设置为True，编译过程将打印详细信息。}






