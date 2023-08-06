# The simple demo of RWKV, which is easy to understand the code.

The aim is to learn new things and gain a deeper understanding of the neural network training process by replicating the demo of RWKV. Next, the focus is on comprehending the underlying principles of RWKV by analyzing the crucial code segments.

## Environment Setup

1. A higher version of CUDA is needed for RWKV's demo, as version 11.0 may not be compatible and may result in errors. 
2. To set up the appropriate Visual Studio build tools for C++ development, download the 2022 version of Visual Studio and select the C++ development tools during installation.Next, add its path to the Path environment variable, such as 
"D:\VS2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
3. Please note that .cu files and .cpp files should not contain Chinese characters; otherwise, errors may occur. If you encounter an error stating that "python38.lib" file is not found, follow these steps:
- Navigate to the installation path of Python and locate the "libs" folder.
- Copy the "python38.lib" file from the "libs" folder. Paste the copied "python38.lib" file into the "Scripts" directory of your virtual environment.
- If the error persists even after copying the "python38.lib" file, rename the duplicated file to "python38_d.lib".
- Set the batch_size to 1 and the GPU memory will be limited to around 2GB.

### Importing the libraries
```
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import math
import os, types
```

### Configs
In Python, you can use the built-in module types and its SimpleNamespace class. This class allows us to create a simple namespace object where we can store multiple attributes (similar to key-value pairs in a dictionary), and these attributes can be accessed using dot notation. A namespace is a container that holds variable and function names, used to organize and manage these names in a program.
```
rwkv_config_1 = types.SimpleNamespace()
rwkv_config_1.datafile = os.path.join("data", "enwik8")

rwkv_config_1.batch_size = 1
rwkv_config_1.ctx_len = 1024

rwkv_config_1.lr = 0.00001
rwkv_config_1.betas = (0.9, 0.999)
rwkv_config_1.eps = 1e-8

rwkv_config_1.device = "cuda" if torch.cuda.is_available() else "cpu"

rwkv_config_1.n_embd = 512
rwkv_config_1.n_layer = 12
```

## Prerequisite knowledge

### Briefly outline the neural network gradient update process.
1. Assuming the deep learning model is abstracted as the formula: L(Y, X) = f(A, X, Y), where X represents the sample data, Y denotes the sample labels, A represents the model parameters, and L is the loss function value. The optimization objective is to find the minimum value of the loss function.
2. Exactly, since X and Y are fixed (representing the input data and labels), the objective is to find the minimum value of the loss function L by adjusting the model parameters A. By changing the trainable parameters (also known as training parameters), the model aims to minimize the loss function, leading to an optimized model that best fits the given data and improves its performance on the task at hand. This process is known as training or optimization, and it involves using various algorithms like gradient descent and its variants to iteratively update the parameters A until convergence to the optimal values that yield the minimum loss.
3. Generally, gradient descent is used to update the parameters, and there are variants such as stochastic gradient descent (SGD), Adam algorithm, and others. The reason for using gradients (derivatives) is that they represent the direction of the function value change. Computing the gradient of A can result in two scenarios: if the gradient is positive, reducing A will decrease L, leading to finding the minimum value; if the gradient is negative, increasing A will decrease L, again helping find the minimum value.Therefore, the core template of gradient descent is often represented as: A = A - learning_rate * A_gradient. By following this update rule, we can adjust the parameters to minimize the loss function. To avoid getting stuck in local minima during optimization, various update strategies have been designed, leading to the development of different optimization algorithms.
4. The reason for not using second-order derivatives for parameter updates lies in two aspects. Firstly, computing first-order derivatives (gradients) is relatively simple compared to computing second-order derivatives, especially in high-dimensional spaces and complex models. Calculating the Hessian matrix requires significantly larger computational overhead. Secondly, the Hessian matrix contains information about the second-order derivatives of the function, describing the curvature and convexity properties of the function at a specific point. In optimization problems, the eigenvalues of the Hessian matrix can provide information about the shape of the function. A positive-definite Hessian matrix indicates a local minimum, a negative-definite Hessian matrix indicates a local maximum, and an indefinite Hessian matrix suggests the point may be a saddle point. This can lead to unstable updates in the optimization process. For these reasons, first-order methods, such as gradient descent and its variants, are preferred in practice due to their simplicity, efficiency, and ability to handle complex optimization problems effectively. While second-order methods, such as Newton's method or variants like L-BFGS, can be used under certain conditions, they are less commonly used in deep learning due to the mentioned challenges and the large-scale nature of neural network optimization problems.
5. In PyTorch, gradient computation is performed by constructing a computation graph using the chain rule of derivatives for backpropagation. When building a neural network model, the output of one layer serves as the input to the next layer, i.e., h(x) = g(f(x)), where f() represents the first layer and g() represents the second layer. The chain rule for derivatives allows us to compute the gradient of h with respect to x as: dh = dg * df. Therefore, the process of backpropagating gradients can be summarized as follows: After computing the gradient of the next layer (dg), we can directly multiply it with the gradient of the previous layer (df) to obtain the gradient of that layer. Here's an example:
```
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
        # grad_output 为其下一层计算处理的梯度
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
Once the gradients are computed, you can update the model parameters using various gradient update strategies. Consequently, when defining the model, we can organize it in a layered manner, where the output of one layer becomes the input to the next layer. During forward propagation, we calculate the output values for each layer, and during backpropagation, we compute the gradients for each layer. Then, using an optimizer, we update the parameters based on these gradients to search for the minimum value of the loss function.






