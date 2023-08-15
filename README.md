# The Simple Demo of RWKV to Help Easily Understand the Codes, Logics and Principles.

The aim is to learn new things and gain a deeper understanding of the neural network training process by replicating the demo of RWKV. Next, the focus is on comprehending the underlying principles of RWKV by analyzing the crucial code segments. 中文版本：https://zhuanlan.zhihu.com/p/647312884

## Environment Setup

1. A higher version of CUDA is needed for RWKV's demo, as version 11.0 may not be compatible and may result in errors. 
2. To set up the appropriate Visual Studio build tools for C++ development, download the 2022 version of Visual Studio and select the C++ development tools during installation.Next, add its path to the Path environment variable, such as 
"D:\VS2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"
3. Please note that .cu files and .cpp files should not contain Chinese characters; otherwise, errors may occur. If you encounter an error stating that "python38.lib" file is not found, follow these steps:
- Navigate to the installation path of Python and locate the "libs" folder.
- Copy the "python38.lib" file from the "libs" folder. Paste the copied "python38.lib" file into the "Scripts" directory of your virtual environment.
- If the error persists even after copying the "python38.lib" file, rename the duplicated file to "python38_d.lib".
4. Set the batch_size to 1 and the GPU memory will be limited to around 2GB.

### Importing the Libraries
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

## Prerequisite Knowledge

### Briefly Outlining the Updating Process of Neural Network Gradients
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
        # Define the forward pass of the sigmoid function in the forward method
        sigmoid_output = 1 / (1 + torch.exp(-input))
        ctx.save_for_backward(sigmoid_output)  # Save sigmoid output for gradient computation in backward pass
        return sigmoid_output

    @staticmethod
    def backward(ctx, grad_output):
        # Define the backward pass of the sigmoid function in the backward method,
        # computing gradients of the input tensor
        # grad_output represents the gradient computed by the next layer's calculations
        sigmoid_output, = ctx.saved_tensors
        grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)  # Derivative form of the sigmoid function
        return grad_input  # Return the gradient of the input, note: if forward takes multiple arguments,
                           # return gradients for each argument, similar to the WKV implementation in rwkv_demo_1.py

# Use the custom MySigmoidFunction
x = torch.tensor([1.0], requires_grad=True)
y = MySigmoidFunction.apply(x)
loss = y.sum()
loss.backward()

print(x.grad)  # The gradient of the tensor is tensor([0.1966])
```
Once the gradients are computed, you can update the model parameters using various gradient update strategies. Consequently, when defining the model, we can organize it in a layered manner, where the output of one layer becomes the input to the next layer. During forward propagation, we calculate the output values for each layer, and during backpropagation, we compute the gradients for each layer. Then, using an optimizer, we update the parameters based on these gradients to search for the minimum value of the loss function.

### Integrating CUDA Computation into PyTorch
CUDA computation (GPU kernel functions) is a parallel computing task that runs on the GPU. It typically utilizes the SIMD (Single Instruction, Multiple Data) execution model, which processes multiple data elements simultaneously at the same time. This approach can significantly accelerate the computation process, thereby improving computational performance and efficiency.

The parallelism in GPU computation is primarily manifested during the execution of kernel functions, where a large number of threads are simultaneously invoked to process data. Threads are the smallest execution units in GPU computing. Each thread independently executes the computational instructions in the kernel function, processing different data elements. Threads are typically organized into thread blocks according to specific rules, and these thread blocks are further arranged into a grid according to specific rules.


The kernel function is written using the ```__global__``` modifier to indicate that it will be executed on the GPU. The kernel function is executed in parallel by different threads, and each thread can access its unique thread ID to determine which part of the computation it should process. In common scenarios, CUDA kernel functions are often launched in the form of one-dimensional thread blocks, and each thread block processes one-dimensional data chunks. Therefore, we use ```int idx = blockIdx.x * blockDim.x + threadIdx.x``` to calculate the unique index of each thread in the one-dimensional data.

- blockIdx.x: Represents the index of the current thread block in the entire grid, where x indicates the index along the x-axis direction.

- blockDim.x: Represents the number of threads in each thread block, where x indicates the size along the x-axis direction of the thread block.

- threadIdx.x: Represents the index of the current thread within its corresponding thread block, where x indicates the index along the x-axis direction of the thread.

Typically, the ```dim3``` modifier is used to declare threadsPerBlock and numBlocks, which represent the number of threads and thread blocks, respectively. The kernel function is then launched on the GPU using the syntax ```kernel_fun<<<numBlocks, threadsPerBlock>>>(args)```, specifying the configuration for the kernel's execution. 

Here's an example from RWKV code:
```
void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y) {
    // Determine the number of threads per block and number of blocks
    dim3 threadsPerBlock( min(C, 32) ); // requires --maxrregcount 60 for optimal performance
    // Ensure that the total number of threads is divisible by the number of threads per block
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);  
    // Launch the forward pass kernel
    kernel_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y);
}
```
```min(C, 32)```: Here, the min function is used to compare C and 32 and select the smaller value as the size of the thread block. This is done to ensure that the thread block size does not exceed 32 because the maximum thread block size supported by each Streaming Multiprocessor (SM) in NVIDIA GPUs is usually 32. ```kernel_forward``` is a pre-defined kernel function.

To integrate CUDA computation into PyTorch, C++ code files (.cpp files) need to be implemented. In the .cpp file, include the ```torch/extension.h``` C++ header file, which contains necessary declarations and macros for writing PyTorch extensions and custom operations in C++ code. This includes Tensor types, Tensor operations, conversions between Tensor and C++ data types, and macros for creating PyTorch extensions (these macros enable easy exporting of C++ functions as Python interfaces and allow them to be recognized and used by PyTorch in the Python environment).

Using the ```PYBIND11_MODULE``` macro to create a PyTorch extension module, provided by PyTorch and the pybind11 library. PyTorch uses pybind11 to provide the C++ interface, enabling Python code to call and use C++ code. The ```PYBIND11_MODULE``` macro simplifies the process of exporting C++ functions as Python modules. The macro ```PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)``` serves as the entry point for the PyTorch extension, where ```TORCH_EXTENSION_NAME``` is the name of the extension module set as a macro definition during compilation, and ```m``` is a pybind11::module object used to bind C++ functions to Python interfaces. The general form is ```m.def("function_name", C++ implementation function pointer, "documentation", argument list)```.

Furthermore, using the ```TORCH_LIBRARY``` macro, C++ functions can be organized into a PyTorch extension library, making these functions callable as a library in the Python environment. ```TORCH_LIBRARY(wkv, m)``` is used, where wkv is the name of the extension library used when importing it in Python, and ```m``` is defined similarly. This also allows C++ programs to be integrated into PyTorch's just-in-time compilation (jit) process.

Here's an example from RWKV code:
```
#include <torch/extension.h>

// CUDA operator declarations
void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y);
void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *gy, float *gw, float *gu, float *gk, float *gv);

// C++ wrapper functions that invoke CUDA operators
void forward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
    cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>());
}
void backward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
    cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), gy.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv forward");
    m.def("backward", &backward, "wkv backward");
}

TORCH_LIBRARY(wkv, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
```

In PyTorch, ```w.data_ptr<float>()``` is a method used to obtain the data pointer of the tensor w. It returns a pointer to the floating-point data stored in the tensor w.

In conclusion, we can integrate CUDA computations into PyTorch and incorporate them into custom neural network layers by using ```torch.autograd.Function``` to define both the forward and backward passes. This enables the integration of CUDA computations into the neural network layers.

### Dynamic Computational Graph and Static Computational Graph

- Dynamic Computational Graph: It refers to the generation of a computation graph dynamically based on the actual input data during each forward pass. PyTorch is a framework that supports dynamic computational graphs, providing high flexibility and interpretability. In dynamic computational graphs, the computation graph is reconstructed every time during the forward pass, making it easy to trace and understand the computation process, which simplifies debugging and error checking. However, dynamic computational graphs require storing a large number of intermediate calculation results during the forward pass, leading to high memory consumption and difficulties in optimization.

- Static Computational Graph:It is a fixed computation graph defined during the model definition phase. TensorFlow is a representative framework that utilizes static computational graphs. In contrast to dynamic computational graphs, there is no need to reconstruct the computation graph during runtime, making static computational graphs more efficient in terms of performance. Additionally, the framework can perform static optimization on the entire computation graph during the compilation phase, thereby improving computational efficiency. Static optimization can include operations fusion, memory optimization, parallel computing, and other optimizations, which directly benefit every runtime execution. The drawback of static computational graphs is the lack of flexibility.

- In PyTorch, inheriting from the torch.jit.ScriptModule class and marking methods as scriptable (converting to static computational graphs) allows us to create scripted models, thereby improving the model's execution efficiency and facilitating deployment and execution. For example
```
import torch
import torch.nn as nn

class SimpleModel(torch.jit.ScriptModule):  
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    @torch.jit.script_method
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

## Loading Data
### Dataset
The enwki8 dataset is a byte-level dataset comprising the first 100 million bytes of Wikipedia XML dumps. This dataset is chosen due to several reasons: Wikipedia serves as an online encyclopedia, encompassing a wide range of subjects, including science, history, technology, culture, etc., resulting in diverse language structures within the dataset, ranging from simple words and short phrases to complex paragraphs and chapters. Efficient compression of such diverse content necessitates algorithms capable of handling various language structures. Wikipedia articles and content often exhibit high redundancy, especially across different pages, with numerous topics and definitions recurring across different articles. Effective compression algorithms should identify and exploit such redundancy to avoid duplicating similar information during the compression process. Wikipedia contains an abundance of proper nouns, specific topics, and internal links. These proper nouns may be unique and unattainable in other texts, thereby increasing the complexity of compression algorithms. Furthermore, the presence of internal links implies that compression algorithms must adeptly handle links and citations. The content within Wikipedia may encompass various text formats, such as tables, lists, citations, etc. The existence of these formats necessitates compression algorithms capable of processing and representing different types of data.

### Definition of ```Dataset``` Class
```
class RWKVDataset(Dataset):
    """
    Input: data - Dataset; ctx_len - Sentence length
    Attributes: self.vocab_size - Vocabulary size; self.data_size - Size of the dataset
    Output: x, y
    """

    def __init__(self, data, ctx_len):
        """
        This is a custom PyTorch Dataset class.
        """
        self.ctx_len = ctx_len  # Maximum text length
        self.data = data  # Original dataset
        self.data_type = str(type(self.data))  # Type of the original dataset

        unique_chars = sorted(list(set(data)))  # Sorted list of unique characters
        self.vocab_size = len(unique_chars)  # Vocabulary size
        self.stoi = {ch: i for i, ch in enumerate(unique_chars)}  # Token to ID mapping
        self.itos = {i: ch for i, ch in enumerate(unique_chars)}  # ID to token mapping
        self.data_size = len(self.data)  # Length of the dataset text
        print(f'Data has {self.data_size} tokens, {self.vocab_size} unique.')
        # Save vocab as json file
        with open('vocab.json', "w", encoding="utf-16") as vocab_file:
            json.dump(self.itos, vocab_file, ensure_ascii=False)  # Save vocabulary as JSON

    def __getitem__(self, _):
        """
        Returns a random sequence from the dataset.
        Randomly selects a start index start_idx from the data and retrieves a subsequence of length ctx_len + 1 from data,
        where the first ctx_len characters are the input x, and the next ctx_len characters are the output y.
        Converts x and y to PyTorch torch.tensor type and returns them.
        """

        start_idx = np.random.randint(0, self.data_size - (self.ctx_len + 1))  # Randomly select a start index
        sequence = [self.stoi[s] for s in self.data[start_idx:start_idx + self.ctx_len + 1]]
        x = torch.tensor(sequence[:-1], dtype=torch.long)  # Input IDs
        y = torch.tensor(sequence[1:], dtype=torch.long)  # Output IDs
        return x, y

    def __len__(self):
        return 10  # Returns the number of data samples in the dataset

```

The code of ```DataLoader```:
```
train_dataset = RWKVDataset(open(rwkv_config_1.datafile, "r", encoding="utf-8").read(), rwkv_config_1.ctx_len)
train_dataloader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=rwkv_config_1.batch_size)
```

In this context, a vocabulary of 6064 bytes is created by deduplicating and sorting characters from 100 million bytes. For a sample length of 8, 9 bytes are randomly selected from the dataset. The first 8 bytes constitute the input 'x', while the subsequent 8 bytes form the output 'y', as demonstrated by (x: [2,4,5,1,8,7,6,9], y: [4,5,1,8,7,6,9,5]). This arrangement enables the generation of sequential sequences where the goal is to predict the next word based on the previous ones.

## Defining RWKV Model

### Model Architecture
The architecture consists of: an Embedding layer, a LayerNorm layer, RWKV blocks (12 blocks), a LayerNorm layer, and a fully connected layer (with an output dimension equal to the vocabulary size).

- Embedding Layer: The ```nn.Embedding``` layer is utilized to transform a vocabulary into word vectors. Below is a brief analysis of its source code.
```
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        # Create an embedding matrix as a trainable parameter of the module
        # num_embeddings corresponds to vocabulary size
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input_indices):
        # Retrieve corresponding embedding vectors from the embedding matrix based on input indices
        # input_indices is an integer tensor where each value corresponds to an index of a token in the vocabulary
        # The result is a tensor with the same size as input_indices, with each index replaced by its corresponding embedding vector
        # These word vectors can then be updated through training
        return self.weight[input_indices]
```

- LayerNorm Layer: By normalizing each layer of a neural network, the distribution of input data is adjusted, thereby enhancing the model's stability, generalization capability, and training efficiency. It introduces learnable shift and scale parameters, enabling the model to adjust the distribution of normalized data adaptively. This allows the model to autonomously modify the importance of different feature dimensions. Below is a source code analysis, with PyTorch's default setting ```elementwise_affine=True```.

```
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape)) # Specific parameter initialization 
                                                                     # strategy may vary
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        variance = input.var(dim=-1, keepdim=True)
        output = (input - mean) / torch.sqrt(variance + self.eps)

        if self.elementwise_affine:
            output = output * self.weight + self.bias # Apply learnable scaling and 
                                                      # shifting parameters to the normalized result.

        return output

```
- RWKV Block: LayerNorm layer, TimeMix block, residual operation, LayerNorm layer, ChannelMix block, residual operation. In practice, these components are encapsulated into a Block class and applied using ```n.Sequential(*[Block(i) for i in range(rwkv_config.n_layer)])``` to sequentially define the model, where the asterisk (*) signifies unpacking, and ```rwkv_config.n_layer = 12```.

### Detailed Explanation of Small Init Embedding

```Small Init Embedding``` refers to the practice of initializing the parameters of the Embedding layer with very small values and adding an additional LayerNorm layer. This approach is primarily employed due to the slow variation of the embedding matrix, which poses challenges for the model to move away from its initial noisy embedding state. Consequently, the embedding matrix is initialized with very small values and then supplemented with a LayerNorm layer. This sequence of steps achieves the effect of "embedding matrix variation -> direction change -> significant LayerNorm variation," leading to improved results, as depicted in Figure 1.

![Figure 1: Effect of small initialization embedding from RWKV.](/pictures/851.png)

### Detailed Explanation of TimeMix Block and ChannelMix Block

The introduction of the RWKV model aims to leverage recurrent neural networks more effectively for large-scale autoregressive tasks. In autoregressive tasks, the model predicts the next element based on a portion of the already generated sequence, gradually producing the complete output sequence. Two key aspects are essential for this task: generating the next word and modeling contextual information from the generated sequence. Excelling at the second aspect is crucial to improve performance in the first.

In contrast to RWKV, the Transformer employs self-attention mechanisms to capture contextual information for predicting the next word. However, this approach presents two disadvantages: high temporal and spatial complexity (due to quadratic time complexity and the need to maintain KV cache for inference) and introducing noise through token-level modeling, which hampers information compression.

So, how does RWKV model contextual information? Through the TokenShift operation and WKV operation.

- TokenShift



TokenShift (time-shift mixing) refers to the fusion of the vector of the current token with the vector of the preceding token to create a new vector for the current token. The TokenShift operation is integrated within the TimeMix module and the ChannelMix module. In the demo code of this paper, we have 12 RWKV blocks, each containing a TimeMix block and a ChannelMix block. As a result, a total of 24 TokenShift operations are performed in the entire code framework. Since neural network layers can be seen as recursive functions, the information of the current token can be recursively learned from information far in the past. Naturally, shallow TokenShift operations learn local information, while deeper TokenShift operations learn more global information. As the author mentions on his GitHub, it makes sense if you think about it. I also found that you may prefer to use less mixing in higher layers.

```
time_shift = nn.ZeroPad2d((0, 0, 1, -1))
time_mix = nn.Parameter(torch.ones(1, 1, rwkv_config_1.n_embd))  # n_embd represents the token vector dimension
xx = time_shift(x)
x = x * time_mix + xx * (1 - time_mix)
```

The operation nn.ZeroPad2d((0, 0, 1, -1)) shifts the text to the left by one token and then adds the original text, achieving the TokenShift operation. Since the shape of x is (B, T, C), the parameters of nn.ZeroPad2d((0, 0, 1, -1)) correspond to (left, right, top, bottom) padding dimensions. In this case, it means no change in the left and right dimensions, adding 1 row of padding at the top in dimension 2, and reducing 1 row at the bottom in dimension 2, thereby achieving leftward shifting.

The time_mix parameter, a learnable parameter, represents the weights for feature fusion. It can be viewed as controlling the acceptance of information from the current token towards the existing sequence tokens.

- WKV




WKV is an adaptation of the self-attention mechanism from Transformers, realized using the principles of Recurrent Neural Networks (RNNs), integrated within the TimeMix module. Recurrent Neural Networks are mechanisms that enable the propagation of sequential information from one time step to the next. They employ a hidden state $h$ to learn contextual information from past sequence steps and update information at the current time step. This involves two primary operations: the reset gate and the update gate. The reset gate, $h_t = g(x_t, h_{t-1})$, utilizes information from the current time step to reset the hidden state's information, while the update gate, $\tilde{x}_t = f(x_t,h_t)$, employs information from the hidden state to update the current time step's information. The following explanation will be complemented with key code excerpts from the TimeMix module's inference process.

```
# Initialize hidden state
state = torch.zeros(rwkv_config_1.n_layer * 4, rwkv_config_1.n_embd)

# TimeMix
i = layer_id
xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)  # state[5*i+0] represents previous token information
xv = x * time_mix_v + state[5*i+0] * (1 - time_mix_v)  # time_mix is the tokenshift weight, determined during training
xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r) 
state[5*i+0] = x  # Update state[5*i+0]

r = torch.sigmoid(rw @ xr)  # rw is learned weight, rw @ xr represents linear transformation
k = kw @ xk  # kw is learned weight, kw @ xk represents linear transformation
v = vw @ xv  # vw is learned weight, vw @ xv represents linear transformation

kk = k
vv = v
aa = state[5*i+1]
bb = state[5*i+2]
pp = state[5*i+3]

# Update gate, RWKV paper equations (25-28)
ww = time_first[i] + kk  # time_first is learned positional variable controlling the update of the current token vector
p = torch.maximum(pp, ww)
e1 = torch.exp(pp - p)
e2 = torch.exp(ww - p)
a = e1 * aa + e2 * vv
b = e1 * bb + e2
wkv = a / b

# Reset gate, RWKV paper equations (29-32)
ww = pp + time_decay[i]  # time_decay is learned positional variable controlling the reset of the state matrix
p = torch.maximum(ww, kk)
e1 = torch.exp(ww - p)
e2 = torch.exp(kk - p)
state[5*i+1] = e1 * aa + e2 * vv
state[5*i+2] = e1 * bb + e2
state[5*i+3] = p

# Calculate rwkv
rwkv = ow @ (r * wkv)  # ow is learned weight
```

In the parameter initialization, time_first is initialized positively, while time_decay is initialized negatively. This approach helps avoid potential overflow issues caused by the max function when used in conjunction with the exp function. Referring to the RWKV paper's formula and carefully examining the key code snippet above, it becomes evident that the WKV implementation within the TimeMix module embodies the principles of a recurrent neural network (RNN). It achieves linear time complexity and reduces space complexity during the inference stage by solely preserving the state hidden state, which models contextual information from the ordered sequences. However, it is worth questioning whether relying solely on positional variables to control information update and reset is reasonable, and whether the problem of long-term forgetting inherent in recurrent neural networks is adequately addressed (although the TokenShift operation helps alleviate this concern).


So far, the TimeMix module of RWKV has addressed the second key point in autoregressive tasks. The subsequent ChannelMix module, which will be introduced next, addresses the first key point: generating the next word.

The ChannelMix module in RWKV is analogous to the Feed-Forward Network (FFN) module in Transformer. Its formula is as follows:$\begin{align}             r_t &= ({\mu}_r \odot o_t + (1-{\mu}_r) \odot o_{t-1}) \cdot  W_r \\             z_t &=  ({\mu}_z \odot o_t + (1-{\mu}_z) \odot o_{t-1}) \cdot  W_z \\             \tilde{x}_t &= \sigma (r_t) \odot (max(z_t, 0)^2 \cdot  W_v)         \end{align}$

The quantity $\sigma (r_t)$ after being passed through a sigmoid activation function lies within the interval (0, 1) and serves as the "forget gate," filtering which information should be retained as the vector representation for the next word through element-wise multiplication $\odot$ operation.


## Loss Function

The choice of the loss function naturally led to the selection of the cross-entropy loss function, with the addition of L2 regularization. However, the L2 regularization applied here pertains to the regularization of the probabilities produced by the model's output. Since the RWKV model's final layer consists of a fully connected layer, the probabilities generated as outputs can potentially be quite large, which may in turn impact the magnitude of the loss value. Thus, it becomes necessary to control the magnitudes of the output probabilities, thereby compelling the model to learn and adjust parameters that result in smaller output probability values.

The code implementation involves adding an additional neural network layer after computing the loss: $h = f(loss,x)$, where the gradient is $d(h)/d(loss)+d(h)/d(x) = 1 + d(h)/d(x)$ , where $x$ represents the output probabilities. As for $d(h)/d(x)$, its gradient is set to a small positive value at the maximum value of $x$, while the gradients for other values are set to zero. 

```
class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, x):
        ctx.save_for_backward(x)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        # The factor is a constant used to control the strength of L2 regularization.
        # It scales the gradient by dividing by the total number of elements in x, i.e., B * T.
        factor = 1e-4 / (x.shape[0] * x.shape[1])
        
        """
        Return a tensor maxx of shape [B, T, 1], where each element is the maximum value in the output x for that sample.
        Also, ids is a tensor of shape [B, T, 1], where each element is the index corresponding to the maximum value maxx.
        """
        maxx, ids = torch.max(x, -1, keepdim=True)
        
        """
        gx is a zero tensor with the same shape as x.
        Then, use the scatter_ function to distribute maxx * factor to the positions of the maximum values in gx for each sample.
        In other words, for each sample, set the gradient at its maximum value position to maxx * factor, and keep the gradients at other positions as zero.
        """
        gx = torch.zeros_like(x)
        gx.scatter_(-1, ids, maxx * factor)
        return (grad_output, gx)
```



