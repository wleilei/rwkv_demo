# The simple demo of RWKV, which is easy to understand the code.

# Description
The original code： [RWKV](https://github.com/BlinkDL/RWKV-LM)

环境配置：
1. cuda版本需为11.8版本，不然可能会报错，下载安装cuda版本在[这](https://developer.nvidia.com/cuda-toolkit-archive)
2. 为C++配置合适的VS生成工具，旧版本网址在[这](https://visualstudio.microsoft.com/zh-hans/vs/older-downloads/)，官网为2022版本，下载2022版本即可，安装界面要选C++开发工具。将环境变量中的Path添加路径（如："D:\VS2019\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"），环境变量通过”系统、高级系统设置、环境变量、系统变量下面的Path“找到。
3. 注意".cu"文件和".cpp"不能含有中文，否则会报错。出现找不到python38.lib文件错误，直接找到libs文件夹，复制到虚拟环境中的Scripts目录下

