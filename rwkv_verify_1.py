""" 
测试文件
 """

import torch
import os, types


rwkv_config_2 = types.SimpleNamespace()

rwkv_config_2.model_name = "rwkv_demo"

w = torch.load(rwkv_config_2.model_name + ".pth", map_location='cpu')

for i in w.keys():
    print(i)



