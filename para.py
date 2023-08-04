""" 
计算模型参数量
 """
import torch

w = torch.load("rwkv_demo" + ".pth", map_location='cpu')
keys = list(w.keys())

sum_p = 0
for i in keys:
    product=1
    a = [product := product * j for j in w[i].shape]
    sum_p += a[-1]
    print(f"神经网络层：{i}, 形状：{w[i].shape}, 参数量{a[-1]}")
print(sum_p)