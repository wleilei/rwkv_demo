""" 
测试文件
 """

import torch
import json
import os, types
from torch.nn import functional as F


rwkv_config_2 = types.SimpleNamespace()

rwkv_config_2.model_name = "rwkv_demo"
rwkv_config_2.RWKV_RESCALE_LAYER = 6 # set x=x/2 every X layer
rwkv_config_2.n_layer = 12
rwkv_config_2.RUN_DEVICE = 'cpu'
rwkv_config_2.n_embd = 512

# 定义模型
class RWKV_RUN(torch.jit.ScriptModule):
    def __init__(self, rwkv_config_2):
        super().__init__()
        self.rwkv_config_2 = rwkv_config_2

        with torch.no_grad():
            w = torch.load(rwkv_config_2.model_name + ".pth", map_location='cpu')
            keys = list(w.keys())
            self.RUN_DEVICE = rwkv_config_2.RUN_DEVICE

            # 更新权重
            for x in keys:
                block_id = 0
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                if 'time_mix.output.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // rwkv_config_2.RWKV_RESCALE_LAYER))
                if 'channel_mix.value.weight' in x:
                    w[x] = w[x] / (2 ** int(block_id // rwkv_config_2.RWKV_RESCALE_LAYER))
                if '.time_' in x:
                    w[x] = w[x].squeeze()
                if '.time_decay' in x:
                    w[x] = w[x].float()
                    w[x] = -torch.exp(w[x])
                elif '.time_first' in x:
                    w[x] = w[x].float()
                else:
                    w[x] = w[x].float()

                w[x].requires_grad = False
                print(w[x])
                if rwkv_config_2.RUN_DEVICE == 'cuda' and x != 'emb.weight':
                    w[x] = w[x].cuda()

        # 保存至多层嵌套命名空间结构
        keys = list(w.keys())
        self.w = types.SimpleNamespace()
        for x in keys:
            xx = x.split('.')
            here = self.w
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x])
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i])

        self.eval()
        torch.cuda.empty_cache()
    # 定义层归一化
    def LN(self, x, w):
        return F.layer_norm(x, (self.rwkv_config_2.n_embd,), weight=w.weight, bias=w.bias)
    
    # 定义Time_mix
    @torch.jit.script_method
    def TM(self, x, state, i:int, time_mix_k, time_mix_v, time_mix_r, time_first, time_decay, kw, vw, rw, ow):
        
        xk = x * time_mix_k + state[5*i+0] * (1 - time_mix_k)
        xv = x * time_mix_v + state[5*i+0] * (1 - time_mix_v)
        xr = x * time_mix_r + state[5*i+0] * (1 - time_mix_r)
        state[5*i+0] = x

        r = torch.sigmoid(rw @ xr)
        k = kw @ xk
        v = vw @ xv

        kk = k
        vv = v
        aa = state[5*i+1]
        bb = state[5*i+2]
        pp = state[5*i+3]
        ww = time_first + kk
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        a = e1 * aa + e2 * vv
        b = e1 * bb + e2
        ww = pp + time_decay
        p = torch.maximum(ww, kk)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(kk - p)
        state[5*i+1] = e1 * aa + e2 * vv
        state[5*i+2] = e1 * bb + e2
        state[5*i+3] = p
        wkv = a / b
        
        return ow @ (r * wkv)

    @torch.jit.script_method
    def CM(self, x, state, i:int, time_mix_k, time_mix_r, kw, vw, rw):
        
        xk = x * time_mix_k + state[5*i+4] * (1 - time_mix_k)
        xr = x * time_mix_r + state[5*i+4] * (1 - time_mix_r)
        state[5*i+4] = x

        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk))
        kv = vw @ k

        return r * kv   
    
    def forward(self,ctx, state, preprocess_only = False):
        with torch.no_grad():
            w = self.w
            args = self.rwkv_config_2

            x = w.emb.weight[ctx[-1]]
            if self.RUN_DEVICE == 'cuda':
                x = x.cuda()

            if state == None:
                state = torch.zeros(args.n_layer * 5, args.n_embd, device=self.RUN_DEVICE)
                for i in range(args.n_layer):
                    state[5*i+4] -= 1e30
            
            for i in range(args.n_layer):
                x = self.LN(x, w.ln_in)

                ww = w.blocks[i].Time_mix
                x = x + self.TM(self.LN(x, w.blocks[i].ln1), state, i,
                                ww.time_mix_k, ww.time_mix_v, ww.time_mix_r, ww.time_first, ww.time_decay, 
                                ww.key.weight, ww.value.weight, ww.receptance.weight, ww.output.weight)
                
                ww = w.blocks[i].Channel_mix
                x = x + self.CM(self.LN(x, w.blocks[i].ln2), state, i,
                                ww.time_mix_k, ww.time_mix_r,
                                ww.key.weight, ww.value.weight, ww.receptance.weight)
                if (i+1) % rwkv_config_2.RWKV_RESCALE_LAYER == 0:
                    x = x / 2
            
            if preprocess_only:
                return state

            x = self.LN(x, w.ln_out)
            x = w.linear.weight @ x

            return x.float(), state

# 定义tokenizer
class RWKV_tokenizer():
    def __init__(self, vocab_file):

        # 加载词汇表
        self.vocab_file = vocab_file
        self._tokenizer = self._load_custom_tokenizer(vocab_file)

    def _load_custom_tokenizer(self,vocab_file):
        with open(vocab_file, "r", encoding="utf-16") as f:
            vocab = json.load(f)

        return vocab

    def __call__(self, text):
        words = [i for i in text]
        input_ids = [self._tokenizer.get(word) for word in words]

        return input_ids

tokenizer = RWKV_tokenizer("vocab.json")

text = "simplicity is the keynote of all true elegance"

text_id = tokenizer(text)

print(text_id)
model = RWKV_RUN(rwkv_config_2)

with torch.no_grad():
    
    state = None
    out = None
    src_len = len(text_id)
    for i in range(src_len):
        x = text_id[:i+1]
        out, state = model.forward(x, state)
        if i < 3 or i >= src_len - 3:
            maxx,id = torch.max(out, -1, keepdim=True)
            print(id.detach().cpu().numpy())
            print(maxx.detach().cpu().numpy())
        if i == 2:
            print('...')










