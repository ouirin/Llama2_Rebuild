import gc
import math
import torch
from torch import nn
from datasets import load_dataset
from quant_tool_gptq.quant_linear import QuantLinear


def quant_model(model, tokenizer, linear_bit=4, linear_group=32):

    data = load_calibration(tokenizer, calibration_file="ppl_dataset", n_sample=1, sample_length=512)

    qlayers = bind_quantizer(model, linear_bit=linear_bit, linear_group=linear_group)

    with torch.no_grad():
        for i in range(len(data)):
            model(data[i], all_kv_cache=None)
            gc.collect()

    replace_linear(model, qlayers)

    return model


class GptqQuantizer:

    def __init__(self, layer, path=None, bit=None, group_size=None):

        self.layer = layer
        self.path = path
        self.bit = bit
        self.group_size = group_size
        self.block_size = 128
        self.maxq = torch.tensor(2 ** (bit-1) - 1).to("cuda")
        self.n_rows, self.n_columns = layer.weight.shape

        self.hessian = torch.zeros((self.n_columns, self.n_columns), device='cpu', dtype=torch.float32)
        self.hook = layer.register_forward_hook(self.forward_hook)
        self.n_samples = 0

    @torch.no_grad()
    def forward_hook(self, module, inputs, outputs):

        _input, = inputs                                                        # unpack layer input
        _input = _input.flatten(0, -2)                                          # (batch_size*seq_n, dim)
        new_samples, d_hidden = _input.shape
        _input = _input.t()                                                     # (dim, batch_size*seq_n)

        self.hessian = self.hessian.to("cuda")
        _input = _input.to("cuda")

        self.hessian *= self.n_samples / (self.n_samples + new_samples)
        self.n_samples += new_samples
        _input = math.sqrt(2 / self.n_samples) * _input.to(self.hessian.dtype)
        self.hessian += _input.matmul(_input.t())                               # generate hessian matrix

        self.hessian = self.hessian.to("cpu")
        _input = _input.to("cpu")

    @torch.no_grad()
    def quantize(self):

        weight = self.layer.weight.float().to("cuda")
        hessian = self.hessian.to("cuda")
        quant_losses = torch.zeros_like(weight)
        grid_weight = torch.zeros_like(weight)

        dead_rows = torch.diag(hessian) == 0                            # (dim)
        hessian[dead_rows, dead_rows] = 1                               # (dim,dim), 位于dead_rows的值置为1，好像可以减少非正定矩阵的报错率
        weight[:, dead_rows] = 0                                        # (4608,dim), 位于dead_rows的权重置为0，也算一种剪枝

        damp = 0.01 * torch.mean(torch.diag(hessian))                   # 0.01=damp阻尼值，用于数值稳定性的常数，使用对角线均值可以维持海森矩阵的特性
        diag = torch.arange(self.n_columns, device=weight.device)
        hessian[diag, diag] += damp                                     # 将阻尼值添加到 hessian 矩阵的对角线上

        hessian_inv = torch.linalg.cholesky(hessian)
        hessian_inv = torch.cholesky_inverse(hessian_inv)
        hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
        assert not hessian_inv.isnan().any()

        scale_list = []

        for i1 in range(0, self.n_columns, self.block_size):

            i2 = min(i1 + self.block_size, self.n_columns)               # weight是按照block_size进行更新的

            weight_block = weight[:, i1:i2].clone()                      # 取一个block_size的权重, 这里必须clone()
            h_inv_block = hessian_inv[i1:i2, i1:i2]                      # 按顺序取海森矩阵，这样海森矩阵就不用更新
            err_block = torch.zeros_like(weight_block)
            losses_block = torch.zeros_like(weight_block)

            for i in range(i2 - i1):

                w = weight_block[:, i].unsqueeze(1)
                d = h_inv_block[i, i]

                if (i1 + i) % self.group_size == 0:
                    w_max, _ = torch.max(weight[:, (i1 + i):(i1 + i + self.group_size)].abs(), dim=1, keepdim=True)
                    scale = torch.clamp(w_max / self.maxq, min=1e-10)
                    scale_list.append(scale)
                q_weight = torch.clamp(torch.round(w / scale), -self.maxq, self.maxq)
                weight[:, i1 + i] = q_weight.squeeze()

                de_quant = q_weight * scale
                grid_weight[:, i1 + i] = de_quant.squeeze()
                losses_block[:, i] = ((w - de_quant) ** 2 / d ** 2).squeeze()
                err = (w - de_quant) / d
                weight_block[:, i:] -= err.matmul(h_inv_block[i, i:].unsqueeze(0))    # 局部权重更新
                err_block[:, i] = err.squeeze()                                       # 存的是 该block整体的（量化前权重-量化后权重），用于全局权重更新

            quant_losses[:, i1:i2] = losses_block / 2
            weight[:, i2:] -= err_block.matmul(hessian_inv[i1:i2, i2:])               # 全局权重更新

        quant_losses = quant_losses.mean().item()                                     # 输出权重的差值
        print(quant_losses)

        q_weight = (weight + 8).to(torch.uint8)
        q_weight = (q_weight[:, ::2]) | ((q_weight[:, 1::2]) << 4)
        scale = torch.cat(scale_list, dim=1)

        return q_weight, scale

    @torch.no_grad()
    def get_quantized_linear(self):

        q_weight, scale = self.quantize()

        qlinear = QuantLinear(in_features=self.layer.in_features, out_features=self.layer.out_features, group_size=self.group_size)

        qlinear.apply_weights_(q_weight, scale, self.layer.bias)

        del self.layer
        gc.collect()

        return qlinear


def get_min_eigenvalue(tensor):

    tensor = tensor.float()
    eigenvalues, eigenvectors = torch.linalg.eig(tensor)
    eigenvalues = eigenvalues.real
    print(min(eigenvalues))

    return min(eigenvalues)


def get_hessian_inv(hessian, diag, weight_dtype, damp=0.01):

    success = False
    while not success:
        try:
            hessian_inv = torch.linalg.cholesky(hessian)  # Cholesky分解海森矩阵
            success = True
        except RuntimeError as error:
            print("Cholesky 分解失败")
            if weight_dtype == torch.float32:
                hessian[diag, diag] += damp
            if weight_dtype == torch.float16:
                hessian[diag, diag] += (damp * 20)

    hessian_inv = torch.cholesky_inverse(hessian_inv)     # Cholesky逆函数计算逆矩阵

    success = False
    while not success:
        try:
            # 如果这里失败，属于torch.clolesky_inverse的稳定性问题, 建议重跑
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
            success = True
        except RuntimeError as error:
            print("二阶段 Cholesky 分解失败，可能对量化精度产生不良影响")
            if weight_dtype == torch.float32:
                damp = abs(get_min_eigenvalue(hessian_inv)) + 1e-10
                hessian_inv[diag, diag] += damp
            if weight_dtype == torch.float16:
                damp = abs(get_min_eigenvalue(hessian_inv)) + 1e-4
                hessian_inv[diag, diag] += damp

    return hessian_inv


def bind_quantizer(model, linear_bit=None, linear_group=None):

    layers_with_path = find_layers(module=model, target=nn.Linear)

    qlayers = {}

    for path, layer in layers_with_path.items():

        qlayers[path] = GptqQuantizer(layer=layer, path=path, bit=linear_bit, group_size=linear_group)

    return qlayers


def replace_linear(model, qlayers):

    for path, layer in qlayers.items():

        cover_layer(model, path, layer.get_quantized_linear())

        print(f"{path} - finished")


def find_layers(module=None, target=None, name=''):

    if isinstance(module, target):

        return {name: module}

    result = {}

    for child_name, child in module.named_children():

        result.update(find_layers(module=child, target=target, name = name + '.' + child_name if name != '' else child_name))

    return result


def cover_layer(module=None, path=None, new_layer=None):

    parent = module
    path_list = path.split('.')
    path_list_len = len(path_list)

    counter = 1

    for child_name in path_list:

        if counter == path_list_len:

            setattr(parent, child_name, new_layer)

        else:

            parent = getattr(parent, child_name)
            counter += 1


def load_calibration(tokenizer, calibration_file, n_sample, sample_length=512):

    testdata = load_dataset(calibration_file, split='test')
    testdata = "\n\n".join(testdata['text'])

    input_ids = tokenizer.encode(testdata)
    input_ids = torch.LongTensor([input_ids])

    data = []

    num_sample = input_ids.numel() // sample_length

    for i in range(num_sample):

        if len(data) < n_sample:

            data.append(input_ids[:, (i * sample_length):((i + 1) * sample_length)])

    return data


# import os
# import pandas as pd
# choices = ["A", "B", "C", "D"]
#
# def format_example(df, idx, include_answer=True):
#
#     prompt = df.iloc[idx, 0]
#     for i in range(4):
#         prompt += "\n{}. {}".format(choices[i], df.iloc[idx, i+1])
#
#     prompt += "\nAnswer:"
#     if include_answer:
#         prompt += " {}\n\n".format(df.iloc[idx, 5])
#
#     return prompt
#
# def load_calibration2(tokenizer, calibration_file, n_sample):
#
#     dataset = []
#
#     for file in os.listdir('test/'):
#
#         df = pd.read_csv('test/' + file, header=None)
#
#         for i in range(df.shape[0]):
#
#             prompt = format_example(df, i, include_answer=False)
#
#             input_ids = tokenizer.encode(prompt)
#             input_ids = torch.LongTensor([input_ids])
#
#             if random.random()<0.01 and len(dataset) < n_sample:
#
#                 dataset.append(input_ids)
#
#     return dataset


