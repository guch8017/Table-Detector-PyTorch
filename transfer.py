"""
用于将keras转换成torch，实际运行不使用
"""

import torch
import numpy as np
from collections import OrderedDict

bn_idx = 0
conv_idx = 0


def main():
    data = np.load(r'table-line.npz', allow_pickle=True)
    result = OrderedDict()

    def load_cnn(has_bias=False):
        global conv_idx
        if conv_idx == 0:
            name = f'/conv2d/conv2d/'
        else:
            name = f'/conv2d_{conv_idx}/conv2d_{conv_idx}/'
        conv_idx += 1
        weight = torch.from_numpy(data[name + 'kernel:0']).permute(3, 2, 0, 1)
        bias = torch.from_numpy(data[name + 'bias:0']) if has_bias else None
        return weight, bias

    def load_bn():
        global bn_idx
        if bn_idx == 0:
            name = f'/batch_normalization/batch_normalization/'
        else:
            name = f'/batch_normalization_{bn_idx}/batch_normalization_{bn_idx}/'
        bn_idx += 1
        gamma = torch.from_numpy(data[name + 'gamma:0'])
        beta = torch.from_numpy(data[name + 'beta:0'])
        mean = torch.from_numpy(data[name + 'moving_mean:0'])
        var = torch.from_numpy(data[name + 'moving_variance:0'])
        return gamma, beta, mean, var

    # Downsample
    for ds_layer in [256, 128, 64, 32, 16, 8, '_center']:
        for idx, l in [(0, 'c'), (1, 'n'), (3, 'c'), (4, 'n')]:
            if l == 'c':
                weight, bias = load_cnn()
                result[f'downsample{ds_layer}.layer.{idx}.weight'] = weight
            elif l == 'n':
                gamma, beta, mean, var = load_bn()
                result[f'downsample{ds_layer}.layer.{idx}.weight'] = gamma
                result[f'downsample{ds_layer}.layer.{idx}.bias'] = beta
                result[f'downsample{ds_layer}.layer.{idx}.running_mean'] = mean
                result[f'downsample{ds_layer}.layer.{idx}.running_var'] = var
                result[f'downsample{ds_layer}.layer.{idx}.num_batches_tracked'] = torch.tensor(0, dtype=torch.int64)

    # Upsample
    for us_layer in [16, 32, 64, 128, 256, 512]:
        for idx, l in [(0, 'c'), (1, 'n'), (3, 'c'), (4, 'n'), (6, 'c'), (7, 'n')]:
            if l == 'c':
                weight, bias = load_cnn()
                result[f'upsample{us_layer}.layer.{idx}.weight'] = weight
            elif l == 'n':
                gamma, beta, mean, var = load_bn()
                result[f'upsample{us_layer}.layer.{idx}.weight'] = gamma
                result[f'upsample{us_layer}.layer.{idx}.bias'] = beta
                result[f'upsample{us_layer}.layer.{idx}.running_mean'] = mean
                result[f'upsample{us_layer}.layer.{idx}.running_var'] = var
                result[f'upsample{us_layer}.layer.{idx}.num_batches_tracked'] = torch.tensor(0, dtype=torch.int64)

    # Classifier
    weight, bias = load_cnn(has_bias=True)
    result['classifier.weight'] = weight
    result['classifier.bias'] = bias
    torch.save(result, 'table-line.pth')


def test_model():
    from .model import LineDetector
    model = LineDetector(2)
    model.load_state_dict(torch.load('table-line.pth'))
    model.eval()
    input_img = torch.from_numpy(np.load(r'input.npy') / 255.0)
    input_img = input_img.permute(2, 0, 1).unsqueeze(0).float()
    output_pred = torch.from_numpy(np.load(r'output.npy'))
    model_out = model(input_img).permute(0, 2, 3, 1)
    print(model_out.shape)


test_model()