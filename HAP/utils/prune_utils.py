"""
@Author: JW Deng
@Time: 2026/7/6 17:00
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from torch.nn.modules.utils import _pair

# ======================================================
# Find layer dependency
# Update input indices (adapt to previous layers)
# Update output indices
# ======================================================
def get_layer_dependencies(model, network):
    # Helper function; ad-hoc fix
    dependencies = OrderedDict()
    # Handling layer dependencies for Transformer networks
    # dependencies = OrderedDict()

    # Encoder part dependencies
    # Encoder embedding layer
    dependencies[model.encoder.embedding] = []
    dependencies[model.encoder.pos_encoding] = []

    # Encoder layers dependencies
    prev_modules = [model.encoder.embedding]
    for i, enc_layer in enumerate(model.encoder.enc_layers):
        # Multi-head attention depends on the output from previous layers
        dependencies[enc_layer.mha.wq] = prev_modules
        dependencies[enc_layer.mha.wk] = prev_modules
        dependencies[enc_layer.mha.wv] = prev_modules
        dependencies[enc_layer.mha.dense] = [enc_layer.mha.wv]  # depends on QKV transformation
        # dependencies[enc_layer.mha.dense] = [enc_layer.mha.wq]  # depends on QKV transformation

        # LayerNorm depends on attention output
        dependencies[enc_layer.layernorm1] = [enc_layer.mha.dense]

        # Feed forward depends on LayerNorm1 output
        dependencies[enc_layer.ffn.w_1] = [enc_layer.mha.dense]
        dependencies[enc_layer.ffn.w_2] = [enc_layer.ffn.w_1]

        # LayerNorm2 depends on FFN output
        dependencies[enc_layer.layernorm2] = [enc_layer.ffn.w_2]

        # Update prev_modules to current layer's final output
        prev_modules = [enc_layer.ffn.w_2]

    # Channel encoder depends on the last Encoder layer
    dependencies[model.channel_encoder[0]] = prev_modules  # Linear(d_model, 256)
    dependencies[model.channel_encoder[2]] = [model.channel_encoder[0]]  # Linear(256, 16)

    # Channel decoder depends on channel encoder
    dependencies[model.channel_decoder.linear1] = [model.channel_encoder[2]]  # Linear(16, d_model)
    dependencies[model.channel_decoder.linear2] = [
        model.channel_decoder.linear1]  # Linear(d_model, 512)
    dependencies[model.channel_decoder.linear3] = [
        model.channel_decoder.linear2]  # Linear(512, d_model)
    dependencies[model.channel_decoder.layernorm] = [model.channel_decoder.linear3]  # LayerNorm

    # Decoder part dependencies
    # Decoder embedding layer
    dependencies[model.decoder.embedding] = []
    dependencies[model.decoder.pos_encoding] = []

    # Decoder layers dependencies
    prev_modules = [model.decoder.embedding]
    for i, dec_layer in enumerate(model.decoder.dec_layers):
        # Self attention
        dependencies[dec_layer.self_mha.wq] = prev_modules
        dependencies[dec_layer.self_mha.wk] = prev_modules
        dependencies[dec_layer.self_mha.wv] = prev_modules
        dependencies[dec_layer.self_mha.dense] = [dec_layer.self_mha.wv]
        dependencies[dec_layer.layernorm1] = [dec_layer.self_mha.dense]

        # Source attention (depends on channel decoder output)
        channel_dec_modules = [model.channel_decoder.linear3]  # or determined based on actual connections
        dependencies[dec_layer.src_mha.wq] = [dec_layer.self_mha.dense]
        dependencies[dec_layer.src_mha.wk] = channel_dec_modules
        dependencies[dec_layer.src_mha.wv] = channel_dec_modules
        dependencies[dec_layer.src_mha.dense] = [dec_layer.src_mha.wv]
        dependencies[dec_layer.layernorm2] = [dec_layer.src_mha.dense]

        # Feed forward
        dependencies[dec_layer.ffn.w_1] = [dec_layer.src_mha.dense]
        dependencies[dec_layer.ffn.w_2] = [dec_layer.ffn.w_1]
        dependencies[dec_layer.layernorm3] = [dec_layer.ffn.w_2]

        prev_modules = [dec_layer.ffn.w_2]

    # Final dense layer depends on the last decoder layer
    dependencies[model.dense] = prev_modules

    return dependencies

def get_layer_dependencies_out(model, network):
    # Helper function; ad-hoc fix
    dependencies_out = OrderedDict()
    # Encoder layers dependencies
    for i, enc_layer in enumerate(model.encoder.enc_layers):
        # Multi-head attention depends on output from previous layers
        dependencies_out[enc_layer.mha.wq] = enc_layer.mha.wk
    dependencies_out[model.channel_decoder.linear1] = model.channel_decoder.linear3
    for i, dec_layer in enumerate(model.decoder.dec_layers):
        # Self attention
        dependencies_out[dec_layer.self_mha.wq] = dec_layer.self_mha.wk
        # Source attention (depends on channel decoder output)
        dependencies_out[dec_layer.src_mha.wq] = dec_layer.src_mha.wk

    return dependencies_out

def update_indices(model, network):
    print("updating indices")
    dependencies = get_layer_dependencies(model, network)
    # update_out_indices
    update_in_dinces(dependencies)
    # update_in_dinces_after_embedding(dependencies)


def update_out_indices(dependencies_out, row_imps):
    pass

def update_in_dinces(dependencies):
    for m, deps in dependencies.items():
        if len(deps) > 0:
            indices = set()
            for d in deps:
                # indices = indices.union(d.out_indices)
                # Process different types of layers
                if isinstance(d, nn.Embedding):
                    if hasattr(d, 'weight') and d.weight is not None:
                        indices = indices.union(range(d.weight.size(1)))
                    else:
                        continue
                elif hasattr(d, 'out_indices') and d.out_indices is not None:
                    # Normal convolutional or linear layers
                    indices = indices.union(d.out_indices)
                # elif isinstance(d, (nn.LayerNorm, nn.BatchNorm2d)):
                #     # Special handling for LayerNorm or BatchNorm layers
                #     # For LayerNorm, typically inherits from previous layer or remains complete
                #     if hasattr(d, 'in_indices') and d.in_indices is not None:
                #         indices = indices.union(d.in_indices)
                #     else:
                #         # If no predefined indices, may need to determine based on weight shape
                #         if hasattr(d, 'weight') and d.weight is not None:
                #             indices = indices.union(range(d.weight.size(0)))
                #         else:
                #             # fallback to previous layer's indices
                #             continue

                else:
                    # Other types of layers
                    continue
            m.in_indices = sorted(list(indices))

def update_in_dinces_after_embedding(dependencies):
    for m, deps in dependencies.items():
        if len(deps) > 0:
            indices = set()
            for d in deps:
                # indices = indices.union(d.out_indices)
                # Process different types of layers
                if isinstance(d, nn.Embedding):
                    # The output dimension of the Embedding layer corresponds to its vocabulary size (num_embeddings)
                    if hasattr(d, 'in_indices') and d.in_indices is not None:
                        indices = indices.union(d.in_indices)
                    else:
                        # If no predefined indices, may need to determine based on weight shape
                        if hasattr(d, 'weight') and d.weight is not None:
                            indices = indices.union(range(d.weight.size(0)))
                        else:
                            # fallback to previous layer's indices
                            continue
            m.in_indices = sorted(list(indices))


def get_threshold(values, percentage):
    v_sorted = sorted(values)
    n = int(len(values) * percentage)
    threshold = v_sorted[n]
    return threshold

def filter_indices(values, threshold):
    indices = []
    for idx, v in enumerate(values):
        if v > threshold:
            indices.append(idx)
    if len(indices) <= 1:
        # we make it at least 1 filters in each layer
        indices = [0]
    return indices

def filter_indices_qkv(values, threshold, num_heads=8):
    indices = []
    for idx, v in enumerate(values):
        if v > threshold:
            indices.append(idx)
    # ensure the number of indices is a multiple of num_heads
    sorted_indices = [idx for idx, _ in sorted(enumerate(values), key=lambda x: x[1], reverse=True)]
    if len(indices) < num_heads:
        # if less than num_heads, round up to num_heads
        indices = indices + [idx for idx in sorted_indices if idx not in indices][:num_heads - len(indices)]
    else:
        remainder = len(indices) % num_heads
        if remainder != 0:
            # remove the last 'remainder' elements to make it divisible by num_heads
            i = 0
            for idx in sorted_indices:
                if i >= num_heads - remainder:
                    break
                if idx in indices:
                    continue
                indices.append(idx)
                i += 1
            # indices = indices + [idx for idx in sorted_indices if idx not in indices][:num_heads - remainder]
    return sorted(indices)

def filter_indices_num(values, num_indices):
    indices = []
    sorted_indices = [idx for idx, _ in sorted(enumerate(values), key=lambda x: x[1], reverse=True)]
    for i,idx in enumerate(sorted_indices):
        if i >= num_indices:
            break
        indices.append(idx)
    return sorted(indices)

def filter_indices_ni(values, threshold, ni_threshold):
    ni_indices = []
    pruned_indices = []
    remained_indices = []
    for idx, v in enumerate(values):
        if v > threshold:
            remained_indices.append(idx)
        elif v > ni_threshold and v<= threshold:
            ni_indices.append(idx)
        else:
            pruned_indices.append(idx)
    if len(remained_indices) <= 1:
        # we make it at least 1 filters in each laer
        remained_indices = [0]
        try:
            ni_indices.remove(0)
        except Exception as e:
            pruned_indices.remove(0)
    return remained_indices, ni_indices, pruned_indices

class LinearLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(LinearLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)

        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1).fill_(self.bias)], 1)
        return x @ self.rotation_matrix

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return "in_features=%s, out_features=%s, trainable=%s" % (self.rotation_matrix.size(1),
                                                                  self.rotation_matrix.size(0),
                                                                  self.trainable)
class ConvLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(ConvLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3)  # out_dim * in_dim
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)
        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        # x: batch_size * in_dim * w * h
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1, x.size(2), x.size(3)).fill_(self.bias)], 1)
        return F.conv2d(x, self.rotation_matrix, None, _pair(1), _pair(0), _pair(1), 1)

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return "in_channels=%s, out_channels=%s, trainable=%s" % (self.rotation_matrix.size(1),
                                                                  self.rotation_matrix.size(0),
                                                                  self.trainable)
# ====== Neuron Implant ====== #
def prune_model_ni(model):
    # Recursively prune the model
    if type(model) == nn.Conv2d:
        # return DW_NIConv2d(model)
        # return NIConv2d_fast(model)
        return NIConv2d(model)

    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(prune_model_ni(m))
        return nn.Sequential(*mods)

    else:
        try:
            newmodel = copy.deepcopy(model)
        except Exception as e:
            print(model)
            print(e)
            exit()

        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(newmodel, attr, prune_model_ni(mod))
        return newmodel


# ======= NI ======
class NIConv2d(nn.Module):
    def __init__(self, conv):
        super(NIConv2d, self).__init__()
        self.out_channels       = conv.out_channels
        self.in_channels        = conv.in_channels
        self.kernel_size        = conv.kernel_size
        self.stride             = conv.stride
        self.padding            = conv.padding
        self.pruned             = 1
        self.remained_indices   = conv.remained_indices
        self.ni_indices         = conv.ni_indices
        self.pruned_indices     = conv.pruned_indices

        self.in_indices         = conv.in_indices
        self.out_indices        = conv.out_indices

        if self.in_indices == None:
            self.in_indices = list(range(conv.weight.size(1)))
        self.in_channels        = len(self.in_indices)


        middle = (self.kernel_size[0]-1) // 2

        self.conv_indices = sorted(self.remained_indices + self.ni_indices)
        print(len(self.remained_indices), len(self.ni_indices), len(self.pruned_indices))

        self.conv1 = nn.Conv2d(self.in_channels, len(self.remained_indices), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True if conv.bias is not None else False)
        self.conv1.weight.data = conv.weight.data[self.remained_indices, :, :, :][:, self.in_indices,: ,:].clone()

        # new indices points to the position of 1x1 indices in conv_indices
        self.ni_indices_new = []
        if len(self.ni_indices)>0:
            for idx in self.ni_indices:
                self.ni_indices_new.append(self.conv_indices.index(idx))

        # new indices points to the position of 3x3 indices in conv_indices
        self.remained_indices_new = []
        if len(self.remained_indices)>0:
            for idx in self.remained_indices:
                self.remained_indices_new.append(self.conv_indices.index(idx))


        if len(self.ni_indices)>0:
            self.conv2 = nn.Conv2d(self.in_channels, len(self.ni_indices), kernel_size=1, stride=self.stride, bias=False)
            self.conv2.weight.data = conv.weight.data[self.ni_indices, :, middle:middle+1, middle:middle+1][:, self.in_indices,: ,:].clone().fill_(0)

        self.out_channels = len(self.conv_indices)

        if conv.bias is not None:
            self.conv1.bias.data = conv.bias.data[m.out_indices]
            self.conv1.bias.grad = None

    def forward(self, x):
        out1 = self.conv1(x)
        out = out1[:,0:1,:,:].expand(out1.shape[0], self.out_channels, out1.shape[2], out1.shape[3]).clone().fill_(0)
        out[:,self.remained_indices_new,:,:] = out1
        if len(self.ni_indices)>0:
            out2 = self.conv2(x)
            out[:,self.ni_indices_new,:,:] = out2
        return out
