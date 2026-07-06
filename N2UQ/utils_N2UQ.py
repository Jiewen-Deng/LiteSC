"""
This is the utils for N2UQ.
@Author: JW Deng
@Time: 2026/7/6 17:00
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from N2UQ.utils_quantization import RangeTracker, AveragedRangeTracker, GlobalRangeTracker, AsymmetricQuantizer
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

## N2UQ quantizers

import torch
import torch.nn as nn

class N2UQ_Symmetric(nn.Module):
    "N2UQ the activation"
    def __init__(self, bits, range_tracker):
        super(N2UQ_Symmetric, self).__init__()
        init_range = 2.0
        self.n_val = 2 ** bits - 1
        self.interval = init_range / self.n_val
        self.start = nn.Parameter(torch.Tensor([-1.0]), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([self.interval] * self.n_val), requires_grad=True)

        self.scale1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.two = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.zero = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-3]), requires_grad=False)
        self.range_tracker = range_tracker

    def update_params(self, input):
        self.range_tracker(input)
        quantized_range = self.one
        float_range = torch.max(torch.abs(self.range_tracker.min_val),
                                torch.abs(self.range_tracker.max_val))
        scale_value = quantized_range / float_range
        self.scale1.copy_(scale_value.squeeze())

    def forward(self, x):
        self.update_params(x)
        if self.training:
            return self._forward_cycle(x)
        else:
            return self._forward_vectorized(x)

    def _forward_cycle(self, x):
        """forward with cycle"""
        x_scaled = x * self.scale1
        x_forward = x_scaled
        x_backward = x_scaled
        step_right = self.minusone + 0.0

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)

        for i in range(self.n_val):
            step_right += self.interval
            if i == 0:
                thre_forward = self.start + a_pos[0] / 2
                thre_backward = self.start + 0.0
                x_forward = torch.where(x_scaled > thre_forward, step_right, self.minusone)
                x_backward = torch.where(x_scaled > thre_backward,
                                         self.interval / a_pos[i] * (x_scaled - thre_backward) + step_right - self.interval,
                                         self.minusone)
            else:
                thre_forward += a_pos[i-1] / 2 + a_pos[i] / 2
                thre_backward += a_pos[i-1]
                x_forward = torch.where(x_scaled > thre_forward, step_right, x_forward)
                x_backward = torch.where(x_scaled > thre_backward,
                                         self.interval / a_pos[i] * (x_scaled - thre_backward) + step_right - self.interval,
                                         x_backward)

        thre_backward += a_pos[i]
        x_backward = torch.where(x_scaled > thre_backward, self.one, x_backward)

        out = x_forward.detach() + x_backward - x_backward.detach()
        out = out / self.scale1
        return out

    def _forward_vectorized(self, x):
        """forward with searchsorted"""
        x_scaled = x * self.scale1

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)

        cumsum_a = torch.cumsum(a_pos, dim=0)
        half_intervals = a_pos / 2.0
        thresholds_forward = self.start + cumsum_a - half_intervals
        thresholds_backward = self.start + cumsum_a

        indices = torch.searchsorted(thresholds_forward.contiguous(), x_scaled.contiguous())
        indices = torch.clamp(indices, 0, self.n_val - 1)

        quantized_values = self.minusone + indices.float() * self.interval

        interval_at_index = a_pos[indices]
        threshold_at_index = thresholds_backward[indices]
        x_backward_scaled = self.interval / interval_at_index * (x_scaled - threshold_at_index) + \
                            self.minusone + indices.float() * self.interval

        out = quantized_values.detach() + x_backward_scaled - x_backward_scaled.detach()
        out = out / self.scale1
        return out

class N2UQ_Symmetric_constellation(nn.Module):
    """N2UQ constellation"""
    def __init__(self, bits, range_tracker):
        super(N2UQ_Symmetric_constellation, self).__init__()
        init_range = 2.0
        self.n_val = 2 ** bits - 1
        self.interval = init_range / self.n_val
        self.start = nn.Parameter(torch.Tensor([-1.0]), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([self.interval] * self.n_val), requires_grad=True)
        self.scale1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.two = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.zero = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-3]), requires_grad=False)
        self.range_tracker = range_tracker

    def update_params(self, input):
        self.range_tracker(input)
        quantized_range = self.one
        float_range = torch.max(torch.abs(self.range_tracker.min_val),
                                torch.abs(self.range_tracker.max_val))
        scale_value = quantized_range / float_range
        self.scale1 = nn.Parameter(scale_value, requires_grad=True)

    def forward(self, x):
        # self.update_params(x)
        if self.training:
            return self._forward_cycle(x)
        else:
            return self._forward_vectorized(x)

    def _forward_cycle(self, x):
        x_scaled = x * self.scale1
        x_forward = x_scaled
        x_backward = x_scaled
        step_right = self.minusone + 0.0

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)

        for i in range(self.n_val):
            step_right += self.interval
            if i == 0:
                thre_forward = self.start + a_pos[0] / 2
                thre_backward = self.start + 0.0
                x_forward = torch.where(x_scaled > thre_forward, step_right, self.minusone)
                x_backward = torch.where(x_scaled > thre_backward,
                                         self.interval / a_pos[i] * (x_scaled - thre_backward) + step_right - self.interval,
                                         self.minusone)
            else:
                thre_forward += a_pos[i-1] / 2 + a_pos[i] / 2
                thre_backward += a_pos[i-1]
                x_forward = torch.where(x_scaled > thre_forward, step_right, x_forward)
                x_backward = torch.where(x_scaled > thre_backward,
                                         self.interval / a_pos[i] * (x_scaled - thre_backward) + step_right - self.interval,
                                         x_backward)

        thre_backward += a_pos[i]
        x_backward = torch.where(x_scaled > thre_backward, self.one, x_backward)

        out = x_forward.detach() + x_backward - x_backward.detach()
        out = out / self.scale1
        return out

    def _forward_vectorized(self, x):
        """向量化版本（同普通版）"""
        x_scaled = x * self.scale1

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)

        cumsum_a = torch.cumsum(a_pos, dim=0)
        half_intervals = a_pos / 2.0
        thresholds_forward = self.start + cumsum_a - half_intervals
        thresholds_backward = self.start + cumsum_a

        indices = torch.searchsorted(thresholds_forward.contiguous(), x_scaled.contiguous())
        indices = torch.clamp(indices, 0, self.n_val - 1)

        quantized_values = self.minusone + indices.float() * self.interval

        interval_at_index = a_pos[indices]
        threshold_at_index = thresholds_backward[indices]
        x_backward_scaled = self.interval / interval_at_index * (x_scaled - threshold_at_index) + \
                            self.minusone + indices.float() * self.interval

        out = quantized_values.detach() + x_backward_scaled - x_backward_scaled.detach()
        out = out / self.scale1
        return out

class N2UQ_Asymmetric(nn.Module):
    def __init__(self, bits, range_tracker):
        super(N2UQ_Asymmetric, self).__init__()
        init_range = 2.0
        self.n_val = 2 ** bits - 1
        self.interval = init_range / (self.n_val + 1)
        self.start = nn.Parameter(torch.Tensor([-1.0]), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        # self.scale2 = nn.Parameter(torch.Tensor([2.0]), requires_grad=True)

        self.scale1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)# False, update by update_params()
        self.two =nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one =nn.Parameter(torch.Tensor([1.0 - self.interval]), requires_grad=False)
        self.zero =nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-3]), requires_grad=False)
        self.range_tracker = range_tracker

    def update_params(self, input):
        """
        scale = 2^(bits - 1)/(max(x_float) - min(x_float))
        zero_point = min(x_float)*scale
        """
        self.range_tracker(input)
        quantized_range = self.one
        float_range = torch.max(torch.abs(self.range_tracker.min_val),
                                torch.abs(self.range_tracker.max_val))
        scale_value = quantized_range / float_range
        self.scale1 = nn.Parameter(scale_value, requires_grad=False)

    def forward(self, x):
        self.update_params(x)# make scale1 to range [-1,1]
        x = x * self.scale1

        x_forward = x
        x_backward = x
        step_right = self.minusone + 0.0

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)

        for i in range(self.n_val):
            step_right += self.interval
            if i == 0:
                thre_forward = self.start + a_pos[0] / 2
                # print('thre_forward',thre_forward)
                thre_backward = self.start + 0.0
                x_forward = torch.where(x > thre_forward, step_right, self.minusone)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, self.minusone)
            else:
                thre_forward += a_pos[i-1] / 2 +  a_pos[i] / 2
                # print('thre_forward', thre_forward)
                thre_backward += a_pos[i-1]
                x_forward = torch.where(x > thre_forward, step_right, x_forward)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, x_backward)

        thre_backward += a_pos[i]
        x_backward = torch.where(x > thre_backward, self.one, x_backward)

        out = x_forward.detach() + x_backward - x_backward.detach()
        out = out / self.scale1# Ensure that the values before and after quantification fall within the same range.
        # out = out * self.scale2

        return out

class N2UQ_Asymmetric_constellation(nn.Module):
    def __init__(self, bits, range_tracker):
        super(N2UQ_Asymmetric_constellation, self).__init__()
        init_range = 2.0
        self.n_val = 2 ** bits - 1
        self.interval = init_range / (self.n_val + 1)
        self.start = nn.Parameter(torch.Tensor([-1.0]), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.scale1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        # self.scale2 = nn.Parameter(torch.Tensor([sqrt(3.0)]), requires_grad=True)

        self.two =nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one =nn.Parameter(torch.Tensor([1.0 - self.interval]), requires_grad=False)
        self.zero =nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-3]), requires_grad=False)
        self.range_tracker = range_tracker

    def update_params(self, input):
        """
        scale = 2^(bits - 1)/(max(x_float) - min(x_float))
        zero_point = min(x_float)*scale
        """
        self.range_tracker(input)
        quantized_range = self.one
        float_range = torch.max(torch.abs(self.range_tracker.min_val),
                                torch.abs(self.range_tracker.max_val))
        scale_value = quantized_range / float_range
        self.scale1 = nn.Parameter(scale_value, requires_grad=True)

    def forward(self, x):

        x = x * self.scale1

        x_forward = x
        x_backward = x
        step_right = self.minusone + 0.0

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)

        for i in range(self.n_val):
            step_right += self.interval
            if i == 0:
                thre_forward = self.start + a_pos[0] / 2
                # print('thre_forward',thre_forward)
                thre_backward = self.start + 0.0
                x_forward = torch.where(x > thre_forward, step_right, self.minusone)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, self.minusone)
            else:
                thre_forward += a_pos[i-1] / 2 +  a_pos[i] / 2
                # print('thre_forward', thre_forward)
                thre_backward += a_pos[i-1]
                x_forward = torch.where(x > thre_forward, step_right, x_forward)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, x_backward)

        thre_backward += a_pos[i]
        x_backward = torch.where(x > thre_backward, self.one, x_backward)

        out = x_forward.detach() + x_backward - x_backward.detach()
        out = out / self.scale1
        # out = out * self.scale2

        return out

## N2UQ Linear layers
class QuantizedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, a_bits=2, w_bits=2):
        super().__init__(
            in_features = in_features,
            out_features = out_features,
            bias = bias)
        # self.weight_quantizer = N2UQ_Asymmetric(bits=w_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=AveragedRangeTracker(q_level='L'))
        self.activation_quantizer = N2UQ_Symmetric(bits=a_bits, range_tracker=AveragedRangeTracker(q_level='L'))

    def forward(self, input):
        q_input = self.activation_quantizer(input)
        q_weight = self.weight_quantizer(self.weight)
        output = F.linear(input=q_input, weight=q_weight, bias=self.bias)

        return output

class QuantizedLinear_cons(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, a_bits=2, w_bits=2):
        super().__init__(
            in_features = in_features,
            out_features = out_features,
            bias = bias)

        self.weight_quantizer = AsymmetricQuantizer(bits=w_bits, range_tracker=AveragedRangeTracker(q_level='L'))

    def forward(self, input):
        q_weight = self.weight_quantizer(self.weight)
        output = F.linear(input=input, weight=q_weight, bias=self.bias)
        return output
