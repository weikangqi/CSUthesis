"""
INT4量化器实现
支持硬件感知部署与量化压缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class INT4Quantizer(nn.Module):
    """
    INT4量化器
    支持对称量化和非对称量化
    """
    
    def __init__(
        self,
        bits: int = 4,
        symmetric: bool = True,
        per_channel: bool = False,
        channel_axis: int = 0,
        calibration_method: str = 'minmax'
    ):
        """
        Args:
            bits: 量化位数，默认4
            symmetric: 是否使用对称量化（True: [-8, 7], False: [0, 15]）
            per_channel: 是否逐通道量化
            channel_axis: 通道轴索引
            calibration_method: 校准方法 ('minmax', 'percentile', 'mse')
        """
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.calibration_method = calibration_method
        
        # INT4范围
        if symmetric:
            self.qmin = -8  # -2^(bits-1)
            self.qmax = 7   # 2^(bits-1) - 1
        else:
            self.qmin = 0
            self.qmax = 15  # 2^bits - 1
        
        # 可学习的缩放因子和零点
        self.register_buffer('scale', None)
        self.register_buffer('zero_point', None)
        
        # 校准统计量
        self.register_buffer('min_val', None)
        self.register_buffer('max_val', None)
        self.register_buffer('calibrated', torch.tensor(False))
    
    def _compute_scale_and_zero_point(
        self,
        min_val: torch.Tensor,
        max_val: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算缩放因子和零点
        
        Args:
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            scale: 缩放因子
            zero_point: 零点
        """
        # 确保范围有效
        min_val = torch.clamp(min_val, min=-1e10, max=1e10)
        max_val = torch.clamp(max_val, min=-1e10, max=1e10)
        
        # 处理零范围情况
        range_val = max_val - min_val
        range_val = torch.clamp(range_val, min=1e-10)
        
        if self.symmetric:
            # 对称量化：零点固定为0
            scale = torch.max(torch.abs(min_val), torch.abs(max_val)) / (self.qmax - self.qmin + 1)
            scale = torch.clamp(scale, min=1e-10)
            zero_point = torch.zeros_like(scale)
        else:
            # 非对称量化
            scale = range_val / (self.qmax - self.qmin)
            scale = torch.clamp(scale, min=1e-10)
            zero_point = self.qmin - min_val / scale
            zero_point = torch.clamp(zero_point, self.qmin, self.qmax)
            zero_point = torch.round(zero_point)
        
        return scale, zero_point
    
    def calibrate(self, x: torch.Tensor, percentile: float = 99.99):
        """
        校准：收集统计量并计算量化参数
        
        Args:
            x: 输入张量
            percentile: 百分位数（用于percentile校准方法）
        """
        if self.calibration_method == 'minmax':
            if self.per_channel:
                # 逐通道统计
                dims = [i for i in range(x.ndim) if i != self.channel_axis]
                min_val = x.amin(dim=dims, keepdim=True)
                max_val = x.amax(dim=dims, keepdim=True)
            else:
                min_val = x.min()
                max_val = x.max()
        
        elif self.calibration_method == 'percentile':
            if self.per_channel:
                dims = [i for i in range(x.ndim) if i != self.channel_axis]
                min_val = torch.quantile(x, (100 - percentile) / 100, dim=dims, keepdim=True)
                max_val = torch.quantile(x, percentile / 100, dim=dims, keepdim=True)
            else:
                min_val = torch.quantile(x, (100 - percentile) / 100)
                max_val = torch.quantile(x, percentile / 100)
        
        elif self.calibration_method == 'mse':
            # MSE校准：搜索最优范围
            if self.per_channel:
                dims = [i for i in range(x.ndim) if i != self.channel_axis]
                min_val = x.amin(dim=dims, keepdim=True)
                max_val = x.amax(dim=dims, keepdim=True)
            else:
                min_val = x.min()
                max_val = x.max()
            
            # 简化版MSE：使用minmax，实际应用中可以使用更复杂的搜索
            pass
        
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")
        
        # 更新统计量
        if self.min_val is None:
            self.min_val = min_val
            self.max_val = max_val
        else:
            self.min_val = torch.minimum(self.min_val, min_val)
            self.max_val = torch.maximum(self.max_val, max_val)
    
    def update_quantization_params(self):
        """根据校准统计量更新量化参数"""
        if self.min_val is None or self.max_val is None:
            return
        
        scale, zero_point = self._compute_scale_and_zero_point(
            self.min_val, self.max_val
        )
        
        self.scale = scale
        self.zero_point = zero_point
        self.calibrated = torch.tensor(True)
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        量化操作
        
        Args:
            x: 输入浮点张量
            
        Returns:
            量化后的整数张量
        """
        if self.scale is None or self.zero_point is None:
            raise RuntimeError("Quantization parameters not initialized. Please calibrate first.")
        
        # 量化：q = round(x / scale + zero_point)
        x_scaled = x / self.scale + self.zero_point
        
        # 四舍五入并截断到量化范围
        x_quantized = torch.round(x_scaled)
        x_quantized = torch.clamp(x_quantized, self.qmin, self.qmax)
        
        return x_quantized.to(torch.int8)  # 使用int8存储，但值范围是int4
    
    def dequantize(self, x_quantized: torch.Tensor) -> torch.Tensor:
        """
        反量化操作
        
        Args:
            x_quantized: 量化后的整数张量
            
        Returns:
            反量化后的浮点张量
        """
        if self.scale is None or self.zero_point is None:
            raise RuntimeError("Quantization parameters not initialized.")
        
        # 反量化：x = (q - zero_point) * scale
        x_dequantized = (x_quantized.float() - self.zero_point) * self.scale
        
        return x_dequantized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：量化-反量化（用于量化感知训练）
        
        Args:
            x: 输入浮点张量
            
        Returns:
            量化-反量化后的浮点张量（用于反向传播）
        """
        if not self.calibrated:
            # 训练模式下，更新统计量
            if self.training:
                self.calibrate(x)
                self.update_quantization_params()
            else:
                # 推理模式下，如果未校准则使用当前值的范围
                self.calibrate(x)
                self.update_quantization_params()
        
        # 量化-反量化
        x_quantized = self.quantize(x)
        x_dequantized = self.dequantize(x_quantized)
        
        return x_dequantized


class INT4Linear(nn.Module):
    """
    INT4量化线性层
    支持量化感知训练和推理
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        symmetric: bool = True,
        per_channel_weight: bool = True,
        per_channel_activation: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 浮点权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 权重量化器（逐通道）
        self.weight_quantizer = INT4Quantizer(
            bits=4,
            symmetric=symmetric,
            per_channel=per_channel_weight,
            channel_axis=0
        )
        
        # 激活量化器（逐层或逐通道）
        self.activation_quantizer = INT4Quantizer(
            bits=4,
            symmetric=symmetric,
            per_channel=per_channel_activation,
            channel_axis=0
        )
        
        self.quantize_activation = False  # 是否量化激活（用于QAT）
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [..., in_features]
            
        Returns:
            输出张量 [..., out_features]
        """
        # 量化激活（如果启用）
        if self.quantize_activation:
            x = self.activation_quantizer(x)
        
        # 量化权重
        weight_quantized = self.weight_quantizer(self.weight)
        
        # INT4矩阵乘法（实际使用浮点模拟，真实部署时使用INT4算子）
        # 这里使用反量化后的权重进行模拟
        weight_dequantized = self.weight_quantizer.dequantize(weight_quantized)
        
        output = F.linear(x, weight_dequantized, self.bias)
        
        return output
    
    def enable_activation_quantization(self):
        """启用激活量化（用于QAT）"""
        self.quantize_activation = True
    
    def disable_activation_quantization(self):
        """禁用激活量化"""
        self.quantize_activation = False


class INT4Conv2d(nn.Module):
    """
    INT4量化卷积层
    支持量化感知训练和推理
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        symmetric: bool = True,
        per_channel_weight: bool = True,
        per_channel_activation: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # 浮点权重
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 权重量化器（逐通道）
        self.weight_quantizer = INT4Quantizer(
            bits=4,
            symmetric=symmetric,
            per_channel=per_channel_weight,
            channel_axis=0
        )
        
        # 激活量化器
        self.activation_quantizer = INT4Quantizer(
            bits=4,
            symmetric=symmetric,
            per_channel=per_channel_activation,
            channel_axis=1  # 通道维度
        )
        
        self.quantize_activation = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            输出张量 [B, C_out, H_out, W_out]
        """
        # 量化激活
        if self.quantize_activation:
            x = self.activation_quantizer(x)
        
        # 量化权重
        weight_quantized = self.weight_quantizer(self.weight)
        weight_dequantized = self.weight_quantizer.dequantize(weight_quantized)
        
        # INT4卷积（使用反量化后的权重模拟）
        output = F.conv2d(
            x, weight_dequantized, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )
        
        return output
    
    def enable_activation_quantization(self):
        """启用激活量化"""
        self.quantize_activation = True
    
    def disable_activation_quantization(self):
        """禁用激活量化"""
        self.quantize_activation = False
