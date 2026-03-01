"""
硬件感知部署工具
支持INT4量化模型的硬件部署优化
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path


class HardwareProfiler:
    """
    硬件性能分析器
    模拟不同硬件平台的性能特征
    """
    
    def __init__(self, platform: str = 'mobile_npu'):
        """
        Args:
            platform: 硬件平台 ('mobile_npu', 'edge_gpu', 'cpu')
        """
        self.platform = platform
        
        # 不同平台的性能特征（相对性能）
        self.platform_specs = {
            'mobile_npu': {
                'int4_multiply': 1.0,  # INT4乘法相对性能
                'int8_multiply': 0.5,  # INT8相对INT4的性能
                'fp32_multiply': 0.1,  # FP32相对INT4的性能
                'memory_bandwidth': 1.0,  # 内存带宽
                'cache_size': 0.5  # 缓存大小
            },
            'edge_gpu': {
                'int4_multiply': 0.8,
                'int8_multiply': 1.0,
                'fp32_multiply': 0.3,
                'memory_bandwidth': 1.2,
                'cache_size': 1.0
            },
            'cpu': {
                'int4_multiply': 0.3,
                'int8_multiply': 0.5,
                'fp32_multiply': 1.0,
                'memory_bandwidth': 0.8,
                'cache_size': 0.3
            }
        }
        
        self.specs = self.platform_specs[platform]
    
    def estimate_latency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        估算模型在不同精度下的延迟
        
        Args:
            model: 模型
            input_shape: 输入形状
            batch_size: 批次大小
            
        Returns:
            延迟估算字典
        """
        # 计算模型复杂度
        total_params = sum(p.numel() for p in model.parameters())
        
        # 估算FLOPs（简化版）
        # 实际应用中可以使用torchprofile等工具
        dummy_input = torch.randn(batch_size, *input_shape)
        
        # INT4延迟估算
        int4_latency = self._estimate_int4_latency(model, dummy_input)
        
        # INT8延迟估算
        int8_latency = int4_latency * (self.specs['int8_multiply'] / self.specs['int4_multiply'])
        
        # FP32延迟估算
        fp32_latency = int4_latency * (self.specs['fp32_multiply'] / self.specs['int4_multiply'])
        
        return {
            'int4_latency_ms': int4_latency * 1000,
            'int8_latency_ms': int8_latency * 1000,
            'fp32_latency_ms': fp32_latency * 1000,
            'total_params': total_params,
            'model_size_int4_mb': total_params * 4 / 8 / 1024 / 1024,
            'model_size_int8_mb': total_params * 8 / 8 / 1024 / 1024,
            'model_size_fp32_mb': total_params * 32 / 8 / 1024 / 1024
        }
    
    def _estimate_int4_latency(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor
    ) -> float:
        """
        估算INT4延迟（简化版）
        
        Args:
            model: 模型
            dummy_input: 虚拟输入
            
        Returns:
            估算延迟（秒）
        """
        # 计算总操作数（简化估算）
        total_ops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # 卷积操作数估算
                in_channels = module.in_channels
                out_channels = module.out_channels
                kernel_size = module.kernel_size[0] * module.kernel_size[1]
                # 简化：假设特征图大小为输入的一半
                feature_size = dummy_input.shape[2] * dummy_input.shape[3] / 4
                ops = in_channels * out_channels * kernel_size * feature_size
                total_ops += ops
            
            elif isinstance(module, nn.Linear):
                # 线性层操作数
                ops = module.in_features * module.out_features
                total_ops += ops
        
        # 根据平台性能估算延迟
        # 假设INT4操作吞吐量为平台相关值
        ops_per_second = 1e9 * self.specs['int4_multiply']  # 简化假设
        
        latency = total_ops / ops_per_second
        return latency


class ModelConverter:
    """
    模型转换器
    将量化模型转换为硬件友好的格式
    """
    
    def __init__(self, target_format: str = 'onnx'):
        """
        Args:
            target_format: 目标格式 ('onnx', 'tflite', 'ncnn')
        """
        self.target_format = target_format
    
    def convert_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        opset_version: int = 13
    ):
        """
        转换为ONNX格式
        
        Args:
            model: 模型
            input_shape: 输入形状
            output_path: 输出路径
            opset_version: ONNX opset版本
        """
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=opset_version,
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            do_constant_folding=True
        )
        
        print(f"模型已转换为ONNX格式: {output_path}")
    
    def export_quantization_info(
        self,
        model: nn.Module,
        output_path: str
    ):
        """
        导出量化信息（用于硬件部署）
        
        Args:
            model: 量化模型
            output_path: 输出路径
        """
        quantization_info = {}
        
        for name, module in model.named_modules():
            if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
                quantization_info[name] = {
                    'scale': module.scale.cpu().numpy().tolist(),
                    'zero_point': module.zero_point.cpu().numpy().tolist(),
                    'symmetric': getattr(module, 'symmetric', True),
                    'bits': getattr(module, 'bits', 4),
                    'qmin': getattr(module, 'qmin', -8),
                    'qmax': getattr(module, 'qmax', 7)
                }
        
        import json
        with open(output_path, 'w') as f:
            json.dump(quantization_info, f, indent=2)
        
        print(f"量化信息已导出到: {output_path}")


class DeploymentOptimizer:
    """
    部署优化器
    针对硬件平台优化模型部署
    """
    
    def __init__(self, platform: str = 'mobile_npu'):
        """
        Args:
            platform: 硬件平台
        """
        self.platform = platform
        self.profiler = HardwareProfiler(platform)
    
    def optimize_for_deployment(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        target_latency_ms: Optional[float] = None
    ) -> Dict[str, any]:
        """
        优化模型以适配部署
        
        Args:
            model: 模型
            input_shape: 输入形状
            target_latency_ms: 目标延迟（毫秒）
            
        Returns:
            优化结果字典
        """
        # 性能分析
        perf_metrics = self.profiler.estimate_latency(model, input_shape)
        
        # 检查是否满足目标延迟
        meets_target = True
        if target_latency_ms is not None:
            meets_target = perf_metrics['int4_latency_ms'] <= target_latency_ms
        
        # 优化建议
        suggestions = []
        
        if not meets_target and target_latency_ms is not None:
            suggestions.append(
                f"当前延迟 {perf_metrics['int4_latency_ms']:.2f}ms "
                f"超过目标 {target_latency_ms}ms，建议进一步优化"
            )
        
        if perf_metrics['model_size_int4_mb'] > 10:
            suggestions.append(
                f"模型大小 {perf_metrics['model_size_int4_mb']:.2f}MB 较大，"
                f"建议考虑剪枝或知识蒸馏"
            )
        
        return {
            'performance_metrics': perf_metrics,
            'meets_target': meets_target,
            'suggestions': suggestions,
            'platform': self.platform
        }
    
    def generate_deployment_report(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str
    ):
        """
        生成部署报告
        
        Args:
            model: 模型
            input_shape: 输入形状
            output_path: 输出路径
        """
        opt_result = self.optimize_for_deployment(model, input_shape)
        
        report = f"""
# 硬件感知部署报告

## 平台信息
- 目标平台: {self.platform}
- 量化精度: INT4

## 性能指标
- INT4延迟: {opt_result['performance_metrics']['int4_latency_ms']:.2f} ms
- INT8延迟: {opt_result['performance_metrics']['int8_latency_ms']:.2f} ms
- FP32延迟: {opt_result['performance_metrics']['fp32_latency_ms']:.2f} ms

## 模型大小
- INT4模型大小: {opt_result['performance_metrics']['model_size_int4_mb']:.2f} MB
- INT8模型大小: {opt_result['performance_metrics']['model_size_int8_mb']:.2f} MB
- FP32模型大小: {opt_result['performance_metrics']['model_size_fp32_mb']:.2f} MB

## 参数量
- 总参数量: {opt_result['performance_metrics']['total_params']:,}

## 优化建议
"""
        for suggestion in opt_result['suggestions']:
            report += f"- {suggestion}\n"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"部署报告已生成: {output_path}")
