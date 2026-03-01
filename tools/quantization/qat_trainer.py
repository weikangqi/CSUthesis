"""
量化感知训练（QAT）工具
支持INT4量化感知训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Callable, Dict, Any
from tqdm import tqdm
import copy


class QATTrainer:
    """
    量化感知训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        calibration_loader: Optional[DataLoader] = None
    ):
        """
        Args:
            model: 待量化的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            criterion: 损失函数
            optimizer: 优化器
            device: 设备
            calibration_loader: 校准数据加载器（用于初始化量化参数）
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer if optimizer is not None else optim.Adam(
            self.model.parameters(), lr=1e-4
        )
        
        self.calibration_loader = calibration_loader
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _find_quantizable_modules(self, module: nn.Module, prefix: str = '') -> Dict[str, nn.Module]:
        """
        查找可量化模块（INT4Linear, INT4Conv2d等）
        
        Args:
            module: 模型
            prefix: 模块名前缀
            
        Returns:
            可量化模块字典 {name: module}
        """
        quantizable_modules = {}
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                # 标准层，需要替换
                quantizable_modules[full_name] = child
            elif hasattr(child, 'quantize_activation'):
                # 已经是量化层
                quantizable_modules[full_name] = child
            else:
                # 递归查找
                quantizable_modules.update(
                    self._find_quantizable_modules(child, full_name)
                )
        return quantizable_modules
    
    def calibrate(self, num_batches: int = 100):
        """
        校准：收集统计量并初始化量化参数
        
        Args:
            num_batches: 用于校准的批次数量
        """
        print("开始校准...")
        self.model.eval()
        
        if self.calibration_loader is None:
            loader = self.train_loader
        else:
            loader = self.calibration_loader
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(loader):
                if batch_idx >= num_batches:
                    break
                
                data = data.to(self.device)
                _ = self.model(data)
        
        # 更新所有量化器的参数
        for module in self.model.modules():
            if hasattr(module, 'update_quantization_params'):
                module.update_quantization_params()
        
        print("校准完成")
    
    def enable_quantization(self):
        """启用量化（激活量化感知训练模式）"""
        for module in self.model.modules():
            if hasattr(module, 'enable_activation_quantization'):
                module.enable_activation_quantization()
    
    def disable_quantization(self):
        """禁用量化（返回浮点模式）"""
        for module in self.model.modules():
            if hasattr(module, 'disable_activation_quantization'):
                module.disable_activation_quantization()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """
        验证
        
        Returns:
            验证指标字典
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def train(
        self,
        num_epochs: int = 10,
        start_quantization_epoch: int = 5,
        calibration_batches: int = 100,
        save_path: Optional[str] = None
    ):
        """
        量化感知训练主循环
        
        Args:
            num_epochs: 总训练轮数
            start_quantization_epoch: 开始量化感知训练的epoch（前几个epoch用浮点训练）
            calibration_batches: 校准使用的批次数量
            save_path: 模型保存路径
        """
        print(f"开始量化感知训练，总轮数: {num_epochs}")
        print(f"前 {start_quantization_epoch} 轮使用浮点训练，之后启用量化感知训练")
        
        # 初始校准
        self.calibrate(num_batches=calibration_batches)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # 从指定epoch开始启用量化
            if epoch >= start_quantization_epoch:
                self.enable_quantization()
                print(f"Epoch {epoch}: 启用量化感知训练")
            else:
                self.disable_quantization()
                print(f"Epoch {epoch}: 浮点训练")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            if 'loss' in val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
            
            # 打印结果
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if 'loss' in val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
            
            # 保存最佳模型
            if 'loss' in val_metrics and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'history': self.history
                    }, save_path)
                    print(f"  保存最佳模型到 {save_path}")
        
        print("训练完成")
        return self.history
    
    def export_quantized_model(self, save_path: str):
        """
        导出量化模型（保存量化参数）
        
        Args:
            save_path: 保存路径
        """
        # 确保量化参数已更新
        for module in self.model.modules():
            if hasattr(module, 'update_quantization_params'):
                module.update_quantization_params()
        
        # 保存模型和量化参数
        quantized_state = {
            'model_state_dict': self.model.state_dict(),
            'quantization_params': {}
        }
        
        # 收集量化参数
        for name, module in self.model.named_modules():
            if hasattr(module, 'scale') and hasattr(module, 'zero_point'):
                quantized_state['quantization_params'][name] = {
                    'scale': module.scale,
                    'zero_point': module.zero_point,
                    'symmetric': module.symmetric if hasattr(module, 'symmetric') else True,
                    'bits': module.bits if hasattr(module, 'bits') else 4
                }
        
        torch.save(quantized_state, save_path)
        print(f"量化模型已导出到 {save_path}")
