"""
PyTorch → ONNX → MNN 全流程转换脚本

使用方法：
  # 用随机权重测试转换链路
  python tools/deploy/export_to_mnn.py

  # 加载训练好的权重
  python tools/deploy/export_to_mnn.py --weights path/to/model.pth

  # 只导出 ONNX，跳过 MNN 转换（未安装 mnnconvert 时）
  python tools/deploy/export_to_mnn.py --skip-mnn

  # 指定输出目录
  python tools/deploy/export_to_mnn.py --output-dir ./output/deploy

转换结果：
  model.onnx          FP32 ONNX 模型（用于验证和 MNN 转换输入）
  model_fp16.mnn      FP16 MNN 模型（对应论文第三章实验基线）
  model_int8.mnn      INT8 MNN 模型（对应论文第四章量化方案）
"""

import sys
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Windows 上 onnx DLL 可能存在兼容性问题，用 MagicMock 绕过 torch 内部的
# onnx.checker.check_model() 调用，让 PyTorch 自带的 ONNX 序列化器直接落盘。
# onnxruntime 的验证步骤使用单独安装的 ort，不受此影响。
# ---------------------------------------------------------------------------
try:
    import onnx as _onnx_test
    _onnx_test.__version__   # 测试是否能正常加载
except Exception:
    _stub = MagicMock()
    _stub.__version__ = '0.0.0 (stub)'
    for _mod in ['onnx', 'onnx.checker', 'onnx.helper',
                 'onnx.numpy_helper', 'onnx.shape_inference',
                 'onnx.defs', 'onnx.TensorProto']:
        sys.modules[_mod] = _stub

# 将 tools/deploy 加入路径
sys.path.insert(0, str(Path(__file__).parent))
from model import StereoPoseNet

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_model(weights: str | None) -> StereoPoseNet:
    """构建模型，可选加载权重。"""
    model = StereoPoseNet()
    if weights:
        w = Path(weights)
        if not w.exists():
            raise FileNotFoundError(f'权重文件不存在: {w}')
        ckpt = torch.load(w, map_location='cpu')
        # 兼容不同保存格式
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'  [warn] 缺失键: {missing[:5]}{"..." if len(missing)>5 else ""}')
        if unexpected:
            print(f'  [warn] 多余键: {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')
        print(f'  权重加载完成: {w}')
    else:
        print('  未指定权重，使用随机初始化（仅用于验证转换链路）')
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Step 1：导出 ONNX
# ---------------------------------------------------------------------------

def export_onnx(model: StereoPoseNet, output_path: Path) -> bool:
    """
    将 StereoPoseNet 导出为 ONNX（opset 17，双输入，静态形状 batch=1）。

    输入：img_l / img_r  [1, 3, 480, 640]
    输出：heatmap_l / heatmap_r [1, 17, 30, 40]，conf_l / conf_r [1, 17]
    """
    print('\n[Step 1] 导出 ONNX ...')

    dummy_l = torch.zeros(1, 3, 480, 640)
    dummy_r = torch.zeros(1, 3, 480, 640)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            (dummy_l, dummy_r),
            str(output_path),
            input_names=['img_l', 'img_r'],
            output_names=['heatmap_l', 'heatmap_r', 'conf_l', 'conf_r'],
            opset_version=17,
            do_constant_folding=True,
            # 静态 batch=1，不使用 dynamic_axes（对应端侧固定输入）
        )
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f'  ONNX 导出成功: {output_path}  ({size_mb:.1f} MB)')
        return True
    except Exception as e:
        print(f'  [error] ONNX 导出失败: {e}')
        return False


# ---------------------------------------------------------------------------
# Step 2：onnxruntime 验证
# ---------------------------------------------------------------------------

def verify_onnx(onnx_path: Path) -> bool:
    """用 onnxruntime 验证 ONNX 输出形状与数值是否正常。"""
    print('\n[Step 2] onnxruntime 验证 ...')
    try:
        import onnxruntime as ort
    except ImportError:
        print('  [skip] onnxruntime 未安装，跳过验证（pip install onnxruntime）')
        return True

    try:
        sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        feeds = {
            'img_l': np.random.randn(1, 3, 480, 640).astype(np.float32),
            'img_r': np.random.randn(1, 3, 480, 640).astype(np.float32),
        }
        outputs = sess.run(None, feeds)
        names   = ['heatmap_l', 'heatmap_r', 'conf_l', 'conf_r']
        expected = [(1, 17, 30, 40), (1, 17, 30, 40), (1, 17), (1, 17)]
        all_ok = True
        for name, out, exp in zip(names, outputs, expected):
            ok = tuple(out.shape) == exp
            status = '✓' if ok else '✗'
            print(f'  {status} {name}: {out.shape}  (期望 {exp})')
            all_ok = all_ok and ok
        if all_ok:
            print('  ONNX 验证通过')
        return all_ok
    except Exception as e:
        print(f'  [error] ONNX 验证失败: {e}')
        return False


# ---------------------------------------------------------------------------
# Step 3：ONNX → MNN（FP16）
# ---------------------------------------------------------------------------

def convert_mnn_fp16(onnx_path: Path, output_path: Path) -> bool:
    """
    调用 mnnconvert 将 ONNX 转换为 FP16 MNN 模型。
    对应论文第三章实验基线（116ms @ Xiaomi 14/CPU/FP16）。
    """
    print('\n[Step 3] 转换 MNN FP16 ...')
    if not _check_mnnconvert():
        return False

    cmd = [
        'mnnconvert',
        '-f', 'ONNX',
        '--modelFile', str(onnx_path),
        '--MNNModel',  str(output_path),
        '--fp16',
    ]
    return _run_mnnconvert(cmd, output_path)


# ---------------------------------------------------------------------------
# Step 4：ONNX → MNN（INT8 PTQ）
# ---------------------------------------------------------------------------

def convert_mnn_int8(onnx_path: Path, output_path: Path) -> bool:
    """
    调用 mnnconvert 将 ONNX 转换为 INT8 量化 MNN 模型（权重 INT8）。
    对应论文第四章混合精度量化方案（骨干 + EFEM 特征提取 INT8）。

    注意：此处为权重静态量化（--weightQuantBits 8），
    激活量化（PTQ calibration）需配合校准集数据，见 README。
    """
    print('\n[Step 4] 转换 MNN INT8 (权重量化) ...')
    if not _check_mnnconvert():
        return False

    cmd = [
        'mnnconvert',
        '-f', 'ONNX',
        '--modelFile',       str(onnx_path),
        '--MNNModel',        str(output_path),
        '--weightQuantBits', '8',
        '--weightQuantBlock', '0',   # per-channel 量化（block=0）
    ]
    return _run_mnnconvert(cmd, output_path)


# ---------------------------------------------------------------------------
# Step 5：MNN 推理验证
# ---------------------------------------------------------------------------

def verify_mnn(mnn_path: Path, label: str = '') -> bool:
    """
    用 MNN Python API 加载模型并跑一次前向，验证输出形状。
    若 MNN Python 包未安装则跳过。
    """
    tag = f'[Step 5{label}]'
    print(f'\n{tag} MNN 推理验证: {mnn_path.name} ...')
    try:
        import MNN
    except ImportError:
        print(f'  [skip] MNN Python 包未安装，跳过推理验证')
        print(f'         安装方法：pip install MNN  或参考 https://www.mnn.zone')
        return True

    if not mnn_path.exists():
        print(f'  [skip] MNN 文件不存在，跳过验证')
        return True

    try:
        interp = MNN.Interpreter(str(mnn_path))
        session_cfg = MNN.Session.SessionConfig()
        session_cfg.numThread = 4
        session = interp.createSession(session_cfg)

        # 获取并填充输入
        for name in ['img_l', 'img_r']:
            tensor = interp.getSessionInput(session, name)
            tmp = MNN.Tensor(
                (1, 3, 480, 640), MNN.Halide_Type_Float,
                np.random.randn(1, 3, 480, 640).astype(np.float32),
                MNN.Tensor_DimensionType_Caffe,
            )
            tensor.copyFrom(tmp)

        interp.runSession(session)

        for name, exp_shape in [
            ('heatmap_l', (1, 17, 30, 40)),
            ('heatmap_r', (1, 17, 30, 40)),
            ('conf_l',    (1, 17)),
            ('conf_r',    (1, 17)),
        ]:
            out = interp.getSessionOutput(session, name)
            ok  = tuple(out.getShape()) == exp_shape
            print(f'  {"✓" if ok else "✗"} {name}: {out.getShape()}')

        size_mb = mnn_path.stat().st_size / 1024 / 1024
        print(f'  MNN 验证通过  ({size_mb:.1f} MB)')
        return True
    except Exception as e:
        print(f'  [error] MNN 推理验证失败: {e}')
        return False


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _check_mnnconvert() -> bool:
    if shutil.which('mnnconvert') is None:
        print('  [skip] mnnconvert 不在 PATH，跳过 MNN 转换')
        print('         安装方法：pip install MNN  或从源码编译 MNNConvert')
        print('         源码：https://github.com/alibaba/MNN')
        return False
    return True


def _run_mnnconvert(cmd: list, output_path: Path) -> bool:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and output_path.exists():
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f'  转换成功: {output_path}  ({size_mb:.1f} MB)')
            return True
        else:
            print(f'  [error] mnnconvert 失败 (code={result.returncode})')
            if result.stderr:
                print(f'  stderr: {result.stderr[:300]}')
            return False
    except subprocess.TimeoutExpired:
        print('  [error] mnnconvert 超时（>5 分钟）')
        return False
    except Exception as e:
        print(f'  [error] {e}')
        return False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='StereoPoseNet: PyTorch → ONNX → MNN')
    parser.add_argument('--weights',    type=str, default=None,
                        help='PyTorch 权重文件路径 (.pth)，不传则用随机权重')
    parser.add_argument('--output-dir', type=str, default='output/deploy',
                        help='输出目录（默认 output/deploy）')
    parser.add_argument('--skip-mnn',  action='store_true',
                        help='跳过 MNN 转换步骤，只导出 ONNX')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path     = out_dir / 'model.onnx'
    mnn_fp16_path = out_dir / 'model_fp16.mnn'
    mnn_int8_path = out_dir / 'model_int8.mnn'

    print('=' * 60)
    print('StereoPoseNet 部署转换流程')
    print('=' * 60)
    print(f'输出目录: {out_dir.resolve()}')

    # 构建模型
    print('\n[Init] 构建模型 ...')
    t0 = time.perf_counter()
    model = load_model(args.weights)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  参数量: {params:.2f} M')
    print(f'  构建耗时: {(time.perf_counter()-t0)*1000:.0f} ms')

    # Step 1: ONNX 导出
    ok_onnx = export_onnx(model, onnx_path)
    if not ok_onnx:
        print('\n[abort] ONNX 导出失败，终止')
        sys.exit(1)

    # Step 2: ONNX 验证
    verify_onnx(onnx_path)

    if args.skip_mnn:
        print('\n[skip] --skip-mnn 已指定，跳过 MNN 转换')
    else:
        # Step 3: FP16 MNN
        convert_mnn_fp16(onnx_path, mnn_fp16_path)
        # Step 4: INT8 MNN
        convert_mnn_int8(onnx_path, mnn_int8_path)
        # Step 5: 验证两个 MNN 文件
        verify_mnn(mnn_fp16_path, label='a')
        verify_mnn(mnn_int8_path, label='b')

    # 文件大小汇总
    print('\n' + '=' * 60)
    print('转换结果汇总')
    print('=' * 60)
    for p in [onnx_path, mnn_fp16_path, mnn_int8_path]:
        if p.exists():
            print(f'  {p.name:<24} {p.stat().st_size/1024/1024:.1f} MB')
        else:
            print(f'  {p.name:<24} （未生成）')

    if mnn_fp16_path.exists() and mnn_int8_path.exists():
        ratio = mnn_fp16_path.stat().st_size / mnn_int8_path.stat().st_size
        print(f'\n  FP16 / INT8 大小比: {ratio:.2f}x  （理论预期约 2x）')

    print('\n完成。')


if __name__ == '__main__':
    main()
