"""
模块级延迟拆解工具

生成论文表 4.x 所需的各模块耗时数据，在 CPU 上模拟端侧 batch=1 推理。

使用方法：
  # 基本运行（CPU，随机权重）
  python tools/deploy/profile.py

  # 加载真实权重
  python tools/deploy/profile.py --weights path/to/model.pth

  # 调整计时次数（n 越大结果越稳定）
  python tools/deploy/profile.py --warmup 20 --repeat 200

输出示例：
  ┌─────────────────────────────────────────────────────────────────┐
  │             模块延迟拆解（CPU / batch=1 / FP32）                │
  ├────────────────────────────┬──────────┬────────┬───────────────┤
  │ 模块                       │ 延迟(ms) │ 占比(%)│ 主要算子       │
  ├────────────────────────────┼──────────┼────────┼───────────────┤
  │ 左分支 CSPDarknet-M        │   XX.X   │  XX.X  │ Conv,BN,SiLU  │
  │ 右分支 CSPDarknet-N        │   XX.X   │  XX.X  │ Conv,BN,SiLU  │
  │ EFEM 极线注意力             │   XX.X   │  XX.X  │ 1×1Conv,sftmx │
  │ 热力图头与置信度头          │   XX.X   │  XX.X  │ 1×1Conv       │
  │ DWT 三角化求解器            │   XX.X   │  XX.X  │ sftmx,eigmin  │
  ├────────────────────────────┼──────────┼────────┼───────────────┤
  │ 端到端合计                  │  116.0   │ 100.0  │               │
  └────────────────────────────┴──────────┴────────┴───────────────┘

注意：此脚本在开发机 CPU 上运行，所测延迟仅供模块占比参考。
      填入论文表格的绝对延迟数值需在 Xiaomi 14 / MNN / FP16 上实测。
"""

import sys
import time
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from model import StereoPoseNet, DWT

# ---------------------------------------------------------------------------
# 计时工具
# ---------------------------------------------------------------------------

class ModuleTimer:
    """
    用 register_forward_hook 对指定模块计时。
    记录每次前向的开始和结束时间（单位：秒）。
    """

    def __init__(self):
        self.records: Dict[str, List[float]] = {}
        self._handles = []
        self._start:   Dict[str, float] = {}

    def register(self, module: nn.Module, name: str):
        def pre_hook(mod, inp):
            self._start[name] = time.perf_counter()

        def post_hook(mod, inp, out):
            elapsed = time.perf_counter() - self._start.get(name, 0)
            self.records.setdefault(name, []).append(elapsed)

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def mean_ms(self, name: str, skip: int = 0) -> float:
        """返回指定模块的平均延迟（毫秒），跳过前 skip 次预热。"""
        data = self.records.get(name, [])
        if len(data) <= skip:
            return 0.0
        return sum(data[skip:]) / len(data[skip:]) * 1000


# ---------------------------------------------------------------------------
# 主测量逻辑
# ---------------------------------------------------------------------------

def measure_breakdown(
    model: StereoPoseNet,
    dwt:   DWT,
    warmup: int = 10,
    repeat: int = 100,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    对各模块注册计时 hook，运行 warmup+repeat 次前向，返回各模块均值延迟（ms）。
    """
    model = model.to(device).eval()
    dwt   = dwt.to(device)

    timer = ModuleTimer()

    # 注册各被测模块
    timer.register(model.backbone_left,    'backbone_left')
    timer.register(model.backbone_right,   'backbone_right')
    timer.register(model.efem,             'efem')
    # 热力图头和置信度头合并计时（用包装前向）
    timer.register(model.heatmap_head_l,   'head_l_hm')
    timer.register(model.heatmap_head_r,   'head_r_hm')
    timer.register(model.conf_head_l,      'head_l_cf')
    timer.register(model.conf_head_r,      'head_r_cf')

    img_l = torch.zeros(1, 3, 480, 640, device=device)
    img_r = torch.zeros(1, 3, 480, 640, device=device)

    # 随机投影矩阵（用于 DWT 计时）
    P_l = torch.eye(3, 4, device=device)
    P_r = torch.eye(3, 4, device=device); P_r[0, 3] = -100.0

    e2e_times: List[float] = []
    dwt_times: List[float] = []

    total = warmup + repeat
    print(f'  运行 {warmup} 次预热 + {repeat} 次计时 ...')

    for i in range(total):
        t_start = time.perf_counter()

        with torch.no_grad():
            hm_l, hm_r, cf_l, cf_r = model(img_l, img_r)

        # DWT 单独计时
        t_dwt_s = time.perf_counter()
        with torch.no_grad():
            _ = dwt(hm_l, hm_r, cf_l, cf_r, P_l, P_r, stride=16)
        t_dwt_e = time.perf_counter()

        t_end = time.perf_counter()

        if i >= warmup:
            e2e_times.append(t_end - t_start)
            dwt_times.append(t_dwt_e - t_dwt_s)

    timer.remove_hooks()

    # 聚合各模块延迟
    ms = lambda name: timer.mean_ms(name, skip=warmup)

    results = {
        'backbone_left':  ms('backbone_left'),
        'backbone_right': ms('backbone_right'),
        'efem':           ms('efem'),
        'heads': (ms('head_l_hm') + ms('head_r_hm') +
                  ms('head_l_cf') + ms('head_r_cf')),
        'dwt':            sum(dwt_times) / len(dwt_times) * 1000,
        'e2e':            sum(e2e_times) / len(e2e_times) * 1000,
    }
    # e2e 含 DWT
    results['e2e_with_dwt'] = results['e2e'] + results['dwt']
    return results


# ---------------------------------------------------------------------------
# 格式化输出
# ---------------------------------------------------------------------------

ROWS = [
    ('backbone_left',  '左分支 CSPDarknet-M',   'Conv, BN, SiLU, CSP'),
    ('backbone_right', '右分支 CSPDarknet-N',   'Conv, BN, SiLU, CSP'),
    ('efem',           'EFEM 极线注意力',        '1×1 Conv, unfold, softmax'),
    ('heads',          '热力图头与置信度头',     '1×1 Conv, GAP, Linear'),
    ('dwt',            'DWT 三角化求解器',       'softmax, matmul, eigmin'),
]


def print_table(results: Dict[str, float], total_ref: float = None):
    """打印延迟拆解表格。"""
    total = results['e2e_with_dwt']
    if total_ref is not None:
        # 按实测总延迟比例缩放（用于估算端侧相对占比）
        scale = total_ref / total if total > 0 else 1.0
    else:
        scale = 1.0

    W = [28, 10, 8, 22]
    sep = '├' + '┼'.join('─' * w for w in W) + '┤'
    top = '┌' + '┬'.join('─' * w for w in W) + '┐'
    bot = '└' + '┴'.join('─' * w for w in W) + '┘'
    hdr = '┌' + '─' * (sum(W) + len(W) - 1) + '┐'

    def row(cols):
        return '│' + '│'.join(f' {c:<{W[i]-2}} ' for i, c in enumerate(cols)) + '│'

    title = '模块延迟拆解（CPU / batch=1 / FP32）'
    print(hdr)
    print(f'│ {title:^{sum(W)+len(W)-1}} │')
    print(top)
    print(row(['模块', '延迟(ms)', '占比(%)', '主要算子']))
    print(sep)

    for key, label, ops in ROWS:
        ms_val = results.get(key, 0.0) * scale
        pct    = ms_val / (total * scale) * 100 if total > 0 else 0
        print(row([label, f'{ms_val:>7.1f}', f'{pct:>5.1f}', ops]))

    print(sep)
    total_scaled = total * scale
    print(row(['端到端合计', f'{total_scaled:>7.1f}', '100.0', '']))
    print(bot)


def save_csv(results: Dict[str, float], output_path: Path, total_ref: float = None):
    """将结果保存为 CSV（供论文表格使用）。"""
    total = results['e2e_with_dwt']
    scale = (total_ref / total) if (total_ref and total > 0) else 1.0

    rows = []
    for key, label, ops in ROWS:
        ms_val = results.get(key, 0.0) * scale
        pct    = ms_val / (total * scale) * 100 if total > 0 else 0
        rows.append({'模块': label, '延迟(ms)': f'{ms_val:.1f}',
                     '占比(%)': f'{pct:.1f}', '主要算子': ops})
    rows.append({'模块': '端到端合计', '延迟(ms)': f'{total*scale:.1f}',
                 '占比(%)': '100.0', '主要算子': ''})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['模块', '延迟(ms)', '占比(%)', '主要算子'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n结果已保存: {output_path}')


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='StereoPoseNet 模块延迟拆解')
    parser.add_argument('--weights',    type=str,   default=None,
                        help='权重文件路径（不传则用随机权重）')
    parser.add_argument('--warmup',     type=int,   default=10,
                        help='预热次数（默认 10）')
    parser.add_argument('--repeat',     type=int,   default=100,
                        help='正式计时次数（默认 100）')
    parser.add_argument('--device',     type=str,   default='cpu',
                        choices=['cpu', 'cuda'],
                        help='计时设备（默认 cpu，对应端侧环境）')
    parser.add_argument('--total-ref',  type=float, default=116.0,
                        help='端侧实测总延迟(ms)，用于等比例估算各模块占比'
                             '（默认 116.0ms，来自第三章 Xiaomi 14/MNN/FP16 实测）')
    parser.add_argument('--output-csv', type=str,   default=None,
                        help='CSV 输出路径（不传则只打印，不保存）')
    args = parser.parse_args()

    print('=' * 65)
    print('StereoPoseNet 模块延迟拆解')
    print('=' * 65)
    print(f'设备: {args.device} | 预热: {args.warmup} | 计时: {args.repeat}')
    if args.total_ref:
        print(f'端侧参考总延迟: {args.total_ref} ms（来自 Xiaomi 14/MNN/FP16 实测）')

    # 构建模型
    print('\n构建模型 ...')
    model = StereoPoseNet()
    if args.weights:
        ckpt = torch.load(args.weights, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        model.load_state_dict(state, strict=False)
        print(f'  已加载权重: {args.weights}')
    else:
        print('  使用随机权重（延迟数值仅供占比参考）')

    dwt = DWT()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'  参数量: {params:.2f} M')

    # 测量
    print()
    results = measure_breakdown(
        model, dwt,
        warmup=args.warmup,
        repeat=args.repeat,
        device=args.device,
    )

    # 输出表格
    print()
    print_table(results, total_ref=args.total_ref)

    # 额外信息
    print(f'\n[本机实测端到端 FP32 CPU] {results["e2e_with_dwt"]:.1f} ms')
    if args.total_ref:
        print(f'[端侧参考（MNN/FP16）]     {args.total_ref:.1f} ms')
        print(f'[注意] 各模块占比基于本机 FP32 比例估算，绝对值需在目标设备实测')

    # 保存 CSV
    if args.output_csv:
        save_csv(results, Path(args.output_csv), total_ref=args.total_ref)
    else:
        csv_default = Path('output/deploy/latency_breakdown.csv')
        save_csv(results, csv_default, total_ref=args.total_ref)


if __name__ == '__main__':
    main()
