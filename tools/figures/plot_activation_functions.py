"""
绘制三种常见激活函数曲线（ReLU / Sigmoid / Tanh）

用于论文第二章“激活函数”小节的配图。
输出：
  - images/activation_functions.pdf
  - images/activation_functions.png

运行方式：
  python tools/figures/plot_activation_functions.py
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def main() -> None:
    x = np.linspace(-6.0, 6.0, 1200, dtype=np.float64)

    y_relu = relu(x)
    y_sigmoid = sigmoid(x)
    y_tanh = np.tanh(x)

    # ── 中文字体（Windows 优先）────────────────────────────────────
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharex=True)

    # ReLU
    ax = axes[0]
    ax.plot(x, y_relu, lw=2.2, color='#1A73E8')
    ax.axhline(0, color='#666', lw=0.8, alpha=0.6)
    ax.axvline(0, color='#666', lw=0.8, alpha=0.6)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.set_title('ReLU')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1.0, 6.2)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

    # Sigmoid
    ax = axes[1]
    ax.plot(x, y_sigmoid, lw=2.2, color='#E65100')
    ax.axhline(0, color='#666', lw=0.8, alpha=0.6)
    ax.axvline(0, color='#666', lw=0.8, alpha=0.6)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.set_title('Sigmoid')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('x')

    # Tanh
    ax = axes[2]
    ax.plot(x, y_tanh, lw=2.2, color='#0D3F7A')
    ax.axhline(0, color='#666', lw=0.8, alpha=0.6)
    ax.axvline(0, color='#666', lw=0.8, alpha=0.6)
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.set_title('Tanh')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('x')

    fig.suptitle('常见激活函数（分坐标系）', y=1.02)
    plt.tight_layout()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pdf = out_dir / 'activation_functions.pdf'
    out_png = out_dir / 'activation_functions.png'

    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=200)
    print(f'已生成: {out_pdf}')
    print(f'已生成: {out_png}')


if __name__ == '__main__':
    main()
