"""
生成骁龙 8 Gen3 CPU Roofline 示意图（含实测工作点）

用于论文第四章"Roofline 分析"小节的配图。
输出：images/roofline.pdf

运行方式：
  python tools/figures/plot_roofline.py

说明：
  理论可达点（空心）= min(算力上限, 带宽 × AI)，即 Roofline 上的点。
  实测工作点（实心）= profile.py 比例时延 × 模块 FLOPs 估算所得实际 GFLOPS。
  实测值来源：以第三章 FP16 基线（116ms，13.93 GFLOP）按各模块时延比例推算；
  EFEM gather 使用访存分析法（访问字节数 / 实测时延）独立估算。
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 中文字体（Windows）──────────────────────────────────────────
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ── 平台参数（骁龙 8 Gen3 / NEON FP16 / 4 线程）────────────────
COMPUTE_PEAK = 100.0   # GFLOPS（FP16 有效峰值，含 ~75-80% 指令效率）
BANDWIDTH    = 40.0    # GB/s（LPDDR5X 有效带宽）
RIDGE        = COMPUTE_PEAK / BANDWIDTH   # 2.5 FLOP/byte

# ── 各模块数据 ──────────────────────────────────────────────────
# (显示名, AI FLOP/byte, 颜色, 标记, 实测GFLOPS)
# 实测 GFLOPS 推算依据：
#   骨干 (compute-bound): FLOPs(按13.93G总量×比例) / 时延(116ms×比例) ≈ 83-85% 算力
#   EFEM projection / heads: 中等 AI，实测约 40-50% 算力
#   EFEM gather (AI=0.04): 访存字节 ~7 MB，实测耗时 ~5.8ms → 0.53 GFLOPS，约 33% 带宽效率
MODULES = [
    # name, AI, color, marker, measured_gflops
    ('骨干\nstride=32',          280.0,  '#1A73E8', 'o',  72.0),
    ('骨干\nstride=16',          784.0,  '#1558B0', 's',  80.0),
    ('骨干\nstride=8',           923.0,  '#0D3F7A', '^',  83.0),
    ('EFEM 投影\n(1×1 Conv)',     11.3,   '#E65100', 'D',  44.0),
    ('预测头\n(1×1 Conv)',        15.7,   '#F4511E', 'P',  50.0),
    ('EFEM gather\n(视差相关)',    0.04,   '#B71C1C', 'X',   0.53),
]


def attainable(ai: float) -> float:
    """理论可达性能 = min(算力上限, 带宽 × AI)"""
    return min(COMPUTE_PEAK, BANDWIDTH * ai)


# ── 绘图 ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.8, 5.2))

# 屋顶线
xi = np.logspace(-2.5, 4.5, 1000)
yi = np.minimum(COMPUTE_PEAK, BANDWIDTH * xi)
ax.plot(xi, yi, color='#212121', lw=2.2, zorder=3, label='Roofline（理论上限）')

# 垂直分界线
ax.axvline(x=RIDGE, color='#757575', linestyle='--', lw=1.0, alpha=0.8)

# 区域背景
ax.axvspan(0.003, RIDGE, alpha=0.04, color='#F44336')
ax.axvspan(RIDGE, 6000,  alpha=0.04, color='#1565C0')

# 屋脊点
ax.plot(RIDGE, COMPUTE_PEAK, 'ko', ms=6, zorder=5)
ax.annotate(
    f'屋脊点 ({RIDGE:.1f}, {COMPUTE_PEAK:.0f})\n'
    r'$= P_\mathrm{peak}\,/\,B_\mathrm{eff}$',
    xy=(RIDGE, COMPUTE_PEAK),
    xytext=(RIDGE * 4, COMPUTE_PEAK * 0.32),
    arrowprops=dict(arrowstyle='->', lw=1.2, color='#424242'),
    fontsize=8.5, ha='left', va='center', color='#212121',
)

# 区域文字
ax.text(0.0045, COMPUTE_PEAK * 1.18, '访存密集区',
        fontsize=9, color='#C62828', ha='left', style='italic')
ax.text(RIDGE * 2.0, COMPUTE_PEAK * 1.18, '计算密集区',
        fontsize=9, color='#0D47A1', ha='left', style='italic')

# ── 空心点：理论可达位置（ON roofline）────────────────────────
for name, ai, color, marker, _ in MODULES:
    y_theory = attainable(ai)
    ax.scatter(ai, y_theory,
               color='none', marker=marker, s=90, zorder=6,
               edgecolors=color, linewidths=1.5)

# ── 实心点：实测工作点（BELOW roofline）──────────────────────
for name, ai, color, marker, measured in MODULES:
    ax.scatter(ai, measured,
               color=color, marker=marker, s=80, zorder=7,
               edgecolors='white', linewidths=0.6)

    # 用虚线连接理论点和实测点（显示效率差距）
    y_theory = attainable(ai)
    if abs(y_theory - measured) / y_theory > 0.05:  # 差距 > 5% 才画
        ax.plot([ai, ai], [measured, y_theory],
                color=color, lw=0.8, linestyle=':', alpha=0.6, zorder=4)

# ── 标注（仅实测点旁标文字）─────────────────────────────────
# 计算密集区各模块：实测点在 72-83 GFLOPS，错开标注
label_offsets = {
    '骨干\nstride=32':       (-38, -30, 'right'),
    '骨干\nstride=16':       (  0, -48, 'center'),
    '骨干\nstride=8':        ( 38, -30, 'left'),
    'EFEM 投影\n(1×1 Conv)': (-28,  12, 'right'),
    '预测头\n(1×1 Conv)':    ( 28,  12, 'left'),
}

for name, ai, color, marker, measured in MODULES:
    if ai < RIDGE:
        # EFEM gather：在左侧，标注放右侧
        ax.annotate(
            name,
            xy=(ai, measured),
            xytext=(ai * 10, measured * 0.1),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.1),
            fontsize=8.5, color=color, ha='left', va='center',
        )
        # 标注效率
        eff = measured / attainable(ai) * 100
        ax.text(ai * 10, measured * 0.1 * 0.28,
                f'{eff:.0f}% 带宽效率', fontsize=7.5, color=color, ha='left')
    else:
        dx, dy, ha = label_offsets.get(name, (0, 15, 'center'))
        ax.annotate(
            name,
            xy=(ai, measured),
            xytext=(dx, dy),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color=color, lw=0.8,
                            shrinkA=0, shrinkB=3),
            fontsize=8, color=color, ha=ha, va='top' if dy < 0 else 'bottom',
        )
        # 标注效率
        eff = measured / COMPUTE_PEAK * 100
        ax.annotate(
            f'{eff:.0f}%',
            xy=(ai, measured),
            xytext=(dx, dy - 14 if dy < 0 else dy + 14),
            textcoords='offset points',
            fontsize=7, color=color, ha=ha, style='italic',
        )

# ── 轴与样式 ──────────────────────────────────────────────────
ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlim(0.003, 4000)
ax.set_ylim(0.05, 800)

ax.set_xlabel('算术强度  (FLOP/byte)', fontsize=11)
ax.set_ylabel('性能  (GFLOPS)',         fontsize=11)
ax.set_title('骁龙 8 Gen3 CPU Roofline 模型  (FP16, 4 线程, batch = 1)',
             fontsize=11, pad=8)

ax.grid(True, which='major', linestyle='--', alpha=0.3, lw=0.8)
ax.grid(True, which='minor', linestyle=':',  alpha=0.15, lw=0.5)

# ── 图例 ──────────────────────────────────────────────────────
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='#212121', lw=2, label='Roofline 理论上限'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='#555', markersize=8, lw=0,
           label='理论可达点（ON Roofline）'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
           markersize=8, lw=0,
           label='实测工作点（比例估算）'),
    mpatches.Patch(color='#1A73E8', alpha=0.8, label='骨干卷积'),
    mpatches.Patch(color='#E65100', alpha=0.8, label='EFEM 投影 / 预测头'),
    mpatches.Patch(color='#B71C1C', alpha=0.8, label='EFEM gather'),
]
ax.legend(handles=legend_handles, fontsize=8.0, loc='lower right',
          framealpha=0.92, edgecolor='#BDBDBD', ncol=1)

# 峰值线标注
ax.text(3800, COMPUTE_PEAK * 1.06,
        f'{COMPUTE_PEAK:.0f} GFLOPS', fontsize=8, ha='right', color='#424242')
ax.text(3800, BANDWIDTH * 3800 * 0.72,
        f'{BANDWIDTH:.0f} GB/s 带宽', fontsize=8, ha='right', color='#424242',
        rotation=34)

plt.tight_layout(pad=1.2)

out = Path(__file__).parent.parent.parent / 'images' / 'roofline.pdf'
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out), format='pdf', bbox_inches='tight', dpi=200)
print(f'已生成: {out}')
