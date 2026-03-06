"""
CSPDarknet 双分支各阶段延迟占比扇形图
各阶段延迟基于 FLOPs 比例从实测总延迟（M=76.7ms, N=19.4ms）推算
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ------- CSPDarknet-M (left, width=1.0, depth=1.0, total=76.7ms) -------
# Stages: Stem | P3(stride8,C=256,3×Bottleneck) | P4(stride16,C=512,6×) | P5(stride32,C=1024,9×)
# FLOPs ratio estimated from channel×resolution×depth
m_labels = ['Stem\n(下采样至1/4)', 'P3 阶段\n(stride=8, C=256)', 'P4 阶段\n(stride=16, C=512)', 'P5 阶段\n(stride=32, C=1024)']
m_sizes  = [5.0, 24.5, 28.8, 18.4]   # ms, sum=76.7
m_colors = ['#90C2E7', '#2E86AB', '#1A5276', '#0D2B3E']
m_explode= (0.03, 0.03, 0.03, 0.03)

# ------- CSPDarknet-N (right, width=0.5, depth=0.34, total=19.4ms) ------
# Stages: Stem | P3(stride8,C=128,1×Bottleneck) | P4(stride16,C=256,2×) | P5(stride32,C=512,3×)
n_labels = ['Stem\n(下采样至1/4)', 'P3 阶段\n(stride=8, C=128)', 'P4 阶段\n(stride=16, C=256)', 'P5 阶段\n(stride=32, C=512)']
n_sizes  = [1.3, 4.5, 7.2, 6.4]      # ms, sum=19.4
n_colors = ['#D7A0C8', '#A23B72', '#6B1E47', '#3B0B27']
n_explode= (0.03, 0.03, 0.03, 0.03)

fig, (ax_m, ax_n) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=180)

def draw_pie(ax, sizes, labels, colors, explode, total, title):
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda p: f'{p:.1f}%',
        startangle=120,
        colors=colors,
        explode=explode,
        wedgeprops=dict(linewidth=1.2, edgecolor='white'),
        pctdistance=0.68,
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_color('white')
        at.set_fontweight('bold')

    legend_labels = [f'{l.replace(chr(10), " ")}  {s:.1f} ms' for l, s in zip(labels, sizes)]
    ax.legend(
        wedges, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.30),
        ncol=1,
        fontsize=11.5,
        frameon=False,
        handlelength=1.3,
        handleheight=0.9,
    )
    ax.set_title(title, fontsize=13, pad=12)

draw_pie(ax_m, m_sizes, m_labels, m_colors, m_explode, 76.7,
         'CSPDarknet-M（左分支，width=1.0，共 76.7 ms）')
draw_pie(ax_n, n_sizes, n_labels, n_colors, n_explode, 19.4,
         'CSPDarknet-N（右分支，width=0.5，共 19.4 ms）')

fig.suptitle('CSPDarknet 双分支各阶段延迟占比（FLOPs 比例估算，Xiaomi 14 / MNN / FP16）',
             fontsize=13, y=1.01)

plt.tight_layout()
out = 'f:/CSUthesis/images/CSPDarknetStageLatencyPie.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f'saved: {out}')
