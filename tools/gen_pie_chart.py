import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

labels = [
    '左分支骨干 (CSPDarknet-M)',
    '右分支骨干 (CSPDarknet-N)',
    '极线特征增强 (EFEM)',
    '热力图头与置信度头',
    'DWT 三角化求解器',
]
sizes   = [76.7, 19.4, 10.0, 7.0, 1.1]
colors  = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8F71']
explode = (0.04, 0.02, 0.03, 0.02, 0.06)

fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=180)

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=None,
    autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
    startangle=130,
    colors=colors,
    explode=explode,
    wedgeprops=dict(linewidth=1.2, edgecolor='white'),
    pctdistance=0.70,
)

for at in autotexts:
    at.set_fontsize(15)
    at.set_color('white')
    at.set_fontweight('bold')

legend_labels = [f'{l}  {s:.1f} ms' for l, s in zip(labels, sizes)]
ax.legend(
    wedges, legend_labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.24),
    ncol=2,
    fontsize=13,
    frameon=False,
    handlelength=1.4,
    handleheight=1.0,
    columnspacing=1.2,
)

ax.set_title('各模块端到端延迟占比（Xiaomi 14 / MNN / FP16 / 4线程，单人场景，共 116.0 ms）',
             fontsize=13.5, pad=14)

plt.tight_layout()
out = 'f:/CSUthesis/images/ModuleDelayRatioFanChart.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print(f'saved: {out}')
