"""
骁龙 8 Gen3 CPU Roofline 图（彻底解决标注重叠）
输出：images/roofline.pdf
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    'font.family':        'SimHei',
    'axes.unicode_minus': False,
    'font.size':          50,
})

PEAK  = 100.0
BW    = 40.0
RIDGE = PEAK / BW   # 2.5

# ── 各模块（名, AI, 颜色, 标记, 实测GFLOPS）──────────────────────────────
MODULES = [
    ('EFEM gather',   0.04,  '#B71C1C', 'X',  0.53),
    ('EFEM 投影',     11.3,  '#E87722', 'D', 44.0),
    ('预测头',        15.7,  '#F4B942', 'P', 50.0),
    ('骨干 P5',      280.0,  '#5B9BD5', 'o', 72.0),
    ('骨干 P4',      784.0,  '#2878B5', 's', 80.0),
    ('骨干 P3',      923.0,  '#1B4F8A', '^', 83.0),
]

# ── 画布 ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(30.0, 20.0))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── 屋顶线 ────────────────────────────────────────────────────────────────
xi = np.logspace(-2.5, 4.0, 2000)
yi = np.minimum(PEAK, BW * xi)
ax.plot(xi, yi, color='#111', lw=3.0, zorder=5)

ax.axvline(x=RIDGE, color='#9E9E9E', ls='--', lw=1.2, alpha=0.8, zorder=4)
ax.axvspan(0.003, RIDGE, alpha=0.05, color='#F44336', zorder=1)
ax.axvspan(RIDGE, 12000, alpha=0.05, color='#1565C0', zorder=1)

# 区域标签 — 顶部
ax.text(0.0043, 1100, '访存密集区', fontsize=50, color='#C62828',
        ha='left', style='italic', va='top')
ax.text(3.5, 1100, '计算密集区', fontsize=50, color='#0D47A1',
        ha='left', style='italic', va='top')

# 屋脊点
ax.plot(RIDGE, PEAK, 'k.', ms=14, zorder=9)
ax.annotate(
    f'屋脊点 ({RIDGE:.1f}, {PEAK:.0f})\n'
    r'$= P_\mathrm{peak}\,/\,B_\mathrm{eff}$',
    xy=(RIDGE, PEAK),
    xytext=(0.35, 50),
    fontsize=50, ha='left', va='top', color='#212121', zorder=10,
    arrowprops=dict(arrowstyle='->', lw=1.4, color='#555',
                    connectionstyle='arc3,rad=0.25'),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc',
              alpha=0.9, lw=0.8),
)

# ── 画点 ─────────────────────────────────────────────────────────────────
for _, ai, color, mk, meas in MODULES:
    y_th = min(PEAK, BW * ai)
    ax.scatter(ai, y_th,  color='none', marker=mk, s=600, zorder=7,
               edgecolors=color, linewidths=3.0)
    ax.scatter(ai, meas,  color=color,  marker=mk, s=500, zorder=8,
               edgecolors='white', linewidths=1.5)
    if abs(y_th - meas) / y_th > 0.05:
        ax.plot([ai, ai], [meas, y_th], color=color, lw=1.2,
                ls=':', alpha=0.6, zorder=4)

# ── 标注函数 ──────────────────────────────────────────────────────────────
def ann(pt_xy, txt_xy, label, eff, color, ha, va, rad=0.0):
    cs = f'arc3,rad={rad}'
    ax.annotate(f'{label}  {eff}',
        xy=pt_xy, xytext=txt_xy,
        fontsize=50, color=color, ha=ha, va=va, zorder=11,
        arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                        connectionstyle=cs, shrinkB=6),
    )

# ── 标注布局（分散到四个象限）────────────────────────────────────────────
#
#  左下区  y < 5  ：EFEM gather, 带宽
#  左中区  y 15-200：预测头, 屋脊点, EFEM 投影
#  左上区  y 650-1000：骨干 P5
#  右中区  y 300-600：骨干 P4, 骨干 P3
#

# EFEM gather：移到左侧，文字贴左边框
ann((0.04, 0.53),  (0.004, 0.8),
    'EFEM gather (视差相关)', '33%带宽',
    '#B71C1C', 'left', 'bottom', rad=0.0)

# 预测头：左侧低位
ann((15.7, 50.0),  (0.004, 18),
    '预测头 (1×1 Conv)', '50%',
    '#F4B942', 'left', 'bottom', rad=-0.15)

# EFEM 投影：左侧中位（屋脊点下方）
ann((11.3, 44.0),  (0.004, 110),
    'EFEM 投影 (1×1 Conv)', '44%',
    '#E87722', 'left', 'bottom', rad=0.12)

# 骨干 P5：Roofline 水平段下方，偏左
ann((280, 72.0),   (50, 62),
    '骨干 P5 (stride=32)', '72%',
    '#5B9BD5', 'left', 'top', rad=0.18)

# 骨干 P4：Roofline 水平段下方，偏右
ann((784, 80.0),   (6500, 32),
    '骨干 P4 (stride=16)', '80%',
    '#2878B5', 'right', 'top', rad=-0.20)

# 骨干 P3：右对齐，中下（P4 下方留足间距）
ann((923, 83.0),   (6500, 310),
    '骨干 P3 (stride=8)', '83%',
    '#1B4F8A', 'right', 'top', rad=-0.15)

# ── 轴 ───────────────────────────────────────────────────────────────────
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.003, 8000)
ax.set_ylim(0.18, 1200)

ax.set_xlabel('算术强度  (FLOP/byte)', fontsize=50, labelpad=8)
ax.set_ylabel('性能  (GFLOPS)',        fontsize=50, labelpad=8)
ax.set_title('骁龙 8 Gen3 CPU  Roofline 模型  (FP16, 4 线程, batch = 1)',
             fontsize=50, fontweight='bold', pad=14)

ax.grid(True, which='major', ls='--', alpha=0.22, lw=0.9, color='#777')
ax.grid(True, which='minor', ls=':',  alpha=0.10, lw=0.6, color='#aaa')

for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor('#333')
    sp.set_linewidth(1.1)
ax.tick_params(which='both', direction='in', top=True, right=True,
               length=6, width=1.0)

# 峰值标注
ax.text(4.5, PEAK * 1.06, f'{PEAK:.0f} GFLOPS 算力峰值',
        fontsize=50, ha='left', va='bottom', color='#555')
# 带宽标注
ax.text(0.007, 2.5, f'{BW:.0f} GB/s 带宽',
        fontsize=50, ha='left', color='#555', rotation=42)

# ── 图例 ─────────────────────────────────────────────────────────────────
handles = [
    Line2D([0],[0], color='#111', lw=2.8, label='Roofline 理论上限'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='#555', ms=32, lw=0, label='理论可达点（ON Roofline）'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#555',
           ms=32, lw=0, label='实测工作点'),
    mpatches.Patch(color='#2878B5', label='骨干卷积（P3/P4/P5）'),
    mpatches.Patch(color='#E87722', label='EFEM 投影 / 预测头'),
    mpatches.Patch(color='#B71C1C', label='EFEM gather'),
]
leg = ax.legend(handles=handles, fontsize=50, loc='lower right',
                framealpha=0.96, edgecolor='#BDBDBD', fancybox=False)
leg.get_frame().set_linewidth(1.0)

fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.09)

out = Path(__file__).parent.parent.parent / 'images' / 'roofline.pdf'
plt.savefig(str(out), format='pdf', bbox_inches='tight')
print(f'已生成: {out}')
