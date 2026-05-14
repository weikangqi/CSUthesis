"""
骁龙 8 Gen3 CPU Roofline 图
输出：images/roofline.pdf
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np

# 设置字体：中文宋体，英文和数字Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['font.family'] = ['Times New Roman', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 50
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'

PEAK  = 100.0
BW    = 40.0
RIDGE = PEAK / BW   # 2.5

MODULES = [
    ('EFEM gather',   0.04,  '#B71C1C', 'X',  0.53),
    ('EFEM 投影',     11.3,  '#E87722', 'D', 44.0),
    ('预测头',        15.7,  '#F4B942', 'P', 50.0),
    ('骨干 P5',      280.0,  '#5B9BD5', 'o', 72.0),
    ('骨干 P4',      784.0,  '#2878B5', 's', 80.0),
    ('骨干 P3',      923.0,  '#1B4F8A', '^', 83.0),
]

fig, ax = plt.subplots(figsize=(30.0, 20.0))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

xi = np.logspace(-2.5, 4.0, 2000)
yi = np.minimum(PEAK, BW * xi)
ax.plot(xi, yi, color='#111', lw=3.0, zorder=5)

ax.axvline(x=RIDGE, color='#9E9E9E', ls='--', lw=1.2, alpha=0.8, zorder=4)
ax.axvspan(0.003, RIDGE, alpha=0.05, color='#F44336', zorder=1)
ax.axvspan(RIDGE, 12000, alpha=0.05, color='#1565C0', zorder=1)

ax.text(0.0043, 1100, '访存密集区', fontsize=50, color='#C62828',
        ha='left', style='italic', va='top')
ax.text(3.5, 1100, '计算密集区', fontsize=50, color='#0D47A1',
        ha='left', style='italic', va='top')

ax.plot(RIDGE, PEAK, 'k.', ms=14, zorder=9)
ax.annotate(
    '屋脊点 ({:.1f}, {:.0f})\n'.format(RIDGE, PEAK) +
    r'$= P_{\rm peak}/B_{\rm eff}$',
    xy=(RIDGE, PEAK),
    xytext=(0.35, 50),
    fontsize=50, ha='left', va='top', color='#212121', zorder=10,
    arrowprops=dict(arrowstyle='->', lw=1.4, color='#555',
                    connectionstyle='arc3,rad=0.25'),
    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc',
              alpha=0.9, lw=0.8),
)

for _, ai, color, mk, meas in MODULES:
    y_th = min(PEAK, BW * ai)
    ax.scatter(ai, y_th,  color='none', marker=mk, s=600, zorder=7,
               edgecolors=color, linewidths=3.0)
    ax.scatter(ai, meas,  color=color,  marker=mk, s=500, zorder=8,
               edgecolors='white', linewidths=1.5)
    if abs(y_th - meas) / y_th > 0.05:
        ax.plot([ai, ai], [meas, y_th], color=color, lw=1.2,
                ls=':', alpha=0.6, zorder=4)

def ann(pt_xy, txt_xy, label, eff, color, ha, va, rad=0.0):
    cs = 'arc3,rad={}'.format(rad)
    ax.annotate('{}  {}'.format(label, eff),
        xy=pt_xy, xytext=txt_xy,
        fontsize=50, color=color, ha=ha, va=va, zorder=11,
        arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                        connectionstyle=cs, shrinkB=6),
    )

ann((0.04, 0.53),  (0.004, 0.8),
    'EFEM gather (视差相关)', '33%带宽',
    '#B71C1C', 'left', 'bottom', rad=0.0)

ann((15.7, 50.0),  (0.004, 18),
    '预测头 (1×1 Conv)', '50%',
    '#F4B942', 'left', 'bottom', rad=-0.15)

ann((11.3, 44.0),  (0.004, 110),
    'EFEM 投影 (1×1 Conv)', '44%',
    '#E87722', 'left', 'bottom', rad=0.12)

ann((280, 72.0),   (50, 62),
    '骨干 P5 (stride=32)', '72%',
    '#5B9BD5', 'left', 'top', rad=0.18)

ann((784, 80.0),   (6500, 32),
    '骨干 P4 (stride=16)', '80%',
    '#2878B5', 'right', 'top', rad=-0.20)

ann((923, 83.0),   (6500, 310),
    '骨干 P3 (stride=8)', '83%',
    '#1B4F8A', 'right', 'top', rad=-0.15)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.003, 8000)
ax.set_ylim(0.18, 1200)

ax.set_xlabel('算术强度  (FLOP/byte)', fontsize=50, labelpad=12)
ax.set_ylabel('性能  (GFLOPS)', fontsize=50, labelpad=12)
ax.set_title('骁龙 8 Gen3 CPU  Roofline 模型  (FP16, 4 线程, batch = 1)',
             fontsize=50, fontweight='bold', pad=14)

ax.grid(True, which='major', ls='--', alpha=0.22, lw=0.9, color='#777')
ax.grid(True, which='minor', ls=':',  alpha=0.10, lw=0.6, color='#aaa')

for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor('#333')
    sp.set_linewidth(1.1)
ax.tick_params(which='both', direction='in', top=True, right=True,
               length=6, width=1.0, labelsize=45, pad=12)

# 自定义 log 轴刻度标签，使用 ASCII 字符
def log_format(x, pos):
    if x == 0:
        return ''
    exp = int(np.log10(x))
    if exp == 0:
        return '1'
    return r'$10^{%d}$' % exp

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_format))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_format))

ax.text(4.5, PEAK * 1.06, '{:.0f} GFLOPS 算力峰值'.format(PEAK),
        fontsize=50, ha='left', va='bottom', color='#555')
ax.text(0.007, 2.5, '{:.0f} GB/s 带宽'.format(BW),
        fontsize=50, ha='left', color='#555', rotation=42)

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

fig.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.22)

out = Path(__file__).parent.parent.parent / 'images' / 'roofline.pdf'
plt.savefig(str(out), format='pdf', bbox_inches='tight')
print('已生成: {}'.format(out))
