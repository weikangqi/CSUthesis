"""
生成第四章两张柱状图（论文美化版，四边框闭合）：
  - 图4-1: ModuleDelayRatioFanChart.png  各模块端到端延迟
  - 图4-2: CSPDarknetStageLatencyPie.png  双分支各阶段延迟
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    'font.family':        'SimHei',
    'axes.unicode_minus': False,
    'font.size':          15,
    'axes.titleweight':   'bold',
})

OUT = r'f:\CSUthesis\images'


def close_frame(ax, color='#273142', lw=1.0):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)
        spine.set_zorder(20)


def add_panel_tag(ax, tag):
    ax.text(
        0.015, 0.975, tag,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=12,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#173B73', edgecolor='none')
    )


# ═══════════════════════════════════════════════════════════════════════════
# 图4-1  各模块端到端延迟  水平条形图
# ═══════════════════════════════════════════════════════════════════════════
labels   = ['左分支骨干\n(CSPDarknet-M)',
            '右分支骨干\n(CSPDarknet-N)',
            'EFEM\n（极线特征增强）',
            '热力图头 &\n置信度头',
            'DWT\n（三角化求解）']
values   = [76.7, 19.4, 10.0, 7.0, 1.1]
percents = [66.1, 16.7,  8.6,  6.0, 0.9]

# 蓝色主色 + 渐变层次
palette = ['#173B73', '#255FA8', '#4B88C7', '#86B3DD', '#C9DDF0']

fig, ax = plt.subplots(figsize=(12.2, 6.9))
fig.patch.set_facecolor('white')
ax.set_facecolor('#F7F9FC')

y_pos = np.arange(len(labels))
bars = ax.barh(y_pos, values, height=0.58,
               color=palette, edgecolor='white', linewidth=1.4,
               zorder=2, clip_on=True)

ax.axvspan(0, 80, color='#EAF1F8', alpha=0.35, zorder=0)
ax.grid(axis='x', color='#D7DFEA', linewidth=0.9, linestyle='--', zorder=0)
ax.set_axisbelow(True)

close_frame(ax)

ax.text(
    92.0, 3.55, '骨干合计占比 82.8%',
    ha='right', va='center',
    fontsize=15.2, color='#173B73', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.30', facecolor='#E8F0FA', edgecolor='#B7C8DE')
)

# 条形内：百分比徽标；条形右端：延迟值
for bar, pct, val in zip(bars, percents, values):
    w = bar.get_width()
    y_c = bar.get_y() + bar.get_height() / 2
    if w > 8:
        x_pct = max(w - 2.2, w * 0.72)
        ax.text(
            x_pct, y_c, f'{pct:.1f}%',
            va='center', ha='right', fontsize=15.0,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#12345E', edgecolor='none', alpha=0.92)
        )
    ax.text(
        w + 1.0, y_c, f'{val:.1f} ms',
        va='center', ha='left', fontsize=16.0, color='#1F2430', fontweight='bold'
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=17.2, linespacing=1.35)
ax.invert_yaxis()
ax.set_xlabel('延迟 (ms)', fontsize=18.2, labelpad=8)
ax.set_xlim(0, 95)
ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.tick_params(which='both', direction='in', top=False, right=False,
               length=4, width=0.9, color='#555', labelsize=16.8)
ax.tick_params(which='minor', length=2.5)

ax.text(
    0.985, 0.025, '合计：116.0 ms',
    transform=ax.transAxes,
    ha='right', va='bottom', fontsize=15.0,
    color='#4B5563',
    bbox=dict(boxstyle='round,pad=0.22', facecolor='#F2F4F7', edgecolor='none')
)

fig.tight_layout(pad=1.5)
out1 = fr'{OUT}\ModuleDelayRatioFanChart.png'
fig.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out1}')


# ═══════════════════════════════════════════════════════════════════════════
# 图4-2  CSPDarknet 双分支各阶段延迟  分组竖向柱状图
# ═══════════════════════════════════════════════════════════════════════════
stages_label = ['Stem', 'P3\n(stride=8)', 'P4\n(stride=16)', 'P5\n(stride=32)']
ratio_stage  = [0.08, 0.37, 0.33, 0.22]

left_total, right_total = 76.7, 19.4
left_ms  = [left_total  * r for r in ratio_stage]
right_ms = [right_total * r for r in ratio_stage]

C_LEFT  = '#173B73'
C_RIGHT = '#5B9BD5'

x     = np.arange(len(stages_label))
width = 0.34

fig, ax = plt.subplots(figsize=(9.4, 6.0))
fig.patch.set_facecolor('white')
ax.set_facecolor('#F7F9FC')

ax.axvspan(0.55, 2.45, color='#FDECC8', alpha=0.45, zorder=0)
ax.text(1.5, 36.3, '主加速区：P3 / P4', ha='center', va='center',
        fontsize=14.4, color='#9A5B00', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFF4D6', edgecolor='#F2CE7A'))

b1 = ax.bar(x - width/2, left_ms,  width,
            label=f'左分支 CSPDarknet-M（{left_total} ms）',
            color=C_LEFT,  edgecolor='white', linewidth=1.2,
            zorder=2, clip_on=True)
b2 = ax.bar(x + width/2, right_ms, width,
            label=f'右分支 CSPDarknet-N（{right_total} ms）',
            color=C_RIGHT, edgecolor='white', linewidth=1.2,
            zorder=2, clip_on=True)

ax.grid(axis='y', color='#D7DFEA', linewidth=0.9, linestyle='--', zorder=0)
ax.set_axisbelow(True)
close_frame(ax)

# 柱顶数值
for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.35,
            f'{h:.1f}', ha='center', va='bottom',
            fontsize=14.6, color=C_LEFT, fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.35,
            f'{h:.1f}', ha='center', va='bottom',
            fontsize=14.6, color='#1a5fa0')

# P3+P4 双向箭头标注
y_ann = 31.8
x1 = x[1] - width/2 - 0.06
x2 = x[2] + width/2 + 0.06
ax.annotate('', xy=(x2, y_ann), xytext=(x1, y_ann),
            arrowprops=dict(arrowstyle='<->', color='#C0392B',
                            lw=1.6, mutation_scale=16))
ax.text((x1 + x2) / 2, y_ann + 0.35, 'P3 + P4 ≈ 70% 延迟',
        ha='center', va='bottom', fontsize=14.8,
        color='#C0392B', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(stages_label, fontsize=17.0, linespacing=1.3)
ax.set_ylabel('延迟 (ms)', fontsize=18.0, labelpad=8)
ax.set_ylim(0, 38)
ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(2.5))
ax.tick_params(which='both', direction='in', top=False, right=False,
               length=4, width=0.9, color='#555', labelsize=16.4)
ax.tick_params(which='minor', length=2.5)

legend = ax.legend(
    fontsize=14.2, loc='lower center', ncol=2,
    bbox_to_anchor=(0.5, 1.08),
    frameon=False, handlelength=1.8, columnspacing=1.6
)

fig.tight_layout(pad=1.5, rect=[0, 0, 1, 0.93])
out2 = fr'{OUT}\CSPDarknetStageLatencyPie.png'
fig.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out2}')
