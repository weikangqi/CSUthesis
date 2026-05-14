"""
生成第四章两张柱状图（论文美化版，四边框闭合）：
  - 图4-1: ModuleDelayRatioFanChart.png  各模块端到端延迟
  - 图4-2: CSPDarknetStageLatencyPie.png  双分支各阶段延迟
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import matplotlib.font_manager as fm

# 设置中文字体为宋体，英文字体为Times New Roman
matplotlib.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun', 'Microsoft YaHei'],
    'axes.unicode_minus': False,
    'font.size': 15,
    'axes.titleweight': 'bold',
})

# 创建字体属性
simsun_font = fm.FontProperties(family='SimSun', size=15)
times_font = fm.FontProperties(family='Times New Roman', size=15)

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
labels   = [u'\u5de6\u5206\u652f\u9aa8\u5e72\n(CSPDarknet-M)',
            u'\u53f3\u5206\u652f\u9aa8\u5e72\n(CSPDarknet-N)',
            u'EFEM\n(\u6781\u7ebf\u7279\u5f81\u589e\u5f3a)',
            u'\u70ed\u529b\u56fe\u5934 &\n\u7f6e\u4fe1\u5ea6\u5934',
            u'DWT\n(\u4e09\u89d2\u5316\u6c42\u89e3)']
values   = [76.7, 19.4, 10.0, 7.0, 1.1]
percents = [66.1, 16.7,  8.6,  6.0, 0.9]

# 蓝色主色 + 渐变层次
palette = ['#173B73', '#255FA8', '#4B88C7', '#86B3DD', '#C9DDF0']

fig, ax = plt.subplots(figsize=(13.8, 7.8))
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
    92.0, 3.55, u'\u9aa8\u5e72\u5408\u8ba1\u5360\u6bd4 82.8%',
    ha='right', va='center',
    fontsize=18.4, color='#173B73', fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.30', facecolor='#E8F0FA', edgecolor='#B7C8DE')
)

# 条形内：百分比徽标；条形右端：延迟值
for bar, pct, val in zip(bars, percents, values):
    w = bar.get_width()
    y_c = bar.get_y() + bar.get_height() / 2
    if w > 8:
        x_pct = max(w - 2.2, w * 0.72)
        ax.text(
            x_pct, y_c, '{:.1f}%'.format(pct),
            va='center', ha='right', fontsize=18.2, fontfamily='Times New Roman',
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#12345E', edgecolor='none', alpha=0.92)
        )
    ax.text(
        w + 1.0, y_c, '{:.1f} ms'.format(val),
        va='center', ha='left', fontsize=19.6, color='#1F2430', fontweight='bold', fontfamily='Times New Roman'
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=21.6, linespacing=1.38)
ax.invert_yaxis()
ax.set_xlabel(u'\u5ef6\u8fdf (ms)', fontsize=21.2, labelpad=8)
ax.set_xlim(0, 95)
ax.set_xticks([0, 20, 40, 60, 80])
ax.set_xticklabels(['0', '20', '40', '60', '80'], fontfamily='Times New Roman', fontsize=20.0)
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
ax.tick_params(which='both', direction='in', top=False, right=False,
               length=4, width=0.9, color='#555')
ax.tick_params(which='minor', length=2.5)

ax.text(
    0.985, 0.025, u'\u5408\u8ba1\uff1a116.0 ms',
    transform=ax.transAxes,
    ha='right', va='bottom', fontsize=17.6,
    color='#4B5563',
    bbox=dict(boxstyle='round,pad=0.22', facecolor='#F2F4F7', edgecolor='none')
)

fig.tight_layout(pad=1.5)
out1 = fr'{OUT}\ModuleDelayRatioFanChart.png'
fig.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved: {}'.format(out1))


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
ax.text(1.5, 36.3, u'\u4e3b\u52a0\u901f\u533a\uff1aP3 / P4', ha='center', va='center',
        fontsize=14.4, color='#9A5B00', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#FFF4D6', edgecolor='#F2CE7A'))

b1 = ax.bar(x - width/2, left_ms,  width,
            label=u'\u5de6\u5206\u652f CSPDarknet-M（{} ms）'.format(left_total),
            color=C_LEFT,  edgecolor='white', linewidth=1.2,
            zorder=2, clip_on=True)
b2 = ax.bar(x + width/2, right_ms, width,
            label=u'\u53f3\u5206\u652f CSPDarknet-N（{} ms）'.format(right_total),
            color=C_RIGHT, edgecolor='white', linewidth=1.2,
            zorder=2, clip_on=True)

ax.grid(axis='y', color='#D7DFEA', linewidth=0.9, linestyle='--', zorder=0)
ax.set_axisbelow(True)
close_frame(ax)

# 柱顶数值
for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.35,
            '{:.1f}'.format(h), ha='center', va='bottom', fontfamily='Times New Roman',
            fontsize=14.6, color=C_LEFT, fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.35,
            '{:.1f}'.format(h), ha='center', va='bottom', fontfamily='Times New Roman',
            fontsize=14.6, color='#1a5fa0')

# P3+P4 双向箭头标注
y_ann = 31.8
x1 = x[1] - width/2 - 0.06
x2 = x[2] + width/2 + 0.06
ax.annotate('', xy=(x2, y_ann), xytext=(x1, y_ann),
            arrowprops=dict(arrowstyle='<->', color='#C0392B',
                            lw=1.6, mutation_scale=16))
ax.text((x1 + x2) / 2, y_ann + 0.35, 'P3 + P4 ≈ 70% {}'.format(u'\u5ef6\u8fdf'),
        ha='center', va='bottom', fontsize=14.8,
        color='#C0392B', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(stages_label, fontsize=17.0, linespacing=1.3, fontfamily='Times New Roman')
ax.set_ylabel(u'\u5ef6\u8fdf (ms)', fontsize=18.0, labelpad=8)
ax.set_ylim(0, 38)
ax.set_yticks([0, 5, 10, 15, 20, 25, 30, 35])
ax.set_yticklabels(['0', '5', '10', '15', '20', '25', '30', '35'], fontfamily='Times New Roman', fontsize=16.4)
ax.yaxis.set_minor_locator(mticker.MultipleLocator(2.5))
ax.tick_params(which='both', direction='in', top=False, right=False,
               length=4, width=0.9, color='#555')
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
print('Saved: {}'.format(out2))
