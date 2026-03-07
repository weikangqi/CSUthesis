import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FIG_W, FIG_H = 13.6, 6.2

C_BG = '#F7F6F2'
C_TEXT = '#1F2933'
C_BORDER = '#D7D2C8'
C_ARROW = '#7A7A7A'

C_ANALYSIS = '#2E5B8A'
C_MODEL = '#4C7F6F'
C_DEPLOY = '#A05A2C'
C_EVAL = '#8A3E4B'
C_NOTE = '#EBD9B7'


def add_box(ax, x, y, w, h, color, title, body, title_fs=11, body_fs=8.8):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle='round,pad=0.08,rounding_size=0.16',
        linewidth=1.1,
        edgecolor='white',
        facecolor=color,
        alpha=0.96,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y + 0.32, title, ha='center', va='center',
            fontsize=title_fs, color='white', fontweight='bold', zorder=4)
    ax.text(x, y - 0.1, body, ha='center', va='center',
            fontsize=body_fs, color='#EDF2F7', linespacing=1.35, zorder=4)


def add_note(ax, x, y, text):
    ax.text(
        x, y, text,
        ha='center', va='center', fontsize=8.6, color=C_TEXT,
        bbox=dict(boxstyle='round,pad=0.35', facecolor=C_NOTE, edgecolor=C_BORDER, alpha=0.95),
        zorder=5,
    )


def add_arrow(ax, x1, y1, x2, y2, text=''):
    ax.annotate(
        '',
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', lw=1.8, color=C_ARROW),
        zorder=2,
    )
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.22, text,
                ha='center', va='bottom', fontsize=8.4, color='#555555')


fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=170)
fig.patch.set_facecolor('white')
ax.set_facecolor(C_BG)
ax.set_xlim(0, 13.6)
ax.set_ylim(0, 6.2)
ax.axis('off')

xs = [1.45, 3.95, 6.45, 8.95, 11.45]
y_top = 4.25
w = 2.0
h = 1.55

add_box(ax, xs[0], y_top, w, h, C_ANALYSIS,
        '延迟拆解',
        '真机逐模块计时\n定位骨干与 EFEM 热点')
add_box(ax, xs[1], y_top, w, h, C_ANALYSIS,
        'Roofline 分析',
        '判别算力/带宽瓶颈\n解释 INT8 收益来源')
add_box(ax, xs[2], y_top, w, h, C_MODEL,
        '敏感性分析',
        '识别低风险权重层\n筛出高风险激活/求解器')
add_box(ax, xs[3], y_top, w, h, C_DEPLOY,
        '混合精度设计',
        '主体卷积 INT8\n关键激活与 DWT 保留 FP32')
add_box(ax, xs[4], y_top, w, h, C_DEPLOY,
        'PTQ 部署',
        'weight-only PTQ\nPyTorch → ONNX → MNN')

for left, right, label in [
    (xs[0] + w / 2, xs[1] - w / 2, '瓶颈类型'),
    (xs[1] + w / 2, xs[2] - w / 2, '精度风险'),
    (xs[2] + w / 2, xs[3] - w / 2, '保护约束'),
    (xs[3] + w / 2, xs[4] - w / 2, '实现配置'),
]:
    add_arrow(ax, left, y_top, right, y_top, label)

eval_y = 1.9
eval_w = 3.6
eval_h = 1.45
add_box(ax, 6.8, eval_y, eval_w, eval_h, C_EVAL,
        '实验验证',
        '精度--效率对比  |  保护层消融  |  多人时延稳定性', title_fs=11.5, body_fs=9.2)

add_arrow(ax, xs[4], y_top - h / 2, 8.3, eval_y + eval_h / 2, '端到端验证')
add_arrow(ax, 5.3, eval_y + eval_h / 2, 3.95, y_top - h / 2, '结果反证设计合理性')

add_note(ax, 1.55, 5.45, '4.2.1\n热点定位')
add_note(ax, 3.95, 5.45, '4.2.2\n硬件机理')
add_note(ax, 6.45, 5.45, '4.2.3\n精度风险')
add_note(ax, 10.2, 5.45, '4.3\n方案与实施')
add_note(ax, 6.8, 0.72, '4.4\n对比实验、消融实验与多人场景稳定性分析')

ax.text(6.8, 5.95, '第四章硬件感知部署与量化压缩的整体方法流程',
        ha='center', va='center', fontsize=13, fontweight='bold', color=C_TEXT)
ax.text(6.8, 5.68,
        '从硬件瓶颈定位出发，经由机理解释与精度风险筛查，形成面向目标平台的混合精度量化方案',
        ha='center', va='center', fontsize=9.2, color='#4A5568')

plt.tight_layout(pad=0.6)
out = 'f:/CSUthesis/images/Chapter4Flow.png'
plt.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
print(f'saved: {out}')
