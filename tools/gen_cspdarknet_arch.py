"""
CSPDarknet 非对称双分支结构示意图（精简版）
左分支 CSPDarknet-M (width=1.0)  vs  右分支 CSPDarknet-N (width=0.5)
纵向排列各阶段（Stem → P3 → P4 → P5），标注通道数与下采样倍数
右侧画出 CSP Block 内部结构（单 Bottleneck 表示）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(13, 7.5), dpi=160,
                          gridspec_kw={'width_ratios': [1.6, 1]})
ax_main, ax_csp = axes

# ─────────────────────────────────────────────────────────
# 左侧：双分支骨干总体结构
# ─────────────────────────────────────────────────────────
ax = ax_main
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

C_M   = '#2E86AB'
C_N   = '#A23B72'
C_IN  = '#555555'
C_TXT = 'white'
ALPHA = 0.88

def box(ax, x, y, w, h, color, label, sublabel='', fontsize=11, subfontsize=9.5):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.08',
                          facecolor=color, edgecolor='white',
                          linewidth=1.0, alpha=ALPHA, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha='center', va='center', fontsize=fontsize,
            color=C_TXT, fontweight='bold', zorder=4)
    if sublabel:
        ax.text(x, y - 0.28, sublabel, ha='center', va='center',
                fontsize=subfontsize, color='#DDDDDD', zorder=4)

def arrow_down(ax, x, y_top, y_bot, color='#888888'):
    ax.annotate('', xy=(x, y_bot + 0.05), xytext=(x, y_top - 0.05),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.4), zorder=2)

# 输入图像
box(ax, 5.0, 9.3, 3.2, 0.65, C_IN, '输入双目图像  640×480×3', fontsize=10.5)

# 分叉线
ax.annotate('', xy=(2.4, 8.35), xytext=(5.0, 8.97),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1.3), zorder=2)
ax.annotate('', xy=(7.6, 8.35), xytext=(5.0, 8.97),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1.3), zorder=2)

# 列标题
ax.text(2.4, 8.65, '左分支（CSPDarknet-M）', ha='center', va='bottom',
        fontsize=11, color=C_M, fontweight='bold')
ax.text(7.6, 8.65, '右分支（CSPDarknet-N）', ha='center', va='bottom',
        fontsize=11, color=C_N, fontweight='bold')
ax.text(2.4, 8.42, 'width=1.0, depth=1.0', ha='center', va='bottom',
        fontsize=9.5, color='#888888')
ax.text(7.6, 8.42, 'width=0.5, depth=0.34', ha='center', va='bottom',
        fontsize=9.5, color='#888888')

# 阶段配置：(主标签, M子标签, N子标签, y中心)
stages = [
    ('Stem  /4',    'Conv-BN-SiLU  128ch', 'Conv-BN-SiLU  64ch',  7.35),
    ('P3  /8  CSP', 'C=256  60×80',         'C=128  60×80',         5.9),
    ('P4  /16 CSP', 'C=512  30×40',         'C=256  30×40',         4.45),
    ('P5  /32 CSP', 'C=1024  15×20',        'C=512  15×20',         3.0),
]
prev_ym = 8.05
prev_yn = 8.05
for (lbl, sub_m, sub_n, yc) in stages:
    box(ax, 2.4, yc, 2.9, 1.0, C_M, lbl, sub_m, fontsize=10, subfontsize=9.5)
    box(ax, 7.6, yc, 2.9, 1.0, C_N, lbl, sub_n, fontsize=10, subfontsize=9.5)
    arrow_down(ax, 2.4, prev_ym, yc + 0.50, C_M)
    arrow_down(ax, 7.6, prev_yn, yc + 0.50, C_N)
    prev_ym = yc - 0.50
    prev_yn = yc - 0.50

# 输出特征
box(ax, 2.4, 2.0, 2.9, 0.65, C_M, '多尺度特征  {F^l_s}', fontsize=10)
box(ax, 7.6, 2.0, 2.9, 0.65, C_N, '多尺度特征  {F^r_s}', fontsize=10)
arrow_down(ax, 2.4, prev_ym, 2.33, C_M)
arrow_down(ax, 7.6, prev_yn, 2.33, C_N)

# 汇合 → EFEM
ax.annotate('', xy=(5.0, 1.35), xytext=(3.0, 1.68),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1.3))
ax.annotate('', xy=(5.0, 1.35), xytext=(7.0, 1.68),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1.3))
box(ax, 5.0, 1.0, 3.0, 0.62, '#F18F01', '极线特征增强（EFEM）', fontsize=10.5)

ax.set_title('非对称双目骨干网络整体结构（CSPDarknet-M / N）',
             fontsize=13, pad=8)

# ─────────────────────────────────────────────────────────
# 右侧：CSP Block 内部结构（精简：单 Bottleneck 框）
# ─────────────────────────────────────────────────────────
ax2 = ax_csp
ax2.set_xlim(0, 5)
ax2.set_ylim(0, 10)
ax2.axis('off')

C_CONV  = '#1A6B8A'
C_SPLIT = '#777777'
C_CONCAT= '#445566'
C_BOT   = '#1A6B8A'
C_OUT   = '#2E86AB'

def box2(ax, x, y, w, h, color, label, fontsize=10):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle='round,pad=0.07',
                          facecolor=color, edgecolor='white',
                          linewidth=0.8, alpha=0.9, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold', zorder=4)

def arr2(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2 + 0.05), xytext=(x, y1 - 0.05),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2), zorder=2)

# 输入
box2(ax2, 2.5, 9.2, 3.8, 0.6, C_IN, '输入特征  C×H×W')

# 1×1 压缩
arr2(ax2, 2.5, 8.9, 8.3)
box2(ax2, 2.5, 8.0, 3.8, 0.55, C_CONV, '1×1 Conv-BN-SiLU（压缩）')

# 分路
ax2.annotate('', xy=(1.3, 7.35), xytext=(2.5, 7.73),
             arrowprops=dict(arrowstyle='->', color='#888', lw=1.1))
ax2.annotate('', xy=(3.7, 7.35), xytext=(2.5, 7.73),
             arrowprops=dict(arrowstyle='->', color='#888', lw=1.1))
ax2.text(1.3, 7.55, '路径 A', ha='center', fontsize=9.5, color='#aaaaaa')
ax2.text(3.7, 7.55, '路径 B', ha='center', fontsize=9.5, color='#aaaaaa')

# 路径 A：直通
box2(ax2, 1.3, 7.0, 2.1, 0.55, C_SPLIT, 'Skip')

# 路径 B：单个 Bottleneck 框（代表 ×n 堆叠）
box2(ax2, 3.7, 7.0, 2.1, 0.55, C_BOT, 'Bottleneck ×n')
arr2(ax2, 3.7, 6.73, 6.13)
box2(ax2, 3.7, 5.85, 2.1, 0.55, C_BOT, '3×3 Conv-BN-SiLU')
arr2(ax2, 3.7, 5.58, 4.98)
box2(ax2, 3.7, 4.7, 2.1, 0.55, C_BOT, '1×1 Conv-BN-SiLU')

# 合并 → Concat
ax2.annotate('', xy=(2.5, 4.2), xytext=(1.3, 6.73),
             arrowprops=dict(arrowstyle='->', color='#888', lw=1.1))
ax2.annotate('', xy=(2.5, 4.2), xytext=(3.7, 4.43),
             arrowprops=dict(arrowstyle='->', color='#888', lw=1.1))
box2(ax2, 2.5, 3.9, 3.8, 0.55, C_CONCAT, 'Concat（通道拼接）')

# 1×1 融合
arr2(ax2, 2.5, 3.63, 3.03)
box2(ax2, 2.5, 2.75, 3.8, 0.55, C_CONV, '1×1 Conv-BN-SiLU（融合）')

# 输出
arr2(ax2, 2.5, 2.48, 1.88)
box2(ax2, 2.5, 1.6, 3.8, 0.55, C_OUT, '输出特征  C×H×W')

ax2.set_title('CSP Block 内部结构', fontsize=13, pad=8)
ax2.text(2.5, 0.75,
         'M: n=3/6/9（P3/P4/P5）\nN: n=1/2/3（P3/P4/P5）',
         ha='center', fontsize=9.5, color='#555555',
         bbox=dict(boxstyle='round', facecolor='#f5f5f5', edgecolor='#cccccc'))

plt.tight_layout(pad=1.5)
out = 'f:/CSUthesis/images/CSPDarknetArch.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'saved: {out}')
