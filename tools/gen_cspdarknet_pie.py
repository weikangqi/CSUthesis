"""
CSPDarknet 双分支各阶段延迟占比扇形图
各阶段延迟基于 FLOPs 比例从实测总延迟（M=76.7ms, N=19.4ms）推算
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体为宋体，英文字体为Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ------- CSPDarknet-M (left, width=1.0, depth=1.0, total=76.7ms) -------
m_labels = [u'Stem\n(\u4e0b\u91c7\u6837\u81f31/4)', u'P3 \u9636\u6bb5\n(stride=8, C=256)', u'P4 \u9636\u6bb5\n(stride=16, C=512)', u'P5 \u9636\u6bb5\n(stride=32, C=1024)']
m_sizes  = [5.0, 24.5, 28.8, 18.4]   # ms, sum=76.7
m_colors = ['#90C2E7', '#2E86AB', '#1A5276', '#0D2B3E']
m_explode= (0.03, 0.03, 0.03, 0.03)

# ------- CSPDarknet-N (right, width=0.5, depth=0.34, total=19.4ms) ------
n_labels = [u'Stem\n(\u4e0b\u91c7\u6837\u81f31/4)', u'P3 \u9636\u6bb5\n(stride=8, C=128)', u'P4 \u9636\u6bb5\n(stride=16, C=256)', u'P5 \u9636\u6bb5\n(stride=32, C=512)']
n_sizes  = [1.3, 4.5, 7.2, 6.4]      # ms, sum=19.4
n_colors = ['#D7A0C8', '#A23B72', '#6B1E47', '#3B0B27']
n_explode= (0.03, 0.03, 0.03, 0.03)

fig, (ax_m, ax_n) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=180)

def draw_pie(ax, sizes, labels, colors, explode, total, title):
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=lambda p: '{:.1f}%'.format(p),
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
        at.set_fontfamily('Times New Roman')

    legend_labels = [u'{}  {:.1f} ms'.format(l.replace(chr(10), ' '), s) for l, s in zip(labels, sizes)]
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
         u'CSPDarknet-M（\u5de6\u5206\u652f\uff0cwidth=1.0\uff0c\u5171 76.7 ms）')
draw_pie(ax_n, n_sizes, n_labels, n_colors, n_explode, 19.4,
         u'CSPDarknet-N（\u53f3\u5206\u652f\uff0cwidth=0.5\uff0c\u5171 19.4 ms）')

fig.suptitle(u'CSPDarknet \u53cc\u5206\u652f\u5404\u9636\u6bb5\u5ef6\u8fdf\u5360\u6bd4（FLOPs \u6bd4\u4f8b\u4f30\u7b97\uff0cXiaomi 14 / MNN / FP16）',
             fontsize=13, y=1.01)

plt.tight_layout()
out = 'f:/CSUthesis/images/CSPDarknetStageLatencyPie.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print('saved: {}'.format(out))
