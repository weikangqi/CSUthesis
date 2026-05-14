import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体为宋体，英文字体为Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun', 'Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

labels = [
    u'\u5de6\u5206\u652f\u9aa8\u5e72',
    u'\u53f3\u5206\u652f\u9aa8\u5e72',
    u'\u6781\u7ebf\u7279\u5f81\u589e\u5f3a',
    u'\u70ed\u529b\u56fe\u5934\u4e0e\u7f6e\u4fe1\u5ea6\u5934',
    u'DWT \u4e09\u89d2\u5316\u6c42\u89e3\u5668',
]
sizes   = [76.7, 19.4, 10.0, 7.0, 1.1]
colors  = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8F71']
explode = (0.04, 0.02, 0.03, 0.02, 0.06)

fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=180)

wedges, texts, autotexts = ax.pie(
    sizes,
    labels=None,
    autopct=lambda p: '{:.1f}%'.format(p) if p > 2 else '',
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
    at.set_fontfamily('Times New Roman')

legend_labels = [u'{} (CSPDarknet-{})  {:.1f} ms'.format(l, m, s) for l, m, s in zip(labels, ['M', 'N', 'EFEM', '', ''], sizes)]
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

ax.set_title(u'\u5404\u6a21\u5757\u7aef\u5230\u7aef\u5ef6\u8fdf\u5360\u6bd4（Xiaomi 14 / MNN / FP16 / 4\u7ebf\u7a0b\uff0c\u5355\u4eba\u573a\u666f\uff0c\u5171 116.0 ms）',
             fontsize=13.5, pad=14)

plt.tight_layout()
out = 'f:/CSUthesis/images/ModuleDelayRatioFanChart.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
print('saved: {}'.format(out))
