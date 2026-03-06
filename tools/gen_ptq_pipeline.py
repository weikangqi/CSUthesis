"""
PTQ 部署流程图：PyTorch → ONNX → mnnconvert → model_mixed.mnn → 真机推理
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(13.5, 5.0), dpi=160)
ax.set_xlim(0, 13.5)
ax.set_ylim(0, 5.0)
ax.axis('off')

C_SRC  = '#2E86AB'
C_ONNX = '#445566'
C_CONV = '#1A5276'
C_OUT  = '#A23B72'
C_INF  = '#C73E1D'
C_NOTE = '#E8A838'
C_TXT  = 'white'
ALPHA  = 0.92

def box(ax, cx, cy, w, h, color, title, sub='', tfs=10, sfs=8.5):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle='round,pad=0.1', facecolor=color,
                       edgecolor='white', linewidth=1.0, alpha=ALPHA, zorder=3)
    ax.add_patch(r)
    dy = 0.18 if sub else 0
    ax.text(cx, cy + dy, title, ha='center', va='center',
            fontsize=tfs, color=C_TXT, fontweight='bold', zorder=4)
    if sub:
        for i, line in enumerate(sub.split('\n')):
            ax.text(cx, cy - 0.18 - i*0.22, line, ha='center', va='top',
                    fontsize=sfs, color='#DDDDDD', zorder=4)

def arr(ax, x1, x2, y):
    ax.annotate('', xy=(x2-0.05, y), xytext=(x1+0.05, y),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5), zorder=2)

def note_below(ax, cx, top_y, text, color=C_NOTE):
    ax.text(cx, top_y, text, ha='center', va='top', fontsize=8.2,
            color='#222222',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=color,
                      edgecolor='#BBBBBB', alpha=0.88))

# 主流程 y = 3.5
Y = 3.5
xs = [1.05, 3.1, 5.6, 8.1, 10.6]   # box centres
ws = [1.7,  1.7, 2.4,  1.8,  2.1]
hs = [1.2,  1.2, 1.55, 1.2,  1.2]

box(ax, xs[0], Y, ws[0], hs[0], C_SRC,  'PyTorch 模型',   'FP16 权重\nStereoPoseNet')
box(ax, xs[1], Y, ws[1], hs[1], C_ONNX, 'ONNX 图',        '固定输入\n1×3×480×640')
box(ax, xs[2], Y, ws[2], hs[2], C_CONV, 'mnnconvert 图优化',
    'Conv-BN 融合\nINT8 per-channel 量化\n量化豁免层配置', tfs=9.5, sfs=8)
box(ax, xs[3], Y, ws[3], hs[3], C_OUT,  'model_mixed.mnn','混合精度\n15.2 MB')
box(ax, xs[4], Y, ws[4], hs[4], C_INF,  '真机推理验证',   'Xiaomi 14\nSnapdragon 8 Gen3')

# 箭头 + 标签
gaps = [(xs[0]+ws[0]/2, xs[1]-ws[1]/2, 'torch.onnx.export\n(opset 17)'),
        (xs[1]+ws[1]/2, xs[2]-ws[2]/2, '--weightQuantBits 8'),
        (xs[2]+ws[2]/2, xs[3]-ws[3]/2, ''),
        (xs[3]+ws[3]/2, xs[4]-ws[4]/2, 'ADB 推送')]
for (x1, x2, lbl) in gaps:
    arr(ax, x1, x2, Y)
    if lbl:
        ax.text((x1+x2)/2, Y+0.72, lbl, ha='center', va='bottom',
                fontsize=8, color='#444444')

# DWT 旁路
dwtY = 1.0
ax.plot([xs[0], xs[0]], [Y - hs[0]/2, dwtY+0.18], color='#999', lw=1.2, zorder=1)
ax.annotate('', xy=(xs[4], dwtY+0.18), xytext=(xs[0], dwtY+0.18),
            arrowprops=dict(arrowstyle='->', color='#999', lw=1.2), zorder=1)
ax.plot([xs[4], xs[4]], [dwtY+0.18, Y - hs[4]/2], color='#999', lw=1.2, zorder=1)
ax.text(6.75, dwtY, 'DWT 旁路：torch.linalg.eigh（非标准 ONNX 算子）→ 独立 Python 脚本，FP32 CPU 执行',
        ha='center', va='center', fontsize=8.8, color='#444444',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0',
                  edgecolor='#BBBBBB', alpha=0.95))

# 注释气泡（放在主流程框下方）
note_below(ax, xs[1], Y - hs[1]/2 - 0.12,
           'ONNX 算子集验证\n输入输出形状确认')
note_below(ax, xs[2], Y - hs[2]/2 - 0.12,
           'scale_c = max|W_c| / 127\n豁免: EFEM attention / 热力图 / DWT')
note_below(ax, xs[3], Y - hs[3]/2 - 0.12,
           'MNN Python API\n形状验证 + 单次前向')
note_below(ax, xs[4], Y - hs[4]/2 - 0.12,
           '预热10次 + 计时100次\ntaskset 绑定大核')

ax.set_title('混合精度量化 PTQ 部署流程（PyTorch → ONNX → MNN）',
             fontsize=12, pad=8)

plt.tight_layout(pad=0.5)
out = 'f:/CSUthesis/images/PTQPipeline.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
print(f'saved: {out}')
