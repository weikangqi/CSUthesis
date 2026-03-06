with open('f:/CSUthesis/content/content.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Locate section boundaries
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '\\subsection{轻量化网络设计}' in line:
        start_idx = i
    if start_idx is not None and '\\subsection{本章小结}' in line:
        end_idx = i
        break

print(f"Section 2.4: lines {start_idx+1}–{end_idx} (replacing {end_idx - start_idx} lines)")

new_section = (
    r'\subsection{轻量化网络设计}' + '\n'
    r'\label{subsec:lightweight_design}' + '\n'
    r'面向移动端或嵌入式设备的视觉算法受到算力、内存带宽与功耗的严格约束。'
    r'对于双目3D人体姿态估计而言，网络需在参数量、计算量（FLOPs）、峰值内存与精度-效率折中四个维度协同权衡，'
    r'同时避免传统立体匹配中高维代价体（Cost Volume）带来的显存开销。'
    r'轻量化设计通常遵循"减少高维张量、优先2D算子、控制通道数与分辨率"的总体原则。'
    r'本小节从高效骨干结构设计、模型压缩与端侧部署工程优化三个层次依次介绍在边缘端实现高效推理的常用方法。' + '\n'
    '\n'
    r'\subsubsection{高效骨干结构设计}' + '\n'
    r'从结构层面来看，面向端侧部署的轻量化通常需要在骨干特征提取、跨尺度与跨视图融合以及预测头设计三个方面协同优化。' + '\n'
    '\n'
    r'首先，在骨干特征提取方面，深度可分离卷积（Depthwise Separable Convolution）是降低计算量的典型手段。'
    r'对于输入特征尺寸为$H\times W$、通道数为$C_{in}$，输出通道数为$C_{out}$、卷积核大小为$k\times k$的标准卷积，'
    r'其计算复杂度近似为：' + '\n'
    r'\begin{equation}' + '\n'
    r'\text{Cost}_{\text{conv}} \propto H W \cdot C_{in} C_{out} \cdot k^2' + '\n'
    r'\label{eq:conv_cost}' + '\n'
    r'\end{equation}' + '\n'
    r'深度可分离卷积将其分解为逐通道卷积（Depthwise）与$1\times 1$点卷积（Pointwise），复杂度近似为：' + '\n'
    r'\begin{equation}' + '\n'
    r'\text{Cost}_{\text{dw}} \propto H W \cdot C_{in}\cdot k^2,\quad' + '\n'
    r'\text{Cost}_{\text{pw}} \propto H W \cdot C_{in} C_{out}' + '\n'
    r'\label{eq:dw_pw_cost}' + '\n'
    r'\end{equation}' + '\n'
    r'当$k>1$且$C_{out}$较大时，式\ref{eq:dw_pw_cost}相较式\ref{eq:conv_cost}可显著降低计算量。'
    r'进一步地，倒残差结构（Inverted Residual）与线性瓶颈（Linear Bottleneck）通过"先扩展后压缩"的通道设计，'
    r'在保证表达能力的同时减少高成本卷积层的使用；'
    r'组卷积（Group Convolution）与通道洗牌（Channel Shuffle）则在降低计算量的同时增强通道间信息交互。'
    r'这些模块共同构成面向端侧的高效特征提取基础。' + '\n'
    '\n'
    r'其次，在跨尺度与跨视图融合方面，人体姿态估计与双目匹配均依赖多尺度信息与跨视图一致性。'
    r'轻量化设计中应重点控制通道宽度与融合路径数量，避免引入过多上/下采样分支与冗余卷积。'
    r'双目结构天然存在"左右分支"带来的参数与算力倍增风险，'
    r'实践中常采用左右视图共享骨干参数的非对称或孪生结构以降低参数量并增强特征一致性。'
    r'端侧系统通常难以承受高分辨率、全视差范围的代价体与3D卷积正则化，'
    r'更可行的思路是在较低分辨率特征上建立粗对应关系，通过逐级细化或迭代更新恢复细节；'
    r'结合立体校正先验时，可将信息聚合限制在极线方向，在保证有效搜索范围的同时控制计算开销。' + '\n'
    '\n'
    r'最后，在预测头与输出表征方面，基于热力图的方法通常定位精度较高，但需要上采样与峰值解析；'
    r'直接坐标回归计算图更简洁，但对特征表达与训练稳定性要求更高。'
    r'在轻量化框架中通常优先选择算子友好、计算路径短的输出形式，'
    r'并通过共享特征、减少分支与控制输出分辨率来进一步降低时延与内存占用。' + '\n'
    '\n'
    r'\subsubsection{模型压缩}' + '\n'
    r'在高效结构的基础上，模型压缩进一步降低存储与计算开销，常用手段包括剪枝、知识蒸馏与量化，'
    r'三者分别从网络稀疏化、精度迁移与数值表示三个角度实现压缩，可单独使用也可组合部署。' + '\n'
    '\n'
    r'剪枝方面，结构化剪枝通过移除冗余通道或注意力头，在保证可部署性的前提下减少计算量。'
    r'由式\ref{eq:conv_cost}可知，通道剪枝带来的理论加速可写为：' + '\n'
    r'\begin{equation}' + '\n'
    r'\frac{\text{Cost}^\prime}{\text{Cost}} \approx \frac{C^\prime_{in} C^\prime_{out}}{C_{in} C_{out}}' + '\n'
    r'\label{eq:pruning_speedup}' + '\n'
    r'\end{equation}' + '\n'
    r'当输入与输出通道同时缩减时可获得更显著的降幅。'
    r'非结构化稀疏在端侧往往难以获得稳定加速，因此更常采用结构化方案并配合编译器优化。' + '\n'
    '\n'
    r'知识蒸馏通过教师网络为学生网络提供"软目标"与中间特征约束，在不增加推理开销的前提下提升小模型精度'
    r'~\cite{hinton2015distilling,romero2015fitnets,zagoruyko2017attention}。'
    r'以温度蒸馏为例，典型损失可写为：' + '\n'
    r'\begin{equation}' + '\n'
    r'\mathcal{L} = (1-\lambda)\mathcal{L}_{task} + \lambda T^2\mathrm{KL}\!\left(\mathrm{softmax}\!\left(\frac{\mathbf{z}_t}{T}\right)\ \middle\|\ \mathrm{softmax}\!\left(\frac{\mathbf{z}_s}{T}\right)\right)' + '\n'
    r'\label{eq:kd_loss}' + '\n'
    r'\end{equation}' + '\n'
    r'其中$\mathbf{z}_t,\mathbf{z}_s$为教师与学生的logits，$T$为温度系数，$\lambda$为权衡系数。'
    r'对于姿态回归任务，也可采用特征蒸馏或分布蒸馏等方式提升鲁棒性。' + '\n'
    '\n'
    r'量化在端侧场景中尤为重要：其不仅降低存储与带宽开销，还能显著提升NPU/DSP等硬件的算子吞吐。'
    r'以$b$比特整数量化为例，常用的仿射量化可写为：' + '\n'
    r'\begin{equation}' + '\n'
    r'q = \mathrm{clip}\!\left(\left\lfloor \frac{x}{s} \right\rceil + z,\ q_{\min},\ q_{\max}\right),\quad' + '\n'
    r'\hat{x} = s \left(q - z\right)' + '\n'
    r'\label{eq:affine_quant}' + '\n'
    r'\end{equation}' + '\n'
    r'其中$x$为浮点值，$q$为量化后的整数，$\hat{x}$为反量化近似值；'
    r'$s>0$为缩放因子，$z$为零点（zero-point）；$q_{\min}, q_{\max}$由位宽确定（如INT8时为$[-128,127]$）。'
    r'对于给定动态范围$[x_{\min},x_{\max}]$，缩放因子选取为：' + '\n'
    r'\begin{equation}' + '\n'
    r's = \frac{x_{\max}-x_{\min}}{q_{\max}-q_{\min}},\quad' + '\n'
    r'z = \left\lfloor q_{\min} - \frac{x_{\min}}{s}\right\rceil' + '\n'
    r'\label{eq:quant_param}' + '\n'
    r'\end{equation}' + '\n'
    r'在实践中常采用逐通道（per-channel）权重量化与量化感知训练（QAT）减小精度损失，'
    r'并通过校准（calibration）合理选择动态范围以避免过多截断。'
    r'量化还可显著减少模型存储：$b$比特量化后存储量约为FP32的$b/32$倍（即$S_b = \frac{b}{8}N_p$字节）。' + '\n'
    '\n'
    r'\subsubsection{端侧部署工程优化}' + '\n'
    r'结构轻量化与模型压缩确定了网络的理论效率上限，而端侧实际推理性能还取决于算子与硬件的匹配程度。'
    r'端侧推理的瓶颈往往不仅来自算术计算量，还来自内存带宽与访存模式：'
    r'频繁的特征拼接与大体积张量读写会显著抬高时延与能耗，'
    r'双目任务中代价体易导致内存峰值过高，需优先以2D算子替代高维操作。'
    r'工程层面应结合算子融合（Operator Fusion）、张量布局优化与内存复用等手段，'
    r'尽量避免大规模3D卷积、超大矩阵乘等端侧不友好算子，才能将理论压缩收益转化为稳定的端侧加速。'
    r'综合来看，轻量化网络设计需要在结构设计、压缩策略与部署工程三个层面协同优化，'
    r'才能在资源受限设备上实现可用的双目3D人体姿态估计性能。' + '\n'
    '\n'
    '\n'
)

lines[start_idx:end_idx] = [new_section]

with open('f:/CSUthesis/content/content.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Done. New total lines: {len(lines)}')
