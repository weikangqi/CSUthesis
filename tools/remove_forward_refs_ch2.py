"""Remove all forward references to Chapter 3/4 from Chapter 2."""

with open('f:/CSUthesis/content/content.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Print current content of target lines for verification
print("=== BEFORE ===")
print(f"L239: {repr(lines[238][:100])}")
print(f"L410: {repr(lines[409][:100])}")
print(f"L487: {repr(lines[486][:100])}")
print(f"L510: {repr(lines[509][:100])}")
print(f"L512: {repr(lines[511][:100])}")
print(f"L551: {repr(lines[550][:100])}")

# --- Fix 1: L239 — remove "共同支撑第三章核心模块的设计" ---
lines[238] = (
    '综合来看，CNN擅长捕捉局部纹理与多尺度空间特征，适合作为高效的视觉骨干网络；'
    'Transformer的注意力机制则擅长建立跨位置、跨视图的全局关联。'
    '在双目视觉任务中，两类结构相辅相成：CNN负责从左右视图中提取层次化视觉特征，'
    '交叉注意力在此基础上实现跨视图特征对齐与极线约束下的信息融合，'
    '两者相辅相成，构成双目视觉理解的常见组合框架。\n'
)

# --- Fix 2: L410 — delete entire forward-reference paragraph ---
# Line 410 (index 409): the "在本文的研究框架中..." paragraph
# Also check if L411 is blank and remove it too
print(f"\nL408: {repr(lines[407])}")
print(f"L409: {repr(lines[408])}")
print(f"L410: {repr(lines[409][:80])}")
print(f"L411: {repr(lines[410])}")

lines[409] = ''   # delete the forward-ref paragraph

# --- Fix 3: L487 — rewrite stereo vision summary paragraph ---
lines[486] = (
    '综上，双目立体视觉的成像模型、极线几何约束与三角测量深度恢复共同构成双目3D视觉的理论基础：'
    '极线约束使跨视图特征搜索从二维退化为一维，显著降低匹配复杂度；'
    '三角测量则提供了将视差转化为可量化深度的几何桥梁，'
    '是双目系统相对于单目方法的核心优势所在。\n'
)

# --- Fix 4: L510 — remove EFEM forward reference tail ---
# Current: ...将二维搜索退化为一维扫描——这也是第三章极线特征增强模块（EFEM）的设计出发点。
lines[509] = (
    '端侧系统通常难以承受高分辨率、全视差范围的代价体与3D卷积正则化，'
    '更可行的替代思路是在低分辨率特征上构建紧凑的跨视图交互，'
    '以分组相关（Group-wise Correlation）等轻量算子代替全通道代价体，从而降低访存压力；'
    '结合立体校正先验，可进一步将信息聚合限制在极线方向，将二维搜索退化为一维扫描，'
    '是兼顾效率与几何精度的可行方案。\n'
)

# --- Fix 5: L512 — delete thesis-specific DWT sentence at end ---
# Current: ...本文采用热力图引导的2D关节定位与可微加权三角化（DWT）相结合的方式，...
lines[511] = (
    '最后，在预测头与输出表征方面，基于热力图的方法通常定位精度较高，但需要上采样与峰值解析；'
    '直接坐标回归计算图更简洁，但对特征表达与训练稳定性要求更高。'
    '在轻量化框架中通常优先选择算子友好、计算路径短的输出形式，'
    '并通过共享特征、减少分支与控制输出分辨率来进一步降低时延与内存占用。\n'
)

# --- Fix 6: L551 — rewrite 本章小结 ---
lines[550] = (
    '本章系统梳理了本文研究所依赖的核心理论与技术基础。'
    '在深度学习基础方面，介绍了卷积神经网络的层次特征提取机制与Transformer的多头注意力及交叉注意力框架，'
    '两者分别适用于高效视觉特征提取与跨位置、跨视图的全局关联建模。'
    '在人体姿态估计理论方面，归纳了2D/3D姿态估计的主流范式、热力图与坐标回归两类表示方法，'
    '以及MPJPE、PA-MPJPE等核心评价指标，明确了双目3D姿态估计的任务定义与评测体系。'
    '在双目立体视觉原理方面，阐述了针孔成像模型、极线几何约束与三角测量深度恢复方法，'
    '这些几何原理是实现高效跨视图特征匹配与三维坐标恢复的理论依据。'
    '在轻量化网络设计方面，总结了深度可分离卷积、倒残差结构等高效骨干设计范式，'
    '以及量化、剪枝与知识蒸馏等模型压缩策略，'
    '为面向资源受限设备的视觉算法设计提供了系统性的理论参照。\n'
)

with open('f:/CSUthesis/content/content.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'\nDone. Total lines: {len(lines)}')

# Verify no more forward refs in Ch2 range (lines 1-553)
ch2 = ''.join(lines[:553])
keywords = ['第三章', '第四章', 'EFEM', 'DWT', 'CSPDarknet', '混合精度', '量化压缩']
for kw in keywords:
    count = ch2.count(kw)
    if count:
        print(f'WARNING: "{kw}" still appears {count} time(s) in Ch2!')
    else:
        print(f'OK: "{kw}" not in Ch2')
