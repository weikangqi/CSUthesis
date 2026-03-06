"""Expand 论文组织结构 section to one paragraph per chapter."""

with open('f:/CSUthesis/content/content.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# L117 (index 116): \subsection{论文组织结构}
# L118 (index 117): single-paragraph text
# L119-124 (index 118-123): figure environment
# Replace only L118 (the paragraph), keep figure unchanged

new_para = (
    '第一章为绪论。介绍本文的研究背景与现实意义，'
    '分析双目3D人体姿态估计在端侧场景中面临的主要挑战，'
    '综述国内外相关研究现状，最后阐述本文的研究内容、主要贡献与论文组织结构。\n'
    '\n'
    '第二章为相关技术与理论介绍。系统梳理本文所依赖的核心理论基础，'
    '包括卷积神经网络与Transformer注意力机制等深度学习基础，'
    '人体姿态估计的任务范式与评价体系，双目立体视觉的成像模型与极线几何约束，'
    '以及面向端侧部署的轻量化结构设计与模型压缩策略。\n'
    '\n'
    '第三章为基于双目相机的轻量化3D人体姿态检测网络设计。'
    '提出一种轻量化非对称双目学习框架：以"左重右轻"的非对称骨干网络消除左右对称冗余，'
    '以极线特征增强模块（EFEM）将跨视图匹配限制在一维极线搜索空间，'
    '以可微加权三角化求解器（DWT）实现端到端的三维关节坐标恢复，'
    '并通过对比实验与消融实验验证各模块的有效性，'
    '如图~\\ref{fig:system_overview}所示。\n'
    '\n'
    '第四章为硬件感知部署与量化压缩。'
    '面向Snapdragon 8 Gen3端侧平台，通过延迟拆解与Roofline模型定位推理瓶颈，'
    '借助量化敏感性分析制定分层混合精度量化方案（骨干INT8、敏感层FP16/FP32），'
    '经后训练量化（PTQ）流程输出可部署的MNN模型，'
    '在真机上验证精度保持与推理加速效果。\n'
    '\n'
    '第五章为总结与展望。对全文研究工作进行系统总结，'
    '分析现有方法的局限性，并对后续研究方向提出展望。\n'
)

lines[117] = new_para

with open('f:/CSUthesis/content/content.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'Done. Total lines: {len(lines)}')
# Quick check
print('L118:', repr(lines[117][:60]))
print('L124:', repr(lines[123][:40]))  # should still be figure
