"""Fix Chapter 2 intro paragraph (L123) — remove forward refs to Ch3/Ch4."""

with open('f:/CSUthesis/content/content.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("L123:", repr(lines[122][:120]))

lines[122] = (
    '本章围绕双目轻量化三维人体姿态估计的研究主线，系统梳理相关核心理论与关键技术，'
    '为方法设计与实验分析提供知识铺垫。全章内容分为四个递进部分：'
    '第一，介绍卷积神经网络与Transformer注意力机制等深度学习基础，'
    '阐述其在视觉特征提取与跨视图信息融合中的作用原理；'
    '第二，梳理人体姿态估计的任务范式、关键点表示方法与评价指标体系，'
    '明确双目3D姿态估计的输出定义与评测标准；'
    '第三，阐述双目立体视觉的成像模型与极线几何约束，'
    '介绍三角测量深度恢复的基本原理；'
    '第四，总结面向端侧部署的轻量化结构设计原则与模型压缩策略，'
    '概述在资源受限设备上实现高效推理的主要技术路径。'
    '四部分由浅入深、相互支撑，共同构成本文方法设计与部署优化的知识基础。\n'
)

with open('f:/CSUthesis/content/content.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Done.")

# Verify
ch2 = ''.join(lines[:553])
keywords = ['第三章', '第四章', 'EFEM', 'DWT', 'CSPDarknet', '混合精度', '量化压缩']
for kw in keywords:
    count = ch2.count(kw)
    status = 'WARNING' if count else 'OK'
    print(f'{status}: "{kw}" appears {count} time(s) in Ch2')
