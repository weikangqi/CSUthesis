"""Insert system overview figure into Chapter 1 论文组织结构 section."""

with open('f:/CSUthesis/content/content.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Line 118 (index 117): the paragraph text
# Line 119 (index 118): blank line
# Line 120 (index 119): \clearpage

print("L117:", repr(lines[116]))
print("L118:", repr(lines[117][:80]))
print("L119:", repr(lines[118]))
print("L120:", repr(lines[119]))

# Append figure reference to end of the paragraph (line 118, index 117)
para = lines[117].rstrip('\n')
# Remove trailing period if present, add figure reference, then period
if para.endswith('。'):
    para = para[:-1]
para += '，其中第三、四章的整体框架如图~\\ref{fig:system_overview}所示。\n'
lines[117] = para

# Insert figure environment after line 118 (index 117), before blank line
figure = (
    '\\begin{figure}[htbp]\n'
    '  \\centering\n'
    '  \\includegraphics[width=0.85\\textwidth]{images/OverallStructureDiagramOfTheSystem.png}\n'
    '  \\caption{系统整体结构示意图。第三章聚焦模型设计与训练，构建轻量化双目3D姿态检测网络；第四章面向端侧落地，开展硬件感知部署与量化压缩。}\n'
    '  \\label{fig:system_overview}\n'
    '\\end{figure}\n'
)

# Insert at index 118 (after the paragraph)
lines.insert(118, figure)

with open('f:/CSUthesis/content/content.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'\nDone. Total lines: {len(lines)}')
print("New L118:", repr(lines[117][:80]))
print("New L119:", repr(lines[118][:60]))
