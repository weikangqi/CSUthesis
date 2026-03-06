with open('f:/CSUthesis/content/content.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines are 1-indexed in editor; Python list is 0-indexed
# Line 350: \paragraph{人体关节点定义与标准}
# Line 351: Para 1 (generic definition) -- REMOVE
# Line 352: blank                        -- REMOVE
# Line 353: Para 2 (dataset differences) -- REMOVE
# Line 354: blank                        -- REMOVE
# Line 355: Para 3 (root-relative)       -- KEEP but trim

print("Line 350:", repr(lines[349]))
print("Line 351:", repr(lines[350][:80]))
print("Line 352:", repr(lines[351]))
print("Line 353:", repr(lines[352][:80]))
print("Line 354:", repr(lines[353]))
print("Line 355:", repr(lines[354][:80]))
print("Line 356:", repr(lines[355]))
print("Line 364:", repr(lines[363][:80]))

# Replace lines 351-354 (indices 350-353) with a single concise paragraph
new_para = (
    '不同数据集在关节点数量与位置上存在差异：COCO定义17个关节点\\cite{lin2014microsoft}，'
    'MPII定义16个\\cite{andriluka2014mpii}，Human3.6M常用17个核心点\\cite{ionescu2014human36m}。'
    '在3D姿态表示中，通常以骨盆（Pelvis）为根节点，所有关节点坐标采用根相对（root-relative）'
    '方式表示，消除全局位置变化的影响，使模型专注于姿态本身。'
    '图~\\ref{F.csu_keypoints}展示了典型的人体骨架关节点布局，'
    '三套标准的对照关系见表~\\ref{tab:pose_comparison}。\n'
)

# Replace lines 351-355 (indices 350-354) with new_para
lines[350:355] = [new_para]

# Find and simplify the transition sentence before the table
# Search for the line containing "为了更清晰地展示不同标准间的差异"
for i, line in enumerate(lines):
    if '为了更清晰地展示不同标准间的差异' in line:
        print(f"Found transition at line {i+1}: {repr(line[:80])}")
        # Remove this line entirely (it's now redundant)
        lines[i] = ''
        break

with open('f:/CSUthesis/content/content.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done. Total lines:', len(lines))
