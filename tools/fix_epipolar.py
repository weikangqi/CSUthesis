with open('f:/CSUthesis/content/content.tex', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Change 1: update figure caption (index 377, file line 378)
lines[377] = (
    '  \\caption{双目对极几何示意图。$C_l$、$C_r$为左右相机光心，$\\mathbf{X}$为空间三维点；'
    '$l_l$、$l_r$为对应极线，$e_l$、$e_r$为对极点；三点$C_l$、$C_r$、$\\mathbf{X}$所确定的平面为'
    '对极平面，其与成像平面的交线即为对应极线。}\n'
)

# Change 2: replace the long symbol-explanation paragraph (index 381, file line 382)
lines[381] = (
    '图中，$\\mathbf{x}_l=[u_l,v_l,1]^\\top$与$\\mathbf{x}_r=[u_r,v_r,1]^\\top$分别为'
    '$\\mathbf{X}$在左右成像平面上的像素齐次坐标，通过各自相机内参矩阵$\\mathbf{K}$与'
    '归一化相机坐标关联（$\\mathbf{x}\\sim\\mathbf{K}\\tilde{\\mathbf{x}}$）。\n'
)

# Change 3: replace F description + rectification paragraph (index 388, file line 389)
new_f_block = (
    '其中$\\mathbf{F}$为基础矩阵（Fundamental Matrix）。'
    '对于已标定的双目系统，两相机间的相对旋转$\\mathbf{R}$与平移$\\mathbf{t}$已知，'
    '可先构造本质矩阵（Essential Matrix）：\n'
    '\\begin{equation}\n'
    '\\mathbf{E} = [\\mathbf{t}]_\\times \\mathbf{R}\n'
    '\\label{eq:essential_matrix}\n'
    '\\end{equation}\n'
    '其中$[\\mathbf{t}]_\\times$为$\\mathbf{t}$的反对称矩阵。'
    '基础矩阵则由本质矩阵与左右相机内参联合给出：\n'
    '\\begin{equation}\n'
    '\\mathbf{F} = \\mathbf{K}_r^{-\\top}\\,\\mathbf{E}\\,\\mathbf{K}_l^{-1}\n'
    '\\label{eq:fundamental_matrix}\n'
    '\\end{equation}\n'
    '上式表明，$\\mathbf{F}$完全由标定参数$\\{\\mathbf{K}_l,\\mathbf{K}_r,\\mathbf{R},\\mathbf{t}\\}$决定。'
    '工程实现中通常对左右图像进行\\textbf{立体校正}（Stereo Rectification）：'
    '将倾斜的极线变换为水平扫描线，使校正后对应点满足$v_l = v_r$，'
    '跨视图匹配从二维搜索退化为一维水平扫描，'
    '视差$d = u_l - u_r$成为唯一待估量，自然引出下一节的三角测量深度恢复。\n'
)
lines[388] = new_f_block

with open('f:/CSUthesis/content/content.tex', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Done. Total lines:', len(lines))
