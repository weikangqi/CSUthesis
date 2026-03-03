"""
轻量化非对称双目三维人体姿态估计模型
根据论文第三章架构描述实现

模型组成：
  - CSPDarknet 非对称双目骨干（左分支全量 M，右分支轻量 N）
  - EFEM 极线特征增强模块（F.pad + unfold，ONNX 兼容）
  - 热力图头与置信度头
  - DWT 可微加权三角化求解器（CPU FP32，不进 ONNX）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# 基础算子
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Conv + BN + SiLU"""
    def __init__(self, in_c: int, out_c: int, k: int = 1, s: int = 1, p: Optional[int] = None):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """CSP Bottleneck（shortcut 可选）"""
    def __init__(self, c: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        h = int(c * e)
        self.cv1 = Conv(c, h, 3, 1)
        self.cv2 = Conv(h, c, 3, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Cross Stage Partial with 2-conv，含 n 个 Bottleneck"""
    def __init__(self, in_c: int, out_c: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.c  = out_c // 2
        self.cv1 = Conv(in_c, out_c, 1, 1)
        self.cv2 = Conv((n + 2) * self.c, out_c, 1, 1)
        self.m   = nn.ModuleList(
            Bottleneck(self.c, shortcut=shortcut) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


# ---------------------------------------------------------------------------
# 非对称双目骨干 CSPDarknet
# ---------------------------------------------------------------------------

class CSPDarknet(nn.Module):
    """
    YOLO 风格的 CSPDarknet 骨干。

    参数：
      width_multiple  : 通道缩放因子（左分支 1.0，右分支 0.5）
      depth_multiple  : 深度缩放因子（左分支 1.0，右分支 0.34）

    输出：
      三个尺度的特征图字典 {'s8', 's16', 's32'}
      左分支通道：256 / 512 / 1024
      右分支通道：128 / 256 / 512
    """
    # 基础通道数（对应 width=1.0）
    BASE_CHANNELS = [64, 128, 256, 512, 1024]
    # 各阶段 C2f 重复次数（对应 depth=1.0）
    BASE_DEPTHS   = [3, 6, 6, 3]

    def __init__(self, width_multiple: float = 1.0, depth_multiple: float = 1.0):
        super().__init__()
        w = width_multiple
        d = depth_multiple
        ch = [max(1, round(c * w)) for c in self.BASE_CHANNELS]
        nd = [max(1, round(n * d)) for n in self.BASE_DEPTHS]

        # stem：stride 2，输出 ch[0]
        self.stem = Conv(3, ch[0], 3, 2)

        # stage 1：stride 4，输出 ch[1]
        self.stage1 = nn.Sequential(
            Conv(ch[0], ch[1], 3, 2),
            C2f(ch[1], ch[1], nd[0], shortcut=True),
        )

        # stage 2：stride 8，输出 ch[2]（s8 特征）
        self.stage2 = nn.Sequential(
            Conv(ch[1], ch[2], 3, 2),
            C2f(ch[2], ch[2], nd[1], shortcut=True),
        )

        # stage 3：stride 16，输出 ch[3]（s16 特征）
        self.stage3 = nn.Sequential(
            Conv(ch[2], ch[3], 3, 2),
            C2f(ch[3], ch[3], nd[2], shortcut=True),
        )

        # stage 4：stride 32，输出 ch[4]（s32 特征）
        self.stage4 = nn.Sequential(
            Conv(ch[3], ch[4], 3, 2),
            C2f(ch[4], ch[4], nd[3], shortcut=True),
        )

        # 各尺度输出通道数（供外部查询）
        self.out_channels = {'s8': ch[2], 's16': ch[3], 's32': ch[4]}

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        s8  = self.stage2(x)
        s16 = self.stage3(s8)
        s32 = self.stage4(s16)
        return s8, s16, s32


def build_darknet_m() -> CSPDarknet:
    """左分支：全量 CSPDarknet-M（width=1.0, depth=1.0）"""
    return CSPDarknet(width_multiple=1.0, depth_multiple=1.0)


def build_darknet_n() -> CSPDarknet:
    """右分支：轻量 CSPDarknet-N（width=0.5, depth=0.34）"""
    return CSPDarknet(width_multiple=0.5, depth_multiple=0.34)


# ---------------------------------------------------------------------------
# 极线特征增强模块（EFEM）
# ---------------------------------------------------------------------------

class EFEM(nn.Module):
    """
    Epipolar Feature Enhancement Module。

    以右分支特征为查询（Q），左分支特征为键值（K/V），沿极线视差窗口
    计算交叉注意力，将左分支几何信息注入右分支表征。

    实现采用 F.pad + unfold 替代动态 gather，完全兼容 ONNX opset 17。

    参数：
      in_c     : 输入通道数（左右分支主尺度特征通道，需相同或对齐后传入）
      proj_c   : QKV 投影通道数（注意力内部维度）
      max_disp : 特征图级最大视差（默认 24，对应原图约 384px @ stride 16）
    """

    def __init__(self, in_c: int, proj_c: int = 128, max_disp: int = 24):
        super().__init__()
        self.proj_c   = proj_c
        self.max_disp = max_disp
        self.scale    = math.sqrt(proj_c)

        # QKV 投影（1×1 Conv，不改变空间尺寸）
        self.phi_q = nn.Sequential(Conv(in_c, proj_c, 1, 1), nn.Conv2d(proj_c, proj_c, 1))
        self.phi_k = nn.Sequential(Conv(in_c, proj_c, 1, 1), nn.Conv2d(proj_c, proj_c, 1))
        self.phi_v = nn.Sequential(Conv(in_c, proj_c, 1, 1), nn.Conv2d(proj_c, proj_c, 1))

        # 融合投影 ψ：将 q 与 g 拼接后映射回 in_c
        self.psi = nn.Conv2d(proj_c * 2, in_c, 1)

    def forward(self, feat_l: torch.Tensor, feat_r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_l: 左分支主尺度特征 [B, in_c, H, W]
            feat_r: 右分支主尺度特征 [B, in_c, H, W]
        Returns:
            q_hat:  增强后右分支特征  [B, in_c, H, W]
        """
        B, C, H, W = feat_r.shape
        D = self.max_disp

        q = self.phi_q(feat_r)   # [B, C', H, W]
        k = self.phi_k(feat_l)   # [B, C', H, W]
        v = self.phi_v(feat_l)   # [B, C', H, W]

        # --- 极线 gather：k/v 在 u+d 位置的特征（d=0..D-1）---
        # 在宽度方向右侧 pad D-1 列（边界填零），使位置 u+D-1 有效
        # 注意：避免 unfold 算子（ONNX 不支持动态 size 的 unfold），
        #       改用显式切片 + stack，循环在 trace 时展开，生成静态 ONNX 图。
        k_pad = F.pad(k, (0, D - 1, 0, 0))          # [B, C', H, W+D-1]
        v_pad = F.pad(v, (0, D - 1, 0, 0))

        # 逐视差位移切片，trace 时展开为 D 个独立 Slice 节点
        k_slices = [k_pad[:, :, :, d:d + W] for d in range(D)]   # D × [B, C', H, W]
        v_slices = [v_pad[:, :, :, d:d + W] for d in range(D)]
        k_stack = torch.stack(k_slices, dim=4)       # [B, C', H, W, D]
        v_stack = torch.stack(v_slices, dim=4)       # [B, C', H, W, D]

        # --- 注意力分数 [B, H, W, D]（FP16 精度敏感层） ---
        scores = (q.unsqueeze(-1) * k_stack).sum(dim=1) / self.scale  # [B, H, W, D]
        alpha  = scores.softmax(dim=-1)                                 # [B, H, W, D]

        # --- 值聚合 [B, C', H, W] ---
        g = (alpha.unsqueeze(1) * v_stack).sum(dim=-1)   # [B, C', H, W]

        # --- 融合输出 ---
        q_hat = self.psi(torch.cat([q, g], dim=1))         # [B, in_c, H, W]
        return q_hat


# ---------------------------------------------------------------------------
# 预测头
# ---------------------------------------------------------------------------

class HeatmapHead(nn.Module):
    """
    关键点热力图预测头。
    输出 J 个热力图，空间尺寸与输入相同（stride 16 特征图上运行）。
    """
    def __init__(self, in_c: int, num_joints: int = 17):
        super().__init__()
        self.head = nn.Sequential(
            Conv(in_c, in_c // 2, 3, 1),
            nn.Conv2d(in_c // 2, num_joints, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)   # [B, J, H, W]


class ConfidenceHead(nn.Module):
    """
    关节置信度预测头。
    输出 J 个置信度 logit（经 softplus 后作为三角化权重）。
    """
    def __init__(self, in_c: int, num_joints: int = 17):
        super().__init__()
        self.head = nn.Sequential(
            Conv(in_c, in_c // 4, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_c // 4, num_joints),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)   # [B, J]


# ---------------------------------------------------------------------------
# 主模型（ONNX 导出范围）
# ---------------------------------------------------------------------------

class StereoPoseNet(nn.Module):
    """
    轻量化非对称双目三维人体姿态估计框架（ONNX 导出部分）。

    前向输出：热力图 + 置信度（共 4 个张量），DWT 求解器单独运行。

    输入分辨率：640×480（stride 16 特征图尺寸为 40×30）
    """

    NUM_JOINTS = 17

    def __init__(self):
        super().__init__()

        # 非对称双目骨干
        self.backbone_left  = build_darknet_m()   # 左分支全量
        self.backbone_right = build_darknet_n()   # 右分支轻量

        # 通道对齐投影（右分支 s16 通道对齐到左分支 s16 通道，供 EFEM 使用）
        c_left_s16  = self.backbone_left.out_channels['s16']    # 512
        c_right_s16 = self.backbone_right.out_channels['s16']   # 256
        self.align_proj = nn.Conv2d(c_right_s16, c_left_s16, 1) if c_right_s16 != c_left_s16 else nn.Identity()

        # EFEM（以左分支 s16 通道数为基准）
        self.efem = EFEM(in_c=c_left_s16, proj_c=128, max_disp=24)

        # 预测头（作用在 stride 16 特征图上）
        self.heatmap_head_l = HeatmapHead(c_left_s16,  self.NUM_JOINTS)
        self.heatmap_head_r = HeatmapHead(c_left_s16,  self.NUM_JOINTS)
        self.conf_head_l    = ConfidenceHead(c_left_s16, self.NUM_JOINTS)
        self.conf_head_r    = ConfidenceHead(c_left_s16, self.NUM_JOINTS)

    def forward(
        self,
        img_l: torch.Tensor,   # [B, 3, H, W]
        img_r: torch.Tensor,   # [B, 3, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            heatmap_l: [B, J, H/16, W/16]
            heatmap_r: [B, J, H/16, W/16]
            conf_l:    [B, J]
            conf_r:    [B, J]
        """
        # 骨干特征提取
        _, s16_l, _ = self.backbone_left(img_l)    # [B, 512, H/16, W/16]
        _, s16_r, _ = self.backbone_right(img_r)   # [B, 256, H/16, W/16]

        # 通道对齐，使右分支特征维度与左分支一致
        s16_r_aligned = self.align_proj(s16_r)     # [B, 512, H/16, W/16]

        # EFEM：将左分支几何信息注入右分支
        s16_r_enhanced = self.efem(s16_l, s16_r_aligned)  # [B, 512, H/16, W/16]

        # 预测头
        heatmap_l = self.heatmap_head_l(s16_l)
        heatmap_r = self.heatmap_head_r(s16_r_enhanced)
        conf_l    = self.conf_head_l(s16_l)
        conf_r    = self.conf_head_r(s16_r_enhanced)

        return heatmap_l, heatmap_r, conf_l, conf_r


# ---------------------------------------------------------------------------
# DWT 可微加权三角化求解器（CPU FP32，不进 ONNX）
# ---------------------------------------------------------------------------

class DWT(nn.Module):
    """
    Differentiable Weighted Triangulation 求解器。

    推理时独立运行在 CPU FP32 上，不包含在 ONNX 导出范围内。
    输入热力图和置信度来自 StereoPoseNet 的输出。

    参考论文算法 1（第三章 3.3.3 节）。
    """

    @staticmethod
    def soft_argmax_2d(
        heatmap: torch.Tensor,   # [B, J, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        软 argmax，返回 u（水平）和 v（垂直）坐标期望。
        Returns: u [B, J], v [B, J]，坐标单位为像素。
        """
        B, J, H, W = heatmap.shape
        # 2D softmax
        p = heatmap.view(B, J, -1).softmax(dim=-1).view(B, J, H, W)

        # 坐标网格
        xs = torch.arange(W, dtype=heatmap.dtype, device=heatmap.device)
        ys = torch.arange(H, dtype=heatmap.dtype, device=heatmap.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')  # [H, W]

        u = (p * grid_x).sum(dim=(-2, -1))  # [B, J]
        v = (p * grid_y).sum(dim=(-2, -1))  # [B, J]
        return u, v

    @staticmethod
    def triangulate(
        u_l: torch.Tensor, v_l: torch.Tensor,   # [J]
        u_r: torch.Tensor, v_r: torch.Tensor,   # [J]
        P_l: torch.Tensor,                       # [3, 4]
        P_r: torch.Tensor,                       # [3, 4]
        w_l: torch.Tensor,                       # [J]
        w_r: torch.Tensor,                       # [J]
        stride: int = 16,
    ) -> torch.Tensor:
        """
        加权最小二乘三角化，求每个关节的三维坐标。

        Args:
            stride: 热力图相对于原图的下采样倍数（将坐标还原到原图像素）
        Returns:
            X3d: [J, 3]，左相机坐标系下的三维关节坐标
        """
        # 热力图坐标还原到原图像素坐标
        u_l = u_l * stride;  v_l = v_l * stride
        u_r = u_r * stride;  v_r = v_r * stride

        J = u_l.shape[0]
        p1l, p2l, p3l = P_l[0], P_l[1], P_l[2]  # 各行
        p1r, p2r, p3r = P_r[0], P_r[1], P_r[2]

        X3d = []
        for j in range(J):
            # 左视图两行约束
            A_lj = torch.stack([
                u_l[j] * p3l - p1l,
                v_l[j] * p3l - p2l,
            ])  # [2, 4]
            # 右视图两行约束
            A_rj = torch.stack([
                u_r[j] * p3r - p1r,
                v_r[j] * p3r - p2r,
            ])  # [2, 4]
            # 堆叠与加权
            A_j = torch.cat([A_lj, A_rj], dim=0)                       # [4, 4]
            W_j = torch.diag(torch.stack([
                w_l[j].sqrt(), w_l[j].sqrt(),
                w_r[j].sqrt(), w_r[j].sqrt(),
            ]))
            B_j = W_j @ A_j                                             # [4, 4]
            M_j = B_j.T @ B_j                                           # [4, 4]

            # 最小特征值对应的特征向量（齐次三维点）
            _, vecs = torch.linalg.eigh(M_j.double())
            X_h = vecs[:, 0].float()                                    # [4]
            X3d.append(X_h[:3] / (X_h[3] + 1e-8))

        return torch.stack(X3d)   # [J, 3]

    def forward(
        self,
        heatmap_l: torch.Tensor,  # [B, J, H, W]
        heatmap_r: torch.Tensor,  # [B, J, H, W]
        conf_l:    torch.Tensor,  # [B, J]
        conf_r:    torch.Tensor,  # [B, J]
        P_l:       torch.Tensor,  # [3, 4]
        P_r:       torch.Tensor,  # [3, 4]
        stride:    int = 16,
    ) -> torch.Tensor:
        """
        Returns:
            joints_3d: [B, J, 3]，左相机坐标系下的三维关节坐标（毫米）
        """
        B = heatmap_l.shape[0]
        u_l, v_l = self.soft_argmax_2d(heatmap_l.float())   # [B, J]
        u_r, v_r = self.soft_argmax_2d(heatmap_r.float())
        w_l = F.softplus(conf_l.float()) + 1e-6             # [B, J]
        w_r = F.softplus(conf_r.float()) + 1e-6

        results = []
        for b in range(B):
            X = self.triangulate(
                u_l[b], v_l[b], u_r[b], v_r[b],
                P_l, P_r, w_l[b], w_r[b], stride,
            )
            results.append(X)
        return torch.stack(results)   # [B, J, 3]


# ---------------------------------------------------------------------------
# 快速测试
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    device = torch.device('cpu')

    model = StereoPoseNet().to(device).eval()
    dwt   = DWT().to(device)

    # 统计参数量
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'StereoPoseNet 参数量: {params:.2f} M')

    # 随机双目输入
    img_l = torch.randn(1, 3, 480, 640, device=device)
    img_r = torch.randn(1, 3, 480, 640, device=device)

    with torch.no_grad():
        hm_l, hm_r, cf_l, cf_r = model(img_l, img_r)

    print(f'heatmap_l: {hm_l.shape}')   # [1, 17, 30, 40]
    print(f'heatmap_r: {hm_r.shape}')
    print(f'conf_l:    {cf_l.shape}')   # [1, 17]
    print(f'conf_r:    {cf_r.shape}')

    # DWT 测试（随机投影矩阵）
    P_l = torch.eye(3, 4)
    P_r = torch.eye(3, 4); P_r[0, 3] = -100.0   # 基线 100mm
    with torch.no_grad():
        joints = dwt(hm_l, hm_r, cf_l, cf_r, P_l, P_r, stride=16)
    print(f'joints_3d: {joints.shape}')   # [1, 17, 3]
    print('模型自检通过')
