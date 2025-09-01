import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2

# -- RGB backbone (WRN50-2 layer3) --
class RGBBackboneWRN50L3(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = wide_resnet50_2(weights='IMAGENET1K_V2' if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# -- PointNet (fallback) --
class PointNetBackbone(nn.Module):
    def __init__(self, d3D=128):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3,64,1), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64,128,1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128,256,1), nn.BatchNorm1d(256), nn.ReLU(True)
        )
        self.fc_global = nn.Sequential(nn.Linear(256,256), nn.ReLU(True), nn.Linear(256,d3D))
        self.fc_fuse = nn.Sequential(nn.Conv1d(256 + d3D, d3D, 1), nn.ReLU(True))
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, xyz):
        # xyz: (B,N,3)
        x = xyz.transpose(1,2)
        feat = self.mlp1(x)  # (B,256,N)
        g = F.adaptive_max_pool1d(feat,1).squeeze(-1)  # (B,256)
        g = self.fc_global(g)  # (B,d3D)
        g_exp = g.unsqueeze(-1).expand(-1,-1,feat.size(-1))
        fused = torch.cat([feat, g_exp], dim=1)
        out = self.fc_fuse(fused)
        return out.transpose(1,2)  # (B,N,d3D)

# -- Adapters --
class AdapterMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden + [out_dim]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(True)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# -- Helpers for sampling 2D features at UV --
def project_points_single(K, R, t, xyz):
    # K: (3,3) R:(3,3) t:(3,), xyz:(N,3)
    pts_cam = (R @ xyz.T + t.view(3,1)).T
    z = pts_cam[:,2].clamp(min=1e-6)
    uvw = (K @ pts_cam.T).T
    u = uvw[:,0] / z
    v = uvw[:,1] / z
    return torch.stack([u,v], dim=-1)

def sample_feats_at_uv_single(feat, uv, H, W):
    # feat: (C,h,w), uv: (N,2) in pixel coords of full image size H,W
    C, h, w = feat.shape
    # map uv from full H,W to feat h,w coordinates
    sx = W / float(w)
    sy = H / float(h)
    u = uv[:,0] / sx
    v = uv[:,1] / sy
    # normalize to [-1,1]
    grid_u = (u / (w - 1) * 2 - 1).clamp(-1,1)
    grid_v = (v / (h - 1) * 2 - 1).clamp(-1,1)
    grid = torch.stack([grid_u, grid_v], dim=-1).view(1,-1,2)
    feat_b = feat.unsqueeze(0)
    sampled = F.grid_sample(feat_b, grid.view(1,1,-1,2), align_corners=True)
    sampled = sampled.squeeze(2).squeeze(0).transpose(0,1)  # (N,C)
    return sampled

# -- PaDiM per-pixel Gaussian model --
def compute_mu_cov_grid(feat_stack):
    # feat_stack: (T, d, H, W)
    T, d, H, W = feat_stack.shape
    X = feat_stack.permute(2,3,0,1).reshape(H*W, T, d)
    mu = X.mean(dim=1)  # (HW,d)
    Xc = X - mu.unsqueeze(1)
    cov = torch.einsum('tnd,tme->nde', Xc, Xc) / max(T-1,1)
    cov = cov + 0.09 * torch.eye(d, device=feat_stack.device).unsqueeze(0)
    return mu, cov

def mahalanobis_grid_single(feat, mu_grid, cov_inv_grid):
    # feat: (d,H,W), mu_grid: (H,W,d), cov_inv_grid: (H,W,d,d)
    d, H, W = feat.shape
    X = feat.permute(1,2,0)  # (H,W,d)
    D = X - mu_grid
    M1 = torch.einsum('hwd,hwde->hwe', D, cov_inv_grid)
    M2 = torch.einsum('hwd,hwe->hw', D, M1)
    return torch.sqrt(torch.clamp(M2, min=0.0))

# Expose a light wrapper class
class TwoM3DFFeatureExtractor:
    def __init__(self, d2D=512, d3D=128, use_pointnet2=False, device='cuda'):
        self.device = device
        self.rgb = RGBBackboneWRN50L3().to(device)
        self.point = PointNetBackbone(d3D=d3D).to(device)
        self.adapt_2d_to_3d = AdapterMLP(d2D, [256,128], d3D).to(device)
        self.adapt_3d_to_2d = AdapterMLP(d3D, [128,256], d2D).to(device)
        self.d2D = d2D
        self.d3D = d3D

    def sample_mv_feats_at_points(self, feats_per_view, cams_per_view, pts, H, W):
        # feats_per_view: list of (C,h,w) cpu tensors or cuda
        # cams_per_view: list of (K,R,t) cpu or cuda
        N = pts.shape[0]
        acc = None
        for feat, (K,R,t) in zip(feats_per_view, cams_per_view):
            uv = project_points_single(K, R, t, pts)
            samp = sample_feats_at_uv_single(feat, uv, H, W)  # (N,C)
            acc = samp if acc is None else acc + samp
        return (acc / len(feats_per_view)).to(self.device)

    def forward_once(self, views_list, pts, cams_list, out_hw=(224,224)):
        # views_list: list of [1,3,H,W] tensors
        # pts: [N,3]
        H, W = out_hw
        # extract rgb features per view
        feats_per_view = []
        for v in views_list:
            f = self.rgb(v.to(self.device)).squeeze(0).detach()  # (C,h,w)
            feats_per_view.append(f)
        # prepare cams (K,R,t) as tensors on cpu or device; pts on cpu for sampling helper
        pts_cpu = pts.cpu()
        f2d_pts = self.sample_mv_feats_at_points(feats_per_view, cams_list, pts_cpu, H, W)  # (N,C)
        f3d = self.point(pts.unsqueeze(0).to(self.device)).squeeze(0)  # (N,d3D)
        # adapters
        a23 = self.adapt_2d_to_3d(f2d_pts)
        a32 = self.adapt_3d_to_2d(f3d)
        # alignment loss
        loss_cos = (1 - F.cosine_similarity(f2d_pts, a32, dim=-1)).mean() + (1 - F.cosine_similarity(f3d, a23, dim=-1)).mean()
        # normalize and fuse
        a23n = F.normalize(a23, dim=-1)
        a32n = F.normalize(a32, dim=-1)
        fmm = torch.cat([a23n, a32n], dim=-1)
        return f2d_pts, f3d, fmm, loss_cos

