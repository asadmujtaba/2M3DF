
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from feature_extractors.twom3df_features import TwoM3DFFeatureExtractor, compute_mu_cov_grid
from utils.metrics import compute_image_level_scores, compute_pixel_level_scores

class TwoM3DFRunner:
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.extractor = TwoM3DFFeatureExtractor(d2D=cfg['d2D'], d3D=cfg['d3D'], device=device)
        self.save_dir = cfg.get('save_path', './twom3df_result')
        os.makedirs(self.save_dir, exist_ok=True)
        self.gaussian_ckpt = os.path.join(self.save_dir, 'gaussian_grid.pkl')

    def train_adapters_and_fit(self, train_loader, out_hw=(224,224), epochs=60, lr=1e-3):
        # Train adapters only
        params = list(self.extractor.adapt_2d_to_3d.parameters()) + list(self.extractor.adapt_3d_to_2d.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        self.extractor.rgb.eval()
        self.extractor.point.eval()
        for ep in range(epochs):
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Adapters epoch {ep+1}"):
                views = batch['views']  # list length M of tensors [B,3,H,W]
                pts = batch['points']   # [B,N,3]
                cams = batch['cams']    # list per sample of list of (K,R,t)
                B = pts.shape[0]
                opt.zero_grad()
                loss_b = 0.0
                for b in range(B):
                    views_b = [v[b:b+1] for v in views]
                    pts_b = pts[b]
                    cams_b = cams[b]
                    _, _, _, loss = self.extractor.forward_once(views_b, pts_b, cams_b, out_hw)
                    loss_b = loss_b + loss
                loss_b = loss_b / B
                loss_b.backward()
                opt.step()
                total_loss += loss_b.item()
            print(f"Epoch {ep+1}/{epochs} avg_loss={total_loss/max(1,len(train_loader))}")

        # After adapters trained, extract fused features across train set and fit per-pixel Gaussians
        feat_list = []
        for batch in tqdm(train_loader, desc='Extract fused features for Gaussian'):
            views = batch['views']
            pts = batch['points']
            cams = batch['cams']
            B = pts.shape[0]
            for b in range(B):
                views_b = [v[b:b+1] for v in views]
                pts_b = pts[b]
                cams_b = cams[b]
                _, _, fmm, _ = self.extractor.forward_once(views_b, pts_b, cams_b, out_hw)
                # fmm: (N, dMM) -> scatter to grid using first view's uv
                # We'll use the util in your repo to project; as a simple approach here, create a grid mapping elsewhere
                # For now: save fmm and corresponding uv to disk for a dedicated fitting step
                feat_list.append({'fmm': fmm.cpu().numpy(), 'pts': pts_b.cpu().numpy(), 'cams': cams_b})
        # Now call an external util to create (T,d,H,W) from feat_list; we'll implement a simple grid builder here
        # NOTE: Replace with your existing utils/generate_multi_view_dataset or mvtec3d_util if available to get K,R,t format
        grid_feats = self._build_grid_from_featlist(feat_list, out_hw)
        mu, cov = compute_mu_cov_grid(grid_feats)
        # store mu and cov
        with open(self.gaussian_ckpt, 'wb') as f:
            pickle.dump({'mu': mu.cpu().numpy(), 'cov': cov.cpu().numpy()}, f)
        print('Saved gaussian grid to', self.gaussian_ckpt)

    def _build_grid_from_featlist(self, feat_list, out_hw):
        # Simple scatter: assumes cams in feat_list[...]['cams'] provide K,R,t with which to compute uv and scatter to HxW
        T = len(feat_list)
        d = feat_list[0]['fmm'].shape[1]
        H,W = out_hw
        grid = np.zeros((T, d, H, W), dtype=np.float32)
        for i,rec in enumerate(feat_list):
            fmm = rec['fmm']  # (N,d)
            pts = rec['pts']  # (N,3)
            cams = rec['cams']
            # project using first cam
            K,R,t = cams[0]
            # convert to numpy arrays
            K = np.array(K)
            R = np.array(R)
            t = np.array(t)
            pts_cam = (R @ pts.T + t.reshape(3,1)).T
            z = pts_cam[:,2].clip(1e-6)
            uvw = (K @ pts_cam.T).T
            u = (uvw[:,0] / z).round().astype(int)
            v = (uvw[:,1] / z).round().astype(int)
            u = np.clip(u, 0, W-1)
            v = np.clip(v, 0, H-1)
            # average collisions
            count = np.zeros((H,W), dtype=np.int32)
            acc = np.zeros((d,H,W), dtype=np.float32)
            for pi in range(pts.shape[0]):
                acc[:, v[pi], u[pi]] += fmm[pi]
                count[v[pi], u[pi]] += 1
            count = np.where(count==0, 1, count)
            acc = acc / count
            grid[i] = acc
        return torch.from_numpy(grid)

    def predict(self, test_loader, out_hw=(224,224)):
        # load gaussian
        import pickle
        with open(self.gaussian_ckpt, 'rb') as f:
            ck = pickle.load(f)
            mu = torch.from_numpy(ck['mu'])
            cov = torch.from_numpy(ck['cov'])
            # invert cov per pixel
            HWW, d, _ = cov.shape
            # reshape back if needed
            # cov originally (HW, d, d)
            inv = torch.linalg.pinv(cov)
        results = []
        for batch in tqdm(test_loader, desc='Predict'):
            views = batch['views']
            pts = batch['points']
            cams = batch['cams']
            gt_masks = batch.get('masks', None)
            B = pts.shape[0]
            for b in range(B):
                views_b = [v[b:b+1] for v in views]
                pts_b = pts[b]
                cams_b = cams[b]
                _, _, fmm, _ = self.extractor.forward_once(views_b, pts_b, cams_b, out_hw)
                # scatter to grid using first cam as during training
                # build grid for single sample
                # reuse same scatter logic as _build_grid_from_featlist for one sample
                # compute per-pixel Mahalanobis with mu and inv
                grid = self._build_grid_from_featlist([{'fmm': fmm.cpu().numpy(), 'pts': pts_b.cpu().numpy(), 'cams': cams_b}], out_hw)[0]
                # grid: (d,H,W)
                # compute mah per pixel
                H,W = out_hw
                d = grid.shape[0]
                # flatten
                vecs = grid.reshape(d, -1).T  # (HW, d)
                mu_flat = mu.reshape(-1, d)
                inv_flat = inv
                diff = vecs - mu_flat
                m = np.einsum('nd,ndd,nd->n', diff, inv_flat, diff)
                m = np.sqrt(np.clip(m, 0, None))
                amap = m.reshape(H,W)
                # normalize
                amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
                results.append({'amap': amap, 'gt_mask': None})
        return results
