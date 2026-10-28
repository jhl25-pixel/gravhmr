"""Rank EMDB-2 sequences by how much GravHMR fixes the ROOT trajectory of its own backbone.
w/o GravHMR = GVHMR + PVA-Net rollout (predictions_vipe_final), w/ GravHMR = paper result (F36hmr + HTD, ViPE / ViPE-K).
Root error = pelvis position error over the whole sequence after ONE Sim3 on the first 100 valid frames
(the alignment the page will show), mm."""
import os, sys, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import numpy as np, torch
H = "/data/home/hangleijin"; sys.path.insert(0, f"{H}/GVHMR_official"); os.chdir(f"{H}/GVHMR_official")
from hmr4d.utils.smplx_utils import make_smplx  # noqa
dev = "cuda"; sx = make_smplx("supermotion").eval().to(dev)
sm = {g: make_smplx("smpl", gender=g).eval().to(dev) for g in ("male", "female")}
x2s = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt", weights_only=False).to_dense().to(dev)
JR = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", weights_only=False).to(dev)
L = torch.load("inputs/EMDB/hmr4d_support/emdb_vit_v4.pt", map_location="cpu", weights_only=False)
ROLL = f"{H}/sportHMR_success/gvhmr_pva/predictions_vipe_final"; OURS = f"{H}/htd_gvhmr_pva_f36hmr_final_work"

def glob_params(d, s):
    f = sorted(glob.glob(f"{d}/{s}/opt/*/opt_results.pt"))
    g = torch.load(f[0], map_location="cpu", weights_only=False)["smpl_params_global_optimized"] if f else \
        torch.load(f"{d}/{s}/hmr4d_results.pt", map_location="cpu", weights_only=False)["smpl_params_global"]
    g = {k: torch.as_tensor(v).float() for k, v in g.items() if k in ("body_pose", "betas", "global_orient", "transl")}
    g["body_pose"] = g["body_pose"][:, :63]; return g

@torch.no_grad()
def pelvis_x(g):
    return torch.cat([(JR @ torch.einsum("uv,lvi->lui", x2s, sx(**{k: v[i:i+256].to(dev) for k, v in g.items()}).vertices))[:, 0]
                      for i in range(0, len(g["transl"]), 256)]).cpu().numpy()
@torch.no_grad()
def pelvis_gt(s):
    lab = L[s]; gp = lab["smpl_params"]
    return torch.cat([(JR @ sm[lab["gender"]](**{k: v[i:i+256].to(dev) for k, v in gp.items()}).vertices)[:, 0]
                      for i in range(0, len(gp["transl"]), 256)]).cpu().numpy()

def sim3(P, G):
    mp, mg = P.mean(0), G.mean(0); U, S, Vt = np.linalg.svd((G - mg).T @ (P - mp))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / ((P - mp) ** 2).sum()
    return s, R, mg - s * R @ mp

rows = []
for s in sorted(x for x in os.listdir(OURS) if x in L):
    m = L[s]["mask"].bool().numpy(); v = np.where(m)[0]; G = pelvis_gt(s)
    out = []
    for d in (ROLL, OURS):
        P = pelvis_x(glob_params(d, s))[:len(G)]
        sc, R, t = sim3(P[v[:100]], G[v[:100]]); A = sc * P @ R.T + t
        out.append(np.linalg.norm(A[v] - G[v], axis=1).mean() * 1000)
    path = np.linalg.norm(np.diff(G[v][:, [0, 2]], axis=0), axis=1).sum()
    rows.append((s, out[0], out[1], len(v), path))
for r in sorted(rows, key=lambda r: r[1] - r[2], reverse=True):
    print(f"{r[0]:34s} w/o {r[1]:7.0f}  w/ {r[2]:6.0f}  gain {r[1]-r[2]:7.0f} mm   n={r[3]:5d}  GT path {r[4]:5.1f} m")
