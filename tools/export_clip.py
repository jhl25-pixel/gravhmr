"""Export one GravHMR result as a compact skinned-animation file for the project-page viewer.

Mesh: SMPL topology (SMPL-X result mapped with smplx2smpl), rest pose with the clip's betas, LBS weights
of SMPL-X folded onto the 22 body joints (hands -> wrists, jaw/eyes -> head; hand poses are zero anyway),
top-4 influences per vertex. Pose blend shapes are dropped (the check below reports what that costs).
World: the same frame the paper renders use -- GeoCalib gravity rotated to +y (render_global_motion.py),
then GVHMR's canonicalisation (first pelvis at origin, lowest vertex on y=0, first frame facing +z).
Output  <out>/<name>.bin  little-endian blob, layout in the JSON header <name>.json.
  python export_clip.py --root ~/sports_wild --dir ~/sports_wild/htd_work --name 7595-720 --out ../data
"""
import os, sys, json, glob, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import numpy as np, torch, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--dir", required=True)
ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
H = "/data/home/hangleijin"
sys.path.insert(0, f"{H}/paper_figs"); sys.path.insert(0, f"{H}/GVHMR_official"); os.chdir(f"{H}/GVHMR_official")
from hmr4d.utils.smplx_utils import make_smplx  # noqa: E402
from hmr4d.utils.geo_transform import compute_T_ayfz2ay  # noqa: E402
dev = "cuda"
sx = make_smplx("supermotion").eval().to(dev)
x2s = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt", weights_only=False).to_dense().to(dev)
FACES = make_smplx("smpl").faces.astype(np.uint16)
JR = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", weights_only=False).numpy()
rod = lambda A: np.stack([cv2.Rodrigues(x)[0] for x in np.asarray(A, np.float64)])
un = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
n = a.name

f = sorted(glob.glob(f"{a.dir}/{n}/opt/*/opt_results.pt"))
g = torch.load(f[0], map_location="cpu", weights_only=False)["smpl_params_global_optimized"] if f else \
    torch.load(f"{a.dir}/{n}/hmr4d_results.pt", map_location="cpu", weights_only=False)["smpl_params_global"]
g = {k: torch.as_tensor(v).float() for k, v in g.items() if k in ("body_pose", "betas", "global_orient", "transl")}
g["body_pose"] = g["body_pose"][:, :63]; L = len(g["transl"])

@torch.no_grad()
def smpl_verts(p):
    out = []
    for i in range(0, L, 256):
        v = sx(**{k: v[i:i + 256].to(dev) for k, v in p.items()}).vertices
        out.append(torch.einsum("uv,lvi->lui", x2s, v).cpu())
    return torch.cat(out).numpy().astype(np.float64)
V = smpl_verts(g)

# ---- world: gravity up (GeoCalib via ViPE camera), then canonicalise (same as render_global_motion.py) ----
Z = np.load(f"{a.root}/vipe/{n}.npz"); pos = {int(q): i for i, q in enumerate(Z["inds"])}
D = Z["data"][[pos[q] for q in range(L)]].astype(np.float64)
U, _, Vt = np.linalg.svd(D[:, :3, :3]); Rc2w = U @ Vt; Cw = D[:, :3, 3]
u_cam = un(-np.load(f"{a.root}/geocalib/{n}.npz")["gravity_cam"][:L].astype(np.float64))
nw = un(un(np.einsum("nij,nj->ni", Rc2w, u_cam)).mean(0))
UP = np.array([0.0, 1.0, 0.0]); ax = np.cross(nw, UP); s = np.linalg.norm(ax)
R_up = cv2.Rodrigues(ax / s * np.arctan2(s, nw @ UP))[0] if s > 1e-8 else np.eye(3)
V1 = V @ R_up.T
J1 = np.einsum("jv,lvi->lji", JR, V1)
off = J1[0, 0].copy(); off[1] = V1[:, :, 1].min()
T = compute_T_ayfz2ay(torch.from_numpy(J1[[0]] - off).float(), inverse=True)[0].double().numpy()
Rt = T[:3, :3] @ R_up; tt = T[:3, 3] - T[:3, :3] @ off           # x' = Rt x + tt
Vw = V @ Rt.T + tt

# ---- skinning rig ----
with torch.no_grad():
    z = lambda m: torch.zeros(1, m, device=dev)
    r = sx(betas=g["betas"][:1].to(dev), body_pose=z(63), global_orient=z(3), transl=z(3))
    Jrest = r.joints[0, :22].cpu().numpy().astype(np.float64)
    Vrest = (x2s @ r.vertices[0]).cpu().numpy().astype(np.float64)
    W = (x2s @ sx.bm.lbs_weights.to(dev)).cpu().numpy()                 # 6890 x 55
par = sx.bm.parents.cpu().numpy()[:22].astype(int); par[0] = -1
W22 = W[:, :22].copy()
for j in range(22, W.shape[1]):                                      # fold jaw/eyes/fingers onto a body joint
    k = j
    while k >= 22: k = int(sx.bm.parents[k])
    W22[:, k] += W[:, j]
idx = np.argsort(-W22, 1)[:, :4]; wt = np.take_along_axis(W22, idx, 1); wt /= wt.sum(1, keepdims=True)

# per-frame: root position (world, after Rt) and joint rotations (root rotated by Rt)
Rg = rod(g["global_orient"].numpy()); Rb = rod(g["body_pose"].numpy().reshape(-1, 3)).reshape(L, 21, 3, 3)
Rroot = np.einsum("ij,njk->nik", Rt, Rg)
root_pos = (Jrest[0] + g["transl"].numpy().astype(np.float64)) @ Rt.T + tt
def quat(M):   # (n,3,3) -> (n,4) xyzw
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(M).as_quat()
Q = np.concatenate([quat(Rroot)[:, None], quat(Rb.reshape(-1, 3, 3)).reshape(L, 21, 4)], 1)   # L,22,4

# ---- check: numpy LBS of exactly what the page will do vs the real mesh ----
def lbs(fr):
    G = np.zeros((22, 4, 4)); from scipy.spatial.transform import Rotation as Ro
    Rl = Ro.from_quat(Q[fr]).as_matrix()
    for j in range(22):
        M = np.eye(4); M[:3, :3] = Rl[j]
        M[:3, 3] = root_pos[fr] if j == 0 else Jrest[j] - Jrest[par[j]]
        G[j] = M if j == 0 else G[par[j]] @ M
    B = G.copy()
    for j in range(22): B[j, :3, 3] -= B[j, :3, :3] @ Jrest[j]
    vh = np.c_[Vrest, np.ones(len(Vrest))]
    return np.einsum("vk,vkij,vj->vi", wt, B[idx], vh)[:, :3]
chk = [int(x) for x in np.linspace(0, L - 1, 6)]
err = np.mean([np.linalg.norm(lbs(k) - Vw[k], axis=1).mean() * 1000 for k in chk])

# ---- camera (world, same transform) ----
Cc = Cw @ Rt.T + tt; Rcam = np.einsum("ij,njk->nik", Rt, Rc2w)
fps_s = open(f"{a.root}/frames/{n}/fps.txt").read().strip().split("/")
fps = float(fps_s[0]) / (float(fps_s[1]) if len(fps_s) > 1 else 1.0)

os.makedirs(a.out, exist_ok=True)
parts, hdr, off_b = [], {}, 0
def add(key, arr, dt):
    global off_b
    b = np.ascontiguousarray(arr, dtype=dt).tobytes(); pad = (-len(b)) % 4
    parts.append(b + b"\0" * pad); hdr[key] = {"offset": off_b, "count": int(np.asarray(arr).size), "dtype": np.dtype(dt).name}
    off_b += len(b) + pad
add("rest_verts", Vrest, np.float32); add("faces", FACES, np.uint16); add("skin_index", idx, np.uint8)
add("skin_weight", wt, np.float32); add("rest_joints", Jrest, np.float32)
add("quats", Q, np.float32); add("root_pos", root_pos, np.float32)
add("cam_pos", Cc, np.float32); add("cam_quat", quat(Rcam), np.float32)
open(f"{a.out}/{n}.bin", "wb").write(b"".join(parts))
meta = {"name": n, "frames": L, "fps": fps, "parents": par.tolist(), "buffers": hdr,
        "floor_y": float(Vw[:, :, 1].min()), "lbs_check_mm": round(float(err), 1),
        "path_len_m": float(np.linalg.norm(np.diff(root_pos[:, [0, 2]], axis=0), axis=1).sum())}
json.dump(meta, open(f"{a.out}/{n}.json", "w"))
print(f"{n}: L={L} fps={fps:.2f} lbs-vs-mesh {err:.1f} mm  bin {off_b/1e6:.2f} MB  path {meta['path_len_m']:.1f} m")
