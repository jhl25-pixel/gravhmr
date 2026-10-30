"""Export GravHMR results for the project-page viewer (skinned SMPL animation + the input video).

One clip = <out>/<id>.json (header) + <id>.bin (buffers) + <id>.mp4 (input frames, same frame count).
Each body is a 22/24-joint SMPL-topology rig (rest verts with the clip's betas, top-4 LBS weights, rest
joints) plus per-frame joint quaternions and root position. Any global similarity (gravity alignment,
canonical start, the Sim3 that puts a prediction in the GT frame) is folded into the rig, so the page
only plays back. Pose blend shapes are dropped; lbs_check_mm reports the cost against the true mesh.

  wild  : one body (GravHMR, paper config), world stood up by GeoCalib gravity through the ViPE camera,
          then GVHMR's canonical start. Also the ViPE camera path.
  emdb  : GT (gendered SMPL), w/o GravHMR (GVHMR + PVA-Net rollout) and w/ GravHMR (F36hmr + HTD, ViPE / ViPE-K),
          predictions Sim3-aligned to GT once on the first 100 valid frames (all joints), then all three put
          in GT's canonical start frame (first GT pelvis at origin, GT feet on y=0, GT facing +z).
  python -I export.py wild --root ~/sports_wild --dir ~/sports_wild/htd_work --name 7595-720 --out ../data
  python -I export.py emdb --seq P8_64_outdoor_skateboard --out ../data
"""
import os, sys, json, glob, argparse, subprocess
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import numpy as np, torch, cv2
from scipy.spatial.transform import Rotation as Ro

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["wild", "emdb"])
ap.add_argument("--root"); ap.add_argument("--dir"); ap.add_argument("--name"); ap.add_argument("--seq")
ap.add_argument("--id", default=None); ap.add_argument("--out", required=True)
ap.add_argument("--vh", type=int, default=480, help="video height")
a = ap.parse_args()
a.out = os.path.abspath(a.out)
H = "/data/home/hangleijin"
sys.path.insert(0, f"{H}/GVHMR_official"); os.chdir(f"{H}/GVHMR_official")
from hmr4d.utils.smplx_utils import make_smplx  # noqa: E402
from hmr4d.utils.geo_transform import compute_T_ayfz2ay  # noqa: E402
dev = "cuda"
sx = make_smplx("supermotion").eval().to(dev)
x2s = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt", weights_only=False).to_dense().to(dev)
FACES = make_smplx("smpl").faces.astype(np.uint16)
JR = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", weights_only=False).numpy()
rod = lambda A: np.stack([cv2.Rodrigues(x)[0] for x in np.asarray(A, np.float64)])
un = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
UP = np.array([0.0, 1.0, 0.0])


def load_pred(d, s):
    f = sorted(glob.glob(f"{d}/{s}/opt/*/opt_results.pt"))
    g = torch.load(f[0], map_location="cpu", weights_only=False)["smpl_params_global_optimized"] if f else \
        torch.load(f"{d}/{s}/hmr4d_results.pt", map_location="cpu", weights_only=False)["smpl_params_global"]
    g = {k: torch.as_tensor(v).float() for k, v in g.items() if k in ("body_pose", "betas", "global_orient", "transl")}
    g["body_pose"] = g["body_pose"][:, :63]; return g


class Body:
    """rig + motion of one model; verts() is the true mesh for the check and for alignment."""
    def __init__(self, g, kind="smplx", gender=None):
        self.g, self.kind, self.n = g, kind, len(g["transl"])
        self.m = sx if kind == "smplx" else make_smplx("smpl", gender=gender).eval().to(dev)
        bm = self.m.bm; nJ = 22 if kind == "smplx" else 24
        with torch.no_grad():
            z = lambda k: torch.zeros(1, k, device=dev)
            npose = 63 if kind == "smplx" else 69
            r = self.m(betas=g["betas"][:1].to(dev), body_pose=z(npose), global_orient=z(3), transl=z(3))
            v0 = r.vertices[0]
            Vr = (x2s @ v0) if kind == "smplx" else v0
            Jr = (bm.J_regressor.to(dev) @ v0)[:nJ]                       # the joints LBS actually rotates about
            W = ((x2s @ bm.lbs_weights.to(dev)) if kind == "smplx" else bm.lbs_weights.to(dev)).cpu().numpy()
        self.Vrest, self.Jrest = Vr.cpu().numpy().astype(np.float64), Jr.cpu().numpy().astype(np.float64)
        P = bm.parents.cpu().numpy().astype(int); self.par = P[:nJ].copy(); self.par[0] = -1
        W2 = W[:, :nJ].copy()
        for j in range(nJ, W.shape[1]):                                  # fold hands/face onto the body joint
            k = j
            while k >= nJ: k = int(P[k])
            W2[:, k] += W[:, j]
        self.idx = np.argsort(-W2, 1)[:, :4]; w = np.take_along_axis(W2, self.idx, 1); self.wt = w / w.sum(1, keepdims=True)
        self.Rl = np.concatenate([rod(g["global_orient"].numpy())[:, None],
                                  rod(g["body_pose"].numpy().reshape(-1, 3)).reshape(self.n, nJ - 1, 3, 3)], 1)
        self.root = self.Jrest[0] + g["transl"].numpy().astype(np.float64)
        self.s, self.R, self.t = 1.0, np.eye(3), np.zeros(3)           # x_world = s R x + t

    @torch.no_grad()
    def verts(self, fr=None):
        fr = np.arange(self.n) if fr is None else np.asarray(fr)
        out = []
        for i in range(0, len(fr), 256):
            q = {k: v[fr[i:i + 256]].to(dev) for k, v in self.g.items()}
            v = self.m(**q).vertices
            out.append((torch.einsum("uv,lvi->lui", x2s, v) if self.kind == "smplx" else v).cpu().numpy())
        V = np.concatenate(out).astype(np.float64)
        return self.s * V @ self.R.T + self.t

    def apply(self, s, R, t):
        self.s, self.R, self.t = s * self.s, R @ self.R, s * (R @ self.t) + t

    def lbs(self, f):
        Rl = self.Rl[f].copy(); Rl[0] = self.R @ Rl[0]
        Jr = self.s * self.Jrest; Vr = self.s * self.Vrest
        G = np.zeros((len(Rl), 4, 4))
        for j in range(len(Rl)):
            M = np.eye(4); M[:3, :3] = Rl[j]
            M[:3, 3] = (self.s * self.R @ self.root[f] + self.t) if j == 0 else Jr[j] - Jr[self.par[j]]
            G[j] = M if j == 0 else G[self.par[j]] @ M
        for j in range(len(Rl)): G[j, :3, 3] -= G[j, :3, :3] @ Jr[j]
        return np.einsum("vk,vkij,vj->vi", self.wt, G[self.idx], np.c_[Vr, np.ones(len(Vr))])[:, :3]

    def pack(self, frames):
        Rl = self.Rl[frames].copy(); Rl[:, 0] = np.einsum("ij,njk->nik", self.R, Rl[:, 0])
        Q = Ro.from_matrix(Rl.reshape(-1, 3, 3)).as_quat().reshape(len(frames), -1, 4)
        rp = self.s * self.root[frames] @ self.R.T + self.t
        chk = [frames[int(k)] for k in np.linspace(0, len(frames) - 1, 5)]
        Vt = self.verts(chk)
        err = float(np.mean([np.linalg.norm(self.lbs(f) - Vt[i], axis=1).mean() for i, f in enumerate(chk)]) * 1000)
        return dict(rest_verts=(self.s * self.Vrest, np.float32), rest_joints=(self.s * self.Jrest, np.float32),
                    skin_index=(self.idx, np.uint8), skin_weight=(self.wt, np.float32),
                    quats=(Q, np.float32), root_pos=(rp, np.float32)), err


def write(cid, bodies, frames, fps, img_paths, extra=None):
    parts, off = [], 0
    def add(arr, dt):
        nonlocal off
        b = np.ascontiguousarray(arr, dtype=dt).tobytes(); b += b"\0" * ((-len(b)) % 4)
        d = {"offset": off, "count": int(np.asarray(arr).size), "dtype": np.dtype(dt).name}; parts.append(b); off += len(b); return d
    hdr = {"id": cid, "frames": len(frames), "fps": fps, "faces": add(FACES, np.uint16), "bodies": []}
    for label, role, B in bodies:
        bufs, err = B.pack(frames)
        hdr["bodies"].append({"label": label, "role": role, "parents": B.par.tolist(), "lbs_check_mm": round(err, 1),
                              **{k: add(*v) for k, v in bufs.items()}})
        print(f"  {label}: lbs vs mesh {err:.1f} mm", flush=True)
    for k, (arr, dt) in (extra or {}).items(): hdr[k] = add(arr, dt)
    os.makedirs(a.out, exist_ok=True)
    open(f"{a.out}/{cid}.bin", "wb").write(b"".join(parts))
    # video: exactly the exported frames, in order, at the clip fps
    lst = f"/tmp/pp_{cid}.txt"
    with open(lst, "w") as fh:
        for p in img_paths: fh.write(f"file '{p}'\nduration {1 / fps:.6f}\n")
        fh.write(f"file '{img_paths[-1]}'\n")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", f"scale=-2:{a.vh},fps={fps}", "-frames:v", str(len(frames)), "-c:v", "libx264", "-preset", "slow",
                    "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", f"{a.out}/{cid}.mp4"], check=True)
    nv = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                             "stream=nb_read_frames", "-of", "csv=p=0", f"{a.out}/{cid}.mp4"], capture_output=True, text=True).stdout.strip())
    hdr["video_frames"] = nv
    json.dump(hdr, open(f"{a.out}/{cid}.json", "w"))
    print(f"{cid}: {len(frames)} frames @ {fps:.2f} fps, video {nv} frames, bin {off / 1e6:.2f} MB, "
          f"mp4 {os.path.getsize(f'{a.out}/{cid}.mp4') / 1e6:.2f} MB", flush=True)


def canon_T(V0, J0):
    """GVHMR canonical start: pelvis at origin (x,z), lowest vertex on y=0, facing +z. Returns (R, t)."""
    off = J0[0].copy(); off[1] = V0[:, 1].min()
    T = compute_T_ayfz2ay(torch.from_numpy(J0[None] - off).float(), inverse=True)[0].double().numpy()
    return T[:3, :3], T[:3, 3] - T[:3, :3] @ off


if a.mode == "wild":
    n = a.name; cid = a.id or n
    B = Body(load_pred(a.dir, n)); L = B.n
    Z = np.load(f"{a.root}/vipe/{n}.npz"); pos = {int(q): i for i, q in enumerate(Z["inds"])}
    D = Z["data"][[pos[q] for q in range(L)]].astype(np.float64)
    U, _, Vt = np.linalg.svd(D[:, :3, :3]); Rc2w = U @ Vt; Cw = D[:, :3, 3]
    u_cam = un(-np.load(f"{a.root}/geocalib/{n}.npz")["gravity_cam"][:L].astype(np.float64))
    nw = un(un(np.einsum("nij,nj->ni", Rc2w, u_cam)).mean(0)); ax = np.cross(nw, UP); sn = np.linalg.norm(ax)
    R_up = cv2.Rodrigues(ax / sn * np.arctan2(sn, nw @ UP))[0] if sn > 1e-8 else np.eye(3)
    B.apply(1.0, R_up, np.zeros(3))
    V0 = B.verts([0])[0]; Rc, tc = canon_T(V0, JR @ V0); B.apply(1.0, Rc, tc)
    Rt, tt = B.R, B.t
    cam_p = Cw @ Rt.T + tt; cam_q = Ro.from_matrix(np.einsum("ij,njk->nik", Rt, Rc2w)).as_quat()
    r = open(f"{a.root}/frames/{n}/fps.txt").read().strip().split("/"); fps = float(r[0]) / (float(r[1]) if len(r) > 1 else 1)
    imgs = sorted(glob.glob(f"{a.root}/frames/{n}/images/*.jpg"))
    assert len(imgs) >= L, (len(imgs), L)
    # local ground height under the subject: per-frame lowest vertex, min over +-0.5 s (drops flight phases),
    # then smoothed. The page draws the floor at this height so a sloped path never buries the body.
    from scipy.ndimage import minimum_filter1d, gaussian_filter1d
    lo = np.concatenate([B.verts(np.arange(i, min(L, i + 256)))[:, :, 1].min(1) for i in range(0, L, 256)])
    ground = gaussian_filter1d(minimum_filter1d(lo, size=max(3, int(round(fps)) | 1), mode="nearest"), sigma=fps / 4, mode="nearest")
    write(cid, [("GravHMR (ours)", "ours", B)], np.arange(L), fps, imgs[:L],
          {"cam_pos": (cam_p, np.float32), "cam_quat": (cam_q, np.float32), "ground_y": (ground, np.float32)})
else:
    s = a.seq; cid = a.id or s
    Lb = torch.load(f"{H}/GVHMR_official/inputs/EMDB/hmr4d_support/emdb_vit_v4.pt", map_location="cpu", weights_only=False)[s]
    gp = {k: v.float() for k, v in Lb["smpl_params"].items()}
    G = Body(gp, "smpl", Lb["gender"])
    W = Body(load_pred(f"{H}/sportHMR_success/gvhmr_pva/predictions_vipe_final", s))
    O = Body(load_pred(f"{H}/htd_gvhmr_pva_f36hmr_final_work", s))
    m = Lb["mask"].bool().numpy(); v = np.where(m)[0]; fr = np.arange(v[0], v[-1] + 1)
    Vg = G.verts(); Jg = np.einsum("jv,lvi->lji", JR, Vg)
    def sim3(P, Q):
        mp, mq = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((Q - mq).T @ (P - mp))
        Dg = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ Dg @ Vt; sc = np.trace(np.diag(S) @ Dg) / ((P - mp) ** 2).sum()
        return sc, R, mq - sc * R @ mp
    for B in (W, O):
        Jp = np.einsum("jv,lvi->lji", JR, B.verts(v[:100]))
        B.apply(*sim3(Jp.reshape(-1, 3), Jg[v[:100]].reshape(-1, 3)))
    Rc, tc = canon_T(Vg[fr[0]], Jg[fr[0]])
    for B in (G, W, O): B.apply(1.0, Rc, tc)
    imgs = sorted(glob.glob(f"{H}/sportsHMR/datasets/EMDB/{s[:2]}/{s[3:]}/images/*.jpg"))
    write(cid, [("Ground truth", "gt", G), ("w/o GravHMR", "base", W), ("w/ GravHMR", "ours", O)], fr, 30.0,
          [imgs[f] for f in fr])
