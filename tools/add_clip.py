"""Add one in-the-wild GravHMR result to the project page (no HTML editing).

Exports the result as a skinned SMPL animation + the input video + a thumbnail into data/, and
registers it in data/clips.json, which the page reads to build the in-the-wild clip list.
The world is stood upright with the GVHMR gravity-view vertical seen through the ViPE camera (the
gravity the method uses; --gravity geocalib uses GeoCalib instead), then put in GVHMR's canonical
start (first pelvis at the origin, lowest vertex on y=0, facing +z) -- the same convention as the
clips already on the page.

  python tools/add_clip.py --gvhmr_root /path/to/GVHMR \
      --result outputs/run/fatc/huayun5/opt/huayun5/opt_results.pt \
      --hmr    outputs/run/hmr/huayun5/hmr4d_results.pt \
      --vipe   outputs/run/vipe/huayun5.npz \
      --video  inputs/huayun5.mp4 --id huayun5 --sport Dancing
  bash tools/publish.sh "add huayun5"

--result also accepts a core result (hmr4d_results.pt). --remove ID drops a clip from the list.
"""
import argparse, glob, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--gvhmr_root", default=os.environ.get("GVHMR_ROOT"), help="GVHMR checkout (body models)")
ap.add_argument("--result", help="opt_results.pt (FATC) or hmr4d_results.pt (core)")
ap.add_argument("--hmr", help="GVHMR prediction hmr4d_results.pt (global + incam orientation, for gravity)")
ap.add_argument("--vipe", help="ViPE camera npz: data (n,4,4) camera-to-world, inds (n,)")
ap.add_argument("--video", help="input video (frames must match the result one to one)")
ap.add_argument("--fps", type=float, default=None, help="default: read from the video")
ap.add_argument("--gravity", default="hmr", choices=["hmr", "geocalib", "none"],
                help="none: the result is already gravity-upright (+y up), e.g. an exported demo bundle")
ap.add_argument("--camera", help="instead of --vipe: npz/pt with T_c2w (n,4,4) in the same world as the result")
ap.add_argument("--geocalib", help="GeoCalib npz with gravity_cam (only for --gravity geocalib)")
ap.add_argument("--id", help="clip id (file name on the page)")
ap.add_argument("--sport", help="label shown on the clip button, e.g. Running")
ap.add_argument("--first", action="store_true", help="put the clip first (the page opens with it)")
ap.add_argument("--max_frames", type=int, default=None, help="keep only the first N frames (e.g. drop a tail where tracking was lost)")
ap.add_argument("--vh", type=int, default=360, help="video height on the page")
ap.add_argument("--out", default=str(ROOT / "data"))
ap.add_argument("--remove", metavar="ID", help="remove a clip from data/clips.json and delete its files")
a = ap.parse_args()
OUT = Path(a.out); LIST = OUT / "clips.json"


def load_list():
    return json.load(open(LIST)) if LIST.exists() else []


def save_list(L):
    json.dump(L, open(LIST, "w"), indent=1)


if a.remove:
    L = [c for c in load_list() if c["id"] != a.remove]; save_list(L)
    for ext in ("json", "bin", "mp4", "jpg"):
        (OUT / f"{a.remove}.{ext}").unlink(missing_ok=True)
    print(f"removed {a.remove}; {len(L)} clips left"); sys.exit(0)

need = [k for k in ("gvhmr_root", "result", "video", "id", "sport") if not getattr(a, k)]
if not (a.vipe or a.camera): need.append("vipe (or --camera)")
if a.gravity == "hmr" and not a.hmr: need.append("hmr (or --gravity none)")
if need:
    ap.error("missing: " + ", ".join("--" + k for k in need))
for k in ("result", "hmr", "vipe", "video", "camera", "geocalib"):
    if getattr(a, k): setattr(a, k, str(Path(getattr(a, k)).resolve()))
OUT = OUT.resolve(); LIST = OUT / "clips.json"

import numpy as np, torch, cv2  # noqa: E402
from scipy.spatial.transform import Rotation as Ro  # noqa: E402
from scipy.ndimage import minimum_filter1d, gaussian_filter1d  # noqa: E402

sys.path.insert(0, str(Path(a.gvhmr_root).resolve())); os.chdir(Path(a.gvhmr_root).resolve())
from hmr4d.utils.smplx_utils import make_smplx  # noqa: E402
from hmr4d.utils.geo_transform import compute_T_ayfz2ay  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
sx = make_smplx("supermotion").eval().to(dev)
x2s = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt", weights_only=False).to_dense().to(dev)
FACES = make_smplx("smpl").faces.astype(np.uint16)
JR = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt", weights_only=False).numpy()
rod = lambda A: np.stack([cv2.Rodrigues(x)[0] for x in np.asarray(A, np.float64)])
un = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
UP = np.array([0.0, 1.0, 0.0])

# ---- result
if a.result.endswith(".npz"):
    r = {k: torch.from_numpy(v) for k, v in np.load(a.result).items()}; g = r     # flat SMPL-X dict
else:
    r = torch.load(a.result, map_location="cpu", weights_only=False)
    g = r.get("smpl_params_global_optimized", r.get("smpl_params_global", r))
g = {k: torch.as_tensor(v).float() for k, v in g.items() if k in ("body_pose", "betas", "global_orient", "transl")}
g["body_pose"] = g["body_pose"][:, :63]
if a.max_frames: g = {k: v[:a.max_frames] for k, v in g.items()}
L = len(g["transl"])


class Body:
    """22-joint SMPL-topology rig of a SMPL-X result; x_world = R x + t folded into the rig."""
    def __init__(self, g):
        self.g, self.n = g, len(g["transl"]); bm = sx.bm
        with torch.no_grad():
            z = lambda k: torch.zeros(1, k, device=dev)
            v0 = sx(betas=g["betas"][:1].to(dev), body_pose=z(63), global_orient=z(3), transl=z(3)).vertices[0]
            Jr = (bm.J_regressor.to(dev) @ v0)[:22]
            W = (x2s @ bm.lbs_weights.to(dev)).cpu().numpy()
        self.Vrest, self.Jrest = (x2s @ v0).cpu().numpy().astype(np.float64), Jr.cpu().numpy().astype(np.float64)
        P = bm.parents.cpu().numpy().astype(int); self.par = P[:22].copy(); self.par[0] = -1
        W2 = W[:, :22].copy()
        for j in range(22, W.shape[1]):
            k = j
            while k >= 22: k = int(P[k])
            W2[:, k] += W[:, j]
        self.idx = np.argsort(-W2, 1)[:, :4]; w = np.take_along_axis(W2, self.idx, 1); self.wt = w / w.sum(1, keepdims=True)
        self.Rl = np.concatenate([rod(g["global_orient"].numpy())[:, None],
                                  rod(g["body_pose"].numpy().reshape(-1, 3)).reshape(self.n, 21, 3, 3)], 1)
        self.root = self.Jrest[0] + g["transl"].numpy().astype(np.float64)
        self.R, self.t = np.eye(3), np.zeros(3)

    @torch.no_grad()
    def verts(self, fr):
        fr = np.asarray(fr); out = []
        for i in range(0, len(fr), 256):
            v = sx(**{k: v[fr[i:i + 256]].to(dev) for k, v in self.g.items()}).vertices
            out.append(torch.einsum("uv,lvi->lui", x2s, v).cpu().numpy())
        return np.concatenate(out).astype(np.float64) @ self.R.T + self.t

    def apply(self, R, t):
        self.R, self.t = R @ self.R, R @ self.t + t

    def lbs(self, f):
        Rl = self.Rl[f].copy(); Rl[0] = self.R @ Rl[0]; G = np.zeros((22, 4, 4))
        for j in range(22):
            M = np.eye(4); M[:3, :3] = Rl[j]
            M[:3, 3] = (self.R @ self.root[f] + self.t) if j == 0 else self.Jrest[j] - self.Jrest[self.par[j]]
            G[j] = M if j == 0 else G[self.par[j]] @ M
        for j in range(22): G[j, :3, 3] -= G[j, :3, :3] @ self.Jrest[j]
        return np.einsum("vk,vkij,vj->vi", self.wt, G[self.idx], np.c_[self.Vrest, np.ones(len(self.Vrest))])[:, :3]


B = Body(g)

# ---- gravity up in the ViPE world, then canonical start
if a.camera:
    Zc = np.load(a.camera) if a.camera.endswith(".npz") else {k: np.asarray(v) for k, v in torch.load(a.camera, map_location="cpu", weights_only=False).items()}
    D = np.asarray(Zc["T_c2w"])[:L].astype(np.float64)
else:
    Z = np.load(a.vipe); pos = {int(q): i for i, q in enumerate(Z["inds"])}
    D = Z["data"][[pos[q] for q in range(L)]].astype(np.float64)
U, _, Vt = np.linalg.svd(D[:, :3, :3]); Rc2w = U @ Vt; Cw = D[:, :3, 3]
if a.gravity == "none":
    u_cam = None
elif a.gravity == "geocalib":
    u_cam = un(-np.load(a.geocalib)["gravity_cam"][:L].astype(np.float64))
else:
    p = torch.load(a.hmr, map_location="cpu", weights_only=False)
    Gv = rod(p["smpl_params_global"]["global_orient"].numpy()[:L]); Gc = rod(p["smpl_params_incam"]["global_orient"].numpy()[:L])
    u_cam = un(np.einsum("nij,nkj->nik", Gv, Gc)[:, 1, :])
nw = UP.copy() if u_cam is None else un(un(np.einsum("nij,nj->ni", Rc2w, u_cam)).mean(0)); ax = np.cross(nw, UP); sn = np.linalg.norm(ax)
R_up = cv2.Rodrigues(ax / sn * np.arctan2(sn, nw @ UP))[0] if sn > 1e-8 else np.eye(3)
B.apply(R_up, np.zeros(3))
V0 = B.verts([0])[0]; J0 = JR @ V0; off = J0[0].copy(); off[1] = V0[:, 1].min()
T = compute_T_ayfz2ay(torch.from_numpy(J0[None] - off).float(), inverse=True)[0].double().numpy()
B.apply(T[:3, :3], T[:3, 3] - T[:3, :3] @ off)
cam_p = Cw @ B.R.T + B.t; cam_q = Ro.from_matrix(np.einsum("ij,njk->nik", B.R, Rc2w)).as_quat()
lo = np.concatenate([B.verts(np.arange(i, min(L, i + 256)))[:, :, 1].min(1) for i in range(0, L, 256)])

# ---- video
cap = cv2.VideoCapture(a.video); nv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = a.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
if nv and nv < L:
    sys.exit(f"video has {nv} frames but the result has {L}; pass the video the pipeline ran on")
ground = gaussian_filter1d(minimum_filter1d(lo, size=max(3, int(round(fps)) | 1), mode="nearest"), sigma=fps / 4, mode="nearest")

# ---- write
Rl = B.Rl.copy(); Rl[:, 0] = np.einsum("ij,njk->nik", B.R, Rl[:, 0])
Q = Ro.from_matrix(Rl.reshape(-1, 3, 3)).as_quat().reshape(L, 22, 4); rp = B.root @ B.R.T + B.t
chk = [int(k) for k in np.linspace(0, L - 1, 5)]; Vc = B.verts(chk)
err = float(np.mean([np.linalg.norm(B.lbs(f) - Vc[i], axis=1).mean() for i, f in enumerate(chk)]) * 1000)
parts, off_b = [], 0
def add(arr, dt):
    global off_b
    b = np.ascontiguousarray(arr, dtype=dt).tobytes(); b += b"\0" * ((-len(b)) % 4)
    d = {"offset": off_b, "count": int(np.asarray(arr).size), "dtype": np.dtype(dt).name}; parts.append(b); off_b += len(b); return d
hdr = {"id": a.id, "frames": L, "fps": float(fps), "faces": add(FACES, np.uint16), "bodies": [{
    "label": "GravHMR (ours)", "role": "ours", "parents": B.par.tolist(), "lbs_check_mm": round(err, 1),
    "rest_verts": add(B.Vrest, np.float32), "rest_joints": add(B.Jrest, np.float32), "skin_index": add(B.idx, np.uint8),
    "skin_weight": add(B.wt, np.float32), "quats": add(Q, np.float32), "root_pos": add(rp, np.float32)}]}
hdr["cam_pos"] = add(cam_p, np.float32); hdr["cam_quat"] = add(cam_q, np.float32); hdr["ground_y"] = add(ground, np.float32)
OUT.mkdir(parents=True, exist_ok=True)
open(OUT / f"{a.id}.bin", "wb").write(b"".join(parts))
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.video, "-vf", f"scale=-2:{a.vh}", "-frames:v", str(L),
                "-r", f"{fps}", "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an", str(OUT / f"{a.id}.mp4")], check=True)
hdr["video_frames"] = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                          "stream=nb_read_frames", "-of", "csv=p=0", str(OUT / f"{a.id}.mp4")],
                                         capture_output=True, text=True).stdout.strip())
json.dump(hdr, open(OUT / f"{a.id}.json", "w"))
cap.set(cv2.CAP_PROP_POS_FRAMES, L // 2); ok, im = cap.read()
if ok:
    h, w = im.shape[:2]; s = max(320 / w, 180 / h); im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = (im.shape[0] - 180) // 2, (im.shape[1] - 320) // 2
    cv2.imwrite(str(OUT / f"{a.id}.jpg"), im[y0:y0 + 180, x0:x0 + 320], [cv2.IMWRITE_JPEG_QUALITY, 88])

Lst = [c for c in load_list() if c["id"] != a.id]; entry = {"id": a.id, "sport": a.sport, "frames": L}
Lst = [entry] + Lst if a.first else Lst + [entry]; save_list(Lst)
tilt = np.degrees(np.arccos(np.clip(nw @ UP, -1, 1)))
print(f"{a.id}: {L} frames @ {fps:.2f} fps, video {hdr['video_frames']} frames, world up {tilt:.0f} deg from +y (rotated), "
      f"lbs vs mesh {err:.1f} mm, bin {off_b / 1e6:.2f} MB -> {OUT}  ({len(Lst)} clips listed)")
