"""Hero loop for the project page: a carousel of input-video panels over a white grid floor, each with its GravHMR
body walking in front and its recent root path traced on the floor. The camera pans along +x at constant speed;
slots repeat every N*S metres, so the video loops seamlessly (clip frames reset only while a slot is off screen).

Bodies come from the page's own exports (data/<id>.json/.bin, numpy LBS = the page's skinning). Per slot, the
clip is rotated about +y so its travel over the visible segment points at the viewer and translated so the current
root sits on the slot; the floor height is the clip's local ground (wild: exported ground_y; EMDB: same filter).
  python -I hero.py --out ../assets/hero        (frames cached in <out>/frames; delete to re-render)
"""
import os, sys, json, glob, argparse, subprocess
os.environ.setdefault("PYOPENGL_PLATFORM", "egl"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import numpy as np, cv2, trimesh
from scipy.spatial.transform import Rotation as Ro
from scipy.ndimage import minimum_filter1d, gaussian_filter1d

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--W", type=int, default=1600); ap.add_argument("--H", type=int, default=900)
ap.add_argument("--fps", type=float, default=25); ap.add_argument("--speed", type=float, default=1.2)
ap.add_argument("--slot", type=float, default=4.2); ap.add_argument("--ss", type=int, default=2)
ap.add_argument("--only", type=int, default=None, help="render a single frame index (for a look)")
ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworker", type=int, default=1)
ap.add_argument("--encode_only", action="store_true")
a = ap.parse_args()
H = "/data/home/hangleijin"; PP = f"{H}/project_page"
sys.path.insert(0, f"{H}/paper_figs"); import hq_render  # noqa: E402
UP = np.array([0.0, 1.0, 0.0])
FRAMES = {"6667-720": f"{H}/sports_wild", "25021-720": f"{H}/sports_wild", "4568-1080": f"{H}/sports_wild",
          "48371-720": f"{H}/sports_wild", "21477-720": f"{H}/cycling_wild"}
CLIPS = ["25021-720", "P7_58_outdoor_parcours", "6667-720", "48371-720", "P8_64_outdoor_skateboard", "21477-720", "4568-1080"]
BLUE = np.array([.30, .50, .92])


def load(cid):
    h = json.load(open(f"{PP}/data/{cid}.json")); b = open(f"{PP}/data/{cid}.bin", "rb").read()
    A = lambda d: np.frombuffer(b, dtype=d["dtype"], count=d["count"], offset=d["offset"]).copy()
    B = next(x for x in h["bodies"] if x["role"] == "ours")
    r = dict(n=h["frames"], fps=h["fps"], faces=A(h["faces"]).reshape(-1, 3).astype(np.int64), par=B["parents"],
             rest=A(B["rest_verts"]).reshape(-1, 3).astype(np.float64), J=A(B["rest_joints"]).reshape(-1, 3).astype(np.float64),
             si=A(B["skin_index"]).reshape(-1, 4).astype(np.int64), sw=A(B["skin_weight"]).reshape(-1, 4).astype(np.float64))
    nJ = len(r["par"]); r["q"] = A(B["quats"]).reshape(h["frames"], nJ, 4).astype(np.float64); r["root"] = A(B["root_pos"]).reshape(-1, 3).astype(np.float64)
    r["ground"] = A(h["ground_y"]).astype(np.float64) if "ground_y" in h else None
    return r


def skin(c, f):
    R = Ro.from_quat(c["q"][f]).as_matrix(); J = c["J"]; P = c["par"]; G = np.zeros((len(P), 4, 4))
    for j in range(len(P)):
        M = np.eye(4); M[:3, :3] = R[j]; M[:3, 3] = c["root"][f] if j == 0 else J[j] - J[P[j]]
        G[j] = M if j == 0 else G[P[j]] @ M
    for j in range(len(P)): G[j, :3, 3] -= G[j, :3, :3] @ J[j]
    v = np.c_[c["rest"], np.ones(len(c["rest"]))]
    return np.einsum("vk,vkij,vj->vi", c["sw"], G[c["si"]], v)[:, :3]


clips = []
for cid in CLIPS:
    c = load(cid)
    if c["ground"] is None:                      # EMDB export: local ground from the body itself, as the wild export does
        lo = np.array([skin(c, f)[:, 1].min() for f in range(c["n"])])
        c["ground"] = gaussian_filter1d(minimum_filter1d(lo, size=int(round(c["fps"])) | 1, mode="nearest"), c["fps"] / 4, mode="nearest")
    if cid in FRAMES:
        c["imgs"] = sorted(glob.glob(f"{FRAMES[cid]}/frames/{cid}/images/*.jpg"))
    else:
        c["imgs"] = None; c["vid"] = f"{PP}/data/{cid}.mp4"
    clips.append(c); print(cid, c["n"], "frames", flush=True)

N, S, v = len(clips), a.slot, a.speed
T = N * S / v; NF = int(round(T * a.fps)); VIS = 7.5                # slot visible for |x - cam| < VIS
need = 2 * VIS / v
for c in clips:
    seg = min(c["n"], int(need * c["fps"]) + 2); c["f0"] = (c["n"] - seg) // 2; c["seg"] = seg
    fr = np.arange(c["f0"], c["f0"] + seg); d = c["root"][fr[-1]] - c["root"][fr[0]]; d[1] = 0
    ang = np.arctan2(d[0], d[2]) if np.linalg.norm(d) > .5 else 0.0
    c["Ry"] = cv2.Rodrigues(np.array([0.0, -ang, 0.0]))[0]           # travel direction -> +z (toward the viewer)
    if c["imgs"] is None:
        cap = cv2.VideoCapture(c["vid"]); c["vframes"] = {}
        for i in range(c["n"]):
            ok, im = cap.read()
            if not ok: break
            if c["f0"] <= i < c["f0"] + seg: c["vframes"][i] = im
print(f"loop {T:.1f} s, {NF} frames", flush=True)


def frame_img(c, f):
    return cv2.imread(c["imgs"][f]) if c["imgs"] is not None else c["vframes"][min(max(f, c["f0"]), c["f0"] + c["seg"] - 1)]


def floor(cx):
    xs = np.arange(np.floor(cx) - 16, np.floor(cx) + 17, 1.0); zs = np.arange(-14, 9, 1.0)
    segs = [(np.array([x, 0, zs[0]]), np.array([x, 0, zs[-1]])) for x in xs] + [(np.array([xs[0], 0, z]), np.array([xs[-1], 0, z])) for z in zs]
    lines = trimesh.util.concatenate([trimesh.creation.cylinder(radius=.009, segment=np.stack(p) + UP * .002, sections=4) for p in segs])
    plane = trimesh.creation.box(extents=[xs[-1] - xs[0], .002, zs[-1] - zs[0]]); plane.apply_translation([(xs[0] + xs[-1]) / 2, -.001, (zs[0] + zs[-1]) / 2])
    return dict(plane=plane, lines=lines, y=0.0, albedo=np.array([.975, .977, .982]), line_albedo=np.array([.80, .82, .85]))


def ribbon(P, w=.05):
    if len(P) < 2: return None
    d = np.gradient(P, axis=0); d[:, 1] = 0; d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.cross(UP, d) * w / 2; L, R = P - nrm, P + nrm
    V = np.concatenate([L, R]) + UP * .004; n = len(P)
    F = np.concatenate([[[i, i + 1, n + i], [i + 1, n + i + 1, n + i]] for i in range(n - 1)])
    return V, F


def look(eye, tgt):
    f = tgt - eye; f /= np.linalg.norm(f); r = np.cross(f, UP); r /= np.linalg.norm(r); u = np.cross(r, f)
    P = np.eye(4); P[:3, :3] = np.stack([r, u, -f], 1); P[:3, 3] = eye; return P


yfov = np.radians(30); fpx = (a.H / 2) / np.tan(yfov / 2)
PW, PH, PZ, PY = 2.7, 2.7 * 9 / 16, -1.7, 1.75                       # panel size, depth, centre height
os.makedirs(f"{a.out}/frames", exist_ok=True)
todo = [a.only] if a.only is not None else ([] if a.encode_only else range(a.worker, NF, a.nworker))
for k in todo:
    o = f"{a.out}/frames/{k:05d}.jpg"
    if os.path.exists(o) and a.only is None: continue
    t = k / a.fps; cam_x = v * t
    pose = look(np.array([cam_x, 1.6, 8.6]), np.array([cam_x, 1.0, -0.6]))
    bodies, trails, panels = [], [], []
    for i, c in enumerate(clips):
        x = i * S + np.round((cam_x - i * S) / (N * S)) * N * S                 # nearest repeat of slot i
        rel = x - cam_x
        if abs(rel) > VIS + 2: continue
        f = c["f0"] + int(np.clip((VIS - rel) / v * c["fps"], 0, c["seg"] - 1))
        V = skin(c, f); r0 = c["root"][f].copy(); g = c["ground"][f]
        Rw = c["Ry"]; off = np.array([x, 0, .35]) - Rw @ np.array([r0[0], 0, r0[2]])
        Vw = V @ Rw.T + off; Vw[:, 1] -= g
        bodies.append((Vw, c["faces"], BLUE))
        f_lo = max(c["f0"], f - int(3.0 * c["fps"]))
        P = c["root"][f_lo:f + 1].copy(); P[:, 1] = 0; P = P @ Rw.T + off; P[:, 1] = 0
        rb = ribbon(P)
        if rb is not None: trails.append((rb[0], rb[1], np.array([.45, .63, .95])))
        panels.append((x, frame_img(c, f)))
    fl = floor(cam_x)
    kw = dict(ss=a.ss, rim=.06, gloss=.35, shininess=40, shadow=.28, fog=(9.0, 24.0))
    key = 3.0 * UP + np.array([.6, 0, .5])
    B = hq_render.render(trails + bodies, fl, pose, a.W, a.H, yfov, key, **kw).astype(np.float32)
    Cb, am = hq_render.render(bodies, None, pose, a.W, a.H, yfov, key, return_alpha=True, cast_shadow=False, ground_y=0.0, **kw)
    img = B.copy()
    for x, im in panels:                                                     # video cards, behind the bodies
        cs = np.array([[x - PW / 2, PY + PH / 2, PZ], [x + PW / 2, PY + PH / 2, PZ], [x + PW / 2, PY - PH / 2, PZ], [x - PW / 2, PY - PH / 2, PZ]])
        q = (cs - pose[:3, 3]) @ pose[:3, :3]; dst = np.stack([a.W / 2 + fpx * q[:, 0] / -q[:, 2], a.H / 2 - fpx * q[:, 1] / -q[:, 2]], 1).astype(np.float32)
        h0, w0 = im.shape[:2]; ar = w0 / h0
        if ar > 16 / 9: cw = int(h0 * 16 / 9); im = im[:, (w0 - cw) // 2:(w0 + cw) // 2]
        else: ch = int(w0 * 9 / 16); im = im[(h0 - ch) // 2:(h0 + ch) // 2]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32); h0, w0 = im.shape[:2]
        src = np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]); M = cv2.getPerspectiveTransform(src, dst)
        mask = np.zeros((h0, w0), np.float32); rr = int(.035 * w0)
        cv2.rectangle(mask, (rr, 0), (w0 - rr, h0), 1, -1); cv2.rectangle(mask, (0, rr), (w0, h0 - rr), 1, -1)
        for cx_, cy_ in ((rr, rr), (w0 - rr, rr), (rr, h0 - rr), (w0 - rr, h0 - rr)): cv2.circle(mask, (cx_, cy_), rr, 1, -1, cv2.LINE_AA)
        wi = cv2.warpPerspective(im, M, (a.W, a.H), flags=cv2.INTER_AREA); wm = cv2.warpPerspective(mask, M, (a.W, a.H))[..., None]
        sh = cv2.GaussianBlur(np.roll(wm[..., 0], 10, 0), (0, 0), 14)[..., None] * .18                   # soft drop shadow
        img = img * (1 - sh) ; img = img * (1 - wm) + wi * wm
    al = (am.astype(np.float32) / 255)[..., None]
    img = img * (1 - al) + Cb.astype(np.float32) * al
    cv2.imwrite(o, cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if k % 25 == 0: print(f"frame {k}/{NF}", flush=True)
if a.only is None and (a.encode_only or a.nworker == 1):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", f"{a.out}/frames/%05d.jpg",
                    "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", "-an", f"{a.out}/hero.mp4"], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", f"{a.out}/frames/{NF // 3:05d}.jpg", "-vf", "scale=1280:-2", "-q:v", "4", f"{a.out}/hero_poster.jpg"], check=True)
    print("saved", f"{a.out}/hero.mp4", os.path.getsize(f"{a.out}/hero.mp4") / 1e6, "MB")
