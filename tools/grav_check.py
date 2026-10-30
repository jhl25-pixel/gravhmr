"""Per wild clip: angle between world-up candidates (raw ViPE world): GeoCalib mean, GV-head (HMR) mean, and the
normal of the plane best fitting the root path (flat-ground assumption). Also GeoCalib per-frame spread."""
import os, sys, glob
import numpy as np, torch, cv2
H = "/data/home/hangleijin"
rod = lambda A: np.stack([cv2.Rodrigues(x)[0] for x in np.asarray(A, np.float64)])
un = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
ang = lambda a, b: np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1)))
for root, d, n in [(f"{H}/sports_wild", "htd_work", x) for x in ("6667-720", "4568-1080", "25021-720", "48371-720")] + [(f"{H}/cycling_wild", "htd_work_paper", "21477-720")]:
    p = torch.load(f"{root}/pred/{n}/hmr4d_results.pt", map_location="cpu", weights_only=False)
    f = sorted(glob.glob(f"{root}/{d}/{n}/opt/*/opt_results.pt"))
    g = torch.load(f[0], map_location="cpu", weights_only=False)["smpl_params_global_optimized"]
    L = len(g["transl"])
    Z = np.load(f"{root}/vipe/{n}.npz"); pos = {int(q): i for i, q in enumerate(Z["inds"])}
    D = Z["data"][[pos[q] for q in range(L)]].astype(np.float64); U, _, Vt = np.linalg.svd(D[:, :3, :3]); Rc2w = U @ Vt
    gc = un(np.einsum("nij,nj->ni", Rc2w, un(-np.load(f"{root}/geocalib/{n}.npz")["gravity_cam"][:L].astype(np.float64))))
    Gv = rod(p["smpl_params_global"]["global_orient"].numpy()[:L]); Gc = rod(p["smpl_params_incam"]["global_orient"].numpy()[:L])
    hm = un(np.einsum("nij,nj->ni", Rc2w, un(np.einsum("nij,nkj->nik", Gv, Gc)[:, 1, :])))
    ug, uh = un(gc.mean(0)), un(hm.mean(0))
    X = g["transl"].numpy().astype(np.float64); Xc = X - X.mean(0); w, V = np.linalg.eigh(Xc.T @ Xc); npl = V[:, 0]
    if npl @ ug < 0: npl = -npl
    print(f"{n:10s} GeoCalib-vs-HMR {ang(ug, uh):4.1f}  GeoCalib-vs-pathplane {ang(ug, npl):4.1f}  HMR-vs-pathplane {ang(uh, npl):4.1f}  "
          f"GeoCalib per-frame spread {np.degrees(np.arccos(np.clip(gc @ ug, -1, 1))).std():.1f} deg")
