"""For candidate EMDB sequences: Sim3 on the first 100 valid frames (the page's alignment), then the share of frames
where GravHMR is worse than the w/o rollout, and the error at a few checkpoints. Used to pick examples whose
improvement holds through the clip, not only at the end."""
import os, sys, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
import numpy as np, torch
H = "/data/home/hangleijin"; sys.path.insert(0, f"{H}/GVHMR_official"); os.chdir(f"{H}/GVHMR_official")
from hmr4d.utils.smplx_utils import make_smplx
dev="cuda"; sx=make_smplx("supermotion").eval().to(dev); x2s=torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt",weights_only=False).to_dense().to(dev)
JR=torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt",weights_only=False).to(dev)
L=torch.load("inputs/EMDB/hmr4d_support/emdb_vit_v4.pt",map_location="cpu",weights_only=False)
ROLL=f"{H}/sportHMR_success/gvhmr_pva/predictions_vipe_final"; OURS=f"{H}/htd_gvhmr_pva_f36hmr_final_work"
def g_(d,s):
    f=sorted(glob.glob(f"{d}/{s}/opt/*/opt_results.pt"))
    g=torch.load(f[0],map_location="cpu",weights_only=False)["smpl_params_global_optimized"] if f else torch.load(f"{d}/{s}/hmr4d_results.pt",map_location="cpu",weights_only=False)["smpl_params_global"]
    g={k:torch.as_tensor(v).float() for k,v in g.items() if k in("body_pose","betas","global_orient","transl")}; g["body_pose"]=g["body_pose"][:,:63]; return g
@torch.no_grad()
def pel(g):
    return torch.cat([(JR@torch.einsum("uv,lvi->lui",x2s,sx(**{k:v[i:i+256].to(dev) for k,v in g.items()}).vertices))[:,0] for i in range(0,len(g["transl"]),256)]).cpu().numpy()
@torch.no_grad()
def pgt(s):
    lab=L[s]; gp=lab["smpl_params"]; m=make_smplx("smpl",gender=lab["gender"]).eval().to(dev)
    return torch.cat([(JR@m(**{k:v[i:i+256].to(dev) for k,v in gp.items()}).vertices)[:,0] for i in range(0,len(gp["transl"]),256)]).cpu().numpy()
def sim3(P,Q):
    mp,mq=P.mean(0),Q.mean(0);U,S,Vt=np.linalg.svd((Q-mq).T@(P-mp));D=np.diag([1,1,np.sign(np.linalg.det(U@Vt))]);R=U@D@Vt
    return np.trace(np.diag(S)@D)/((P-mp)**2).sum(),R,mq-np.trace(np.diag(S)@D)/((P-mp)**2).sum()*R@mp
for s in sys.argv[1:]:
    v=np.where(L[s]["mask"].bool().numpy())[0]; G=pgt(s); out={}
    for k,d in (("base",ROLL),("ours",OURS)):
        P=pel(g_(d,s))[:len(G)]; sc,R,t=sim3(P[v[:100]],G[v[:100]]); out[k]=np.linalg.norm((sc*P@R.T+t)[v]-G[v],axis=1)
    w=out["ours"]>out["base"]; q=np.linspace(0,len(v)-1,7).astype(int)
    print(f"{s:32s} mean w/o {out['base'].mean():.2f} w/ {out['ours'].mean():.2f}  worse {w.mean()*100:4.0f}% of frames  "
          f"| " + " ".join(f"{out['base'][i]:.2f}->{out['ours'][i]:.2f}" for i in q))
