#!/usr/bin/env python3
"""P3: Validate b_crit and D_attn vs PPL correlation across models"""
import numpy as np, torch, torch.nn as nn, json, gc, warnings
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
warnings.filterwarnings("ignore")

def uq(col,bits):
    nl=2**bits;vmin,vmax=col.min(),col.max()
    if vmax-vmin<1e-10:return col.copy()
    s=(vmax-vmin)/(nl-1);q=np.clip(np.round((col-vmin)/s).astype(int),0,nl-1);return q*s+vmin

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--output-dir",default="results")
    args=p.parse_args(); device="cuda:0"
    
    print(f"\n# b_crit validation: {args.model}")
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    mdl=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,trust_remote_code=True).to(device).eval()
    ds=load_dataset("wikitext","wikitext-2-raw-v1",split="test")
    txt="\n\n".join([t for t in ds["text"] if t.strip()])
    all_ids=tok.encode(txt,return_tensors="pt",truncation=False)
    cal=all_ids[:,:2048].to(device)
    cfg=mdl.config;nkv=getattr(cfg,'num_key_value_heads',cfg.num_attention_heads)
    nh=cfg.num_attention_heads;nly=cfg.num_hidden_layers;dh=cfg.hidden_size//nh;G=nh//nkv
    lyrs=mdl.model.layers
    
    # Calibrate K and Q
    kc,qc,hks={},{},[]
    def mh(li):
        def fn(mod,a,kw):
            hs=a[0] if a else kw.get('hidden_states')
            if hs is not None:
                k=mod.k_proj(hs)[0].detach().cpu().float().numpy().reshape(-1,nkv,dh)
                q=mod.q_proj(hs)[0].detach().cpu().float().numpy().reshape(-1,nh,dh)
                for h in range(nkv):kc[(li,h)]=k[:,h,:];qc[(li,h)]=q[:,h*G:(h+1)*G,:].mean(1)
        return fn
    for l in range(nly):hks.append(lyrs[l].self_attn.register_forward_pre_hook(mh(l),with_kwargs=True))
    with torch.no_grad():mdl(cal,use_cache=False)
    for h in hks:h.remove()
    
    # Compute κ(Σ_K), κ(Σ_Q), b_crit per head
    kappa_k_all, kappa_q_all, bcrit_all, angles_all = [],[],[],[]
    for (l,h) in sorted(kc.keys()):
        K,Q=kc[(l,h)],qc[(l,h)]
        Kc=K-K.mean(0); Qc=Q-Q.mean(0)
        SK=Kc.T@Kc/K.shape[0]+1e-6*np.eye(dh)
        SQ=Qc.T@Qc/Q.shape[0]+1e-6*np.eye(dh)
        eigK=np.linalg.eigvalsh(SK); eigQ=np.linalg.eigvalsh(SQ)
        kk=eigK[-1]/max(eigK[0],1e-10); kq=eigQ[-1]/max(eigQ[0],1e-10)
        bc=0.5*np.log2(max(kk*kq,1)); 
        kappa_k_all.append(float(kk)); kappa_q_all.append(float(kq)); bcrit_all.append(float(bc))
        # PCA-Q angle
        VK=np.linalg.eigh(SK)[1]; VQ=np.linalg.eigh(SQ)[1]
        sv=np.linalg.svd(np.abs(VK[:,-10:].T@VQ[:,-10:]),compute_uv=False)
        angles_all.append(float(np.degrees(np.arccos(np.clip(sv[0],-1,1)))))
    
    results = {
        "model": args.model,
        "kappa_K": {"mean":float(np.mean(kappa_k_all)),"median":float(np.median(kappa_k_all)),"max":float(np.max(kappa_k_all))},
        "kappa_Q": {"mean":float(np.mean(kappa_q_all)),"median":float(np.median(kappa_q_all)),"max":float(np.max(kappa_q_all))},
        "b_crit": {"mean":float(np.mean(bcrit_all)),"median":float(np.median(bcrit_all)),"max":float(np.max(bcrit_all))},
        "pca_q_angle": {"mean":float(np.mean(angles_all)),"std":float(np.std(angles_all))},
        "n_heads": len(kappa_k_all),
    }
    
    print(f"  κ(Σ_K): mean={np.mean(kappa_k_all):.0f}, median={np.median(kappa_k_all):.0f}")
    print(f"  κ(Σ_Q): mean={np.mean(kappa_q_all):.0f}, median={np.median(kappa_q_all):.0f}")
    print(f"  b_crit: mean={np.mean(bcrit_all):.1f}, median={np.median(bcrit_all):.1f}")
    print(f"  PCA-Q angle: {np.mean(angles_all):.1f}° ± {np.std(angles_all):.1f}°")
    
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S");tag=args.model.replace("/","_")
    outf=Path(args.output_dir)/f"bcrit_{tag}_{ts}.json"
    with open(outf,"w") as f:json.dump(results,f,indent=2)
    print(f"  Saved: {outf}")
    del mdl;torch.cuda.empty_cache();gc.collect()

if __name__=="__main__":main()
