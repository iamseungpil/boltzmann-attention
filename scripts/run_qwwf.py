#!/usr/bin/env python3
"""Query-Weighted Water-Filling: PCA rotation + query-aware bit allocation"""
import argparse, numpy as np, torch, torch.nn as nn, gc, json, warnings
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
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True); p.add_argument("--bits",type=int,nargs="+",default=[2,3,4])
    p.add_argument("--output-dir",default="results")
    args=p.parse_args(); device="cuda:0"
    
    print(f"\n# QW-WF: {args.model}")
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    mdl=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16,trust_remote_code=True).to(device).eval()
    ds=load_dataset("wikitext","wikitext-2-raw-v1",split="test")
    txt="\n\n".join([t for t in ds["text"] if t.strip()])
    all_ids=tok.encode(txt,return_tensors="pt",truncation=False)
    cal=all_ids[:,:2048].to(device)
    cfg=mdl.config;nkv=getattr(cfg,'num_key_value_heads',cfg.num_attention_heads)
    nh=cfg.num_attention_heads;nly=cfg.num_hidden_layers;dh=cfg.hidden_size//nh;G=nh//nkv
    lyrs=mdl.model.layers
    
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
    
    pcab,eigv,qw_={},{},{}
    for (l,h) in kc:
        K,Q=kc[(l,h)],qc[(l,h)];Kc=K-K.mean(0)
        SK=Kc.T@Kc/K.shape[0]+1e-6*np.eye(dh);SQ=((Q-Q.mean(0)).T@(Q-Q.mean(0)))/Q.shape[0]+1e-6*np.eye(dh)
        ev,V=np.linalg.eigh(SK);pcab[(l,h)]=V;eigv[(l,h)]=np.maximum(ev,1e-10)
        qw_[(l,h)]=np.sqrt(np.maximum(np.diag(V.T@SQ@V),1e-10))
    
    clen,maxt=2048,min(all_ids.shape[1],50000);nc=(maxt-1)//clen
    results={"model":args.model,"methods":{}}
    
    for bits in args.bits:
        print(f"  --- {bits}-bit ---")
        # Standard PCA
        def mk_pca(li):
            def fn(m,i,o):
                k=o[0] if isinstance(o,tuple) else o;kn=k.detach().cpu().float().numpy();sh=kn.shape;kf=kn.reshape(-1,nkv,dh)
                for h in range(nkv):
                    R=pcab.get((li,h))
                    if R is not None:Kr=kf[:,h,:]@R;[None for j in range(dh) if not (Kr.__setitem__((slice(None),j),uq(Kr[:,j],bits)) or True)];kf[:,h,:]=Kr@R.T
                return torch.tensor(kf.reshape(sh),dtype=k.dtype,device=k.device)
            return fn
        # Fix: proper quantization
        def mk_pca2(li):
            def fn(m,i,o):
                k=o[0] if isinstance(o,tuple) else o;kn=k.detach().cpu().float().numpy();sh=kn.shape;kf=kn.reshape(-1,nkv,dh)
                for h in range(nkv):
                    R=pcab.get((li,h))
                    if R is not None:
                        Kr=kf[:,h,:]@R
                        for j in range(dh):Kr[:,j]=uq(Kr[:,j],bits)
                        kf[:,h,:]=Kr@R.T
                return torch.tensor(kf.reshape(sh),dtype=k.dtype,device=k.device)
            return fn
        
        def mk_qwf(li):
            def fn(m,i,o):
                k=o[0] if isinstance(o,tuple) else o;kn=k.detach().cpu().float().numpy();sh=kn.shape;kf=kn.reshape(-1,nkv,dh)
                for h in range(nkv):
                    R=pcab.get((li,h))
                    if R is None:continue
                    Kr=kf[:,h,:]@R;lam=eigv.get((li,h),np.ones(dh));w=qw_.get((li,h),np.ones(dh))
                    imp=lam*w;li2=np.log2(np.maximum(imp,1e-10))
                    ba=bits+0.5*(li2-np.mean(li2));ba=np.clip(ba,2,4);ba=ba*(dh*bits/max(ba.sum(),1e-10))
                    ba=np.maximum(np.round(ba),2).astype(int);ba=np.minimum(ba,4)
                    for j in range(dh):Kr[:,j]=uq(Kr[:,j],int(ba[j]))
                    kf[:,h,:]=Kr@R.T
                return torch.tensor(kf.reshape(sh),dtype=k.dtype,device=k.device)
            return fn
        
        for label,hf in [("pca",mk_pca2),("qw_wf",mk_qwf)]:
            hs=[];
            for l in range(nly):hs.append(lyrs[l].self_attn.k_proj.register_forward_hook(hf(l)))
            nll,t=0.0,0
            with torch.no_grad():
                for ci in range(nc):
                    s=ci*clen;e=s+clen+1
                    if e>maxt:break
                    inp=all_ids[:,s:e].to(device);out=mdl(inp[:,:-1],use_cache=False)
                    nll+=nn.CrossEntropyLoss(reduction='sum')(out.logits[0],inp[0,1:]).item();t+=inp.shape[1]-1
            for h in hs:h.remove()
            ppl=np.exp(nll/t);print(f"    {label}: {ppl:.4f}");results["methods"][f"{label}_{bits}bit"]=ppl
    
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S");tag=args.model.replace("/","_")
    outf=Path(args.output_dir)/f"qwwf_{tag}_{ts}.json"
    with open(outf,"w") as f:json.dump(results,f,indent=2)
    print(f"  Saved: {outf}")
    del mdl;torch.cuda.empty_cache();gc.collect()

if __name__=="__main__":main()
