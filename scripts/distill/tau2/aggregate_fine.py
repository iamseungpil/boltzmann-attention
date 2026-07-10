#!/usr/bin/env python3
"""전 (모델×도메인) fine 분해를 도메인별 pooled 집계 → G1-G9 도메인-불변성.
사용: python3 aggregate_fine.py /path/to/traj_dir"""
import sys, glob, os
from collections import Counter, defaultdict
import fine_function_decomp as F

# leaf → 일반기능 G
def gmap(leaf):
    l=leaf.lower()
    if "coverage" in l: return "G1_COVERAGE"
    if "reach" in l or "gather" in l: return "G2_REACH"
    if "verify" in l: return "G3_VERIFY"
    if "escalate" in l or "persist" in l: return "G4_PERSISTENCE"
    if "over-action" in l or "scope" in l: return "G5_SCOPE"
    if "guidance" in l: return "G9_GUIDANCE"
    if "arg_variant" in l or "arg_numeric" in l or l.startswith("arg_other") or "arg_categorical" in l: return "G6_OPERAND"
    if "arg_reference" in l: return "G7_REFERENCE"
    if "arg_freetext" in l: return "G6_OPERAND"
    if "wrong_op" in l: return "Xop_noise"
    return "Xother"

def dom_of(fn):
    for d in ("retail","airline","telecom","banking"):
        if fn.endswith(f"_{d}"): return d
    return "?"

def main(d):
    files=sorted(glob.glob(os.path.join(d,"*.json")))
    pooled=defaultdict(Counter)   # domain -> G -> count
    poolfails=Counter()           # domain -> total fails
    permodel=defaultdict(list)    # domain -> [(model, pass, {G:%})]
    for f in files:
        lbl=os.path.basename(f).replace(".json","")
        dom=dom_of(lbl)
        if dom=="?": continue
        try:
            _,leaf,nf=F.decomp(f,lbl)
        except Exception as e:
            print("ERR",lbl,e); continue
        if nf==0: continue
        gc=Counter()
        for k,v in leaf.items(): gc[gmap(k)]+=v
        pooled[dom].update(gc); poolfails[dom]+=nf
    print("\n\n############ 도메인별 POOLED G-분포 (전 모델) ############")
    Gs=["G1_COVERAGE","G2_REACH","G3_VERIFY","G4_PERSISTENCE","G5_SCOPE","G6_OPERAND","G7_REFERENCE"]
    hdr="{:16}".format("function")+"".join(f"{d[:8]:>10}" for d in ("retail","airline","telecom","banking"))
    print(hdr)
    for g in Gs:
        row=f"{g:16}"
        for dom in ("retail","airline","telecom","banking"):
            nf=poolfails[dom]; row+=f"{(100*pooled[dom][g]/nf if nf else 0):>9.1f}%"
        print(row)
    print("\nfails/domain:", dict(poolfails))
    # 불변성 판정
    print("\n★도메인-불변(전 도메인 ≥5%): ", end="")
    inv=[g for g in Gs if all((100*pooled[dom][g]/poolfails[dom] if poolfails[dom] else 0)>=5 for dom in ("retail","airline","banking"))]
    print(inv)

if __name__=="__main__":
    main(sys.argv[1] if len(sys.argv)>1 else r"C:/tmp/traj")
