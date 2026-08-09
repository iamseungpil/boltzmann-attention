# -*- coding: utf-8 -*-
r"""x177 — 오름차순과 내림차순의 **층별 값**을 나란히 본다 (유료 0·사용자 발의).

## 왜 (최소 대응쌍이 생겼다)

C356: 같은 25행을 정렬 규칙만 바꿔 재배열하면 `name_asc` 는 오답, `name_desc` 는 정답이다.
**내용·이름·개수·후보목록이 전부 같고 순서만 다르다** — 층별 렌즈에 넣을 수 있는 가장 깨끗한
입력이다. 기전 설명 아홉 개가 통제로 죽은 자리에서, 이제 **내부를 본다**.

## 답할 수 있는 것 (C349 가 정한 범위)

후보 공간은 **마지막 ~5층에서만** 읽힌다(32B 60/64 · 14B 44/48). 그 앞은 후보 질량이 0.0001
수준이라 비교하면 잡음을 비교하는 것이다. 그래서 이 프로브의 질문은 정확히 하나다:

    **읽기 창 진입 시점에 이미 오답이 앞서 있는가, 아니면 창 안에서 갈리는가.**

전자면 분기는 그 전에 일어났고(다음은 attention 축) · 후자면 읽기 단계가 순서에 반응한다.

## 겸하는 것

층별은 HF 포워드가 필요하고 32B 는 GPTQ 라 못 읽는다(bf16 은 CPU 오프로드로 가능·느림).
**14B 는 GPU 에 통째로 들어간다** ⇒ 이 런은 **복제**도 겸한다 — 14B 최종 층의 답이 곧
*"다른 모델에서도 오름차순이 지는가"* 이다. 재현 안 되면 층별 비교는 무의미해지고 대신
**32B 특유**라는 결과를 얻는다. 어느 쪽이든 값이 있다.

⚠계기 한계(C356·토큰): 후보 첫 토큰이 1글자면(`'L'`) 여러 후보에 걸친다. 여기서는 후보마다
`tok.encode(c)[0]` 을 쓰므로 **접두 그룹 문제가 남는다** — 그래서 **후보질량 열을 함께** 내고,
질량이 실재하는 구간(≥0.1)에서만 판정한다.

실행: CUDA_VISIBLE_DEVICES=0 py -3 x177_order_lens.py <MODEL_PATH> [ORDERS]
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch                                                   # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer    # noqa: E402

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

TASK = "task_099"
GOLD, WRONG = "World Blue", "Lime Green"


def entropy(ps):
    tot = sum(ps) or 1.0
    return -sum((p / tot) * math.log(p / tot) for p in ps if p > 0)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    want = (sys.argv[2] if len(sys.argv) > 2 else "name_asc,name_desc").split(",")

    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    lines = LG.eligible_text(730, {}, maps, spec,
                             {"qualifying_deposit_usd": 30000}).strip().splitlines()
    head = [l for l in lines if not (l.startswith("  ") and ":" in l)]
    body = [l for l in lines if l.startswith("  ") and ":" in l]
    name = lambda l: l.strip().split(":")[0].strip()          # noqa: E731
    MS = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), TASK)
    pre = "Here is a customer-service conversation so far.\n\n" + Y.render(MS) + "\n\n"
    orders = {"name_asc": sorted(body, key=name),
              "name_desc": sorted(body, key=name, reverse=True)}

    tok = AutoTokenizer.from_pretrained(path)
    dm = os.environ.get("T2_LENS_DEVMAP", "cuda:0")
    kw = {"max_memory": {0: os.environ.get("T2_LENS_GPU_MEM", "36GiB"), "cpu": "300GiB"}} \
        if dm == "auto" else {}
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 device_map=dm, **kw)
    model.eval()
    L = model.config.num_hidden_layers
    norm, headm = model.model.norm, model.lm_head
    first = {}
    for c in [name(l) for l in body]:
        ids = tok.encode(c, add_special_tokens=False)
        if ids:
            first.setdefault(ids[0], []).append(c)
    ids = sorted(first)
    gid = next((i for i in ids if any(c.startswith(GOLD.split()[0]) for c in first[i])), None)
    wid = next((i for i in ids if any(c.startswith(WRONG.split()[0]) for c in first[i])), None)
    print("model=%s layers=%d · 후보 %d → 첫토큰 버킷 %d" % (path, L, len(body), len(ids)))
    print("  gold 버킷=%s · wrong 버킷=%s" % (first.get(gid), first.get(wid)))

    for oname in want:
        order = orders[oname]
        nm = [name(l) for l in order]
        tbl = "\n".join(head[:1] + order + head[1:]).strip()
        prompt = pre + tbl + "\n\n" + X.FACTS[TASK] + "\n\n" + X.QUESTION
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            try:
                out = model(**enc, output_hidden_states=True, logits_to_keep=1)
            except TypeError:
                out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        print("\n=== %s · gold자리 %d/%d · wrong자리 %d/%d · 토큰 %d ==="
              % (oname, nm.index(GOLD) + 1, len(nm), nm.index(WRONG) + 1, len(nm),
                 enc["input_ids"].shape[1]))
        print("%-5s %9s %9s %9s %8s  %s" % ("층", "p(gold)", "p(wrong)", "후보질량", "H", "argmax"))
        for li in range(1, len(hs)):
            if li < L - 7 and li % max(1, L // 8):
                continue
            h = hs[li][0, -1, :]
            with torch.no_grad():
                lg = headm(norm(h)).float().cpu()
            sub = torch.softmax(lg[ids], dim=-1)
            cmass = float(torch.softmax(lg, dim=-1)[ids].sum())
            pg = float(sub[ids.index(gid)]) if gid in ids else float("nan")
            pw = float(sub[ids.index(wid)]) if wid in ids else float("nan")
            top = first[ids[int(torch.argmax(sub))]][0]
            print("%-5d %9.4f %9.4f %9.4f %8.3f  %s%s"
                  % (li, pg, pw, cmass, entropy([float(x) for x in sub]), top,
                     "  ← 읽기창" if cmass >= 0.1 else ""))
    print("\n  질량 ≥0.1 인 구간에서만 판정한다. 진입 시점에 이미 갈려 있으면 분기는 그 전이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
