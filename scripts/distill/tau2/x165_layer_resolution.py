# -*- coding: utf-8 -*-
r"""x165 — **해소 깊이 ℓ\***: 답이 몇 번째 층에서 정해지는가 (유료 0·로컬 HF 포워드).

## 왜 (C347 → 이 실험)

C347: 14B 와 32B 는 **폭이 같은데**(hidden 5120·heads 40·kv 8·head_dim 128) j\* 가 최소 5배
다르다 ⇒ *용량 ∝ 표현 차원* 기각. 남은 축은 **깊이**(48 대 64)와 MLP 폭이다. 깊이 읽기는
*"층 하나 = Hopfield 업데이트 한 번"* 이므로 32B 가 버티는 이유는 더 많이 **저장**해서가 아니라
혼합을 더 여러 번 **정제**해서다 — 용량이 아니라 **정제 예산**.

## 검정 가능한 형태로

    j\* = 곡선 ℓ\*(j) 가 층 수 L 에 도달하는 j

j 가 커질수록 해소에 층이 더 들고, **예산 L 을 넘기는 순간 붕괴**한다. 그러면 32B(64)와
14B(48)의 j\* 차이는 **곡선이 겹치고 L 만 다르면** 깊이만으로 설명된다.

  P6a  j<j\* 에서 j↑ ⇒ ℓ\* 단조 증가
  P6b  j=j\* 에서 ℓ\* 가 L 을 넘김 — 또는 **이른 층에서 오답 고정 후 후반이 못 되돌림**
       (둘은 다른 그림이고 층별 궤적이 가른다)
  P6c  같은 j 에서 14B 의 ℓ\*/L 가 32B 보다 크다(여유가 적다)
  P6d  곡선이 겹치고 L 만 다르면 깊이 확정 · 곡선 자체가 다르면 폭·MLP 교란이 살아 있다

## 계기와 그 한계 (미리 적는다·[[08]])

 · **logit lens**: 층 ℓ 의 마지막 위치 잔차를 **최종 norm 통과 후** unembed 한다. 초기 층의
   신뢰도가 낮은 것은 알려진 한계다(tuned lens 는 학습이 필요) — 초기 층 수치는 [D] 로 읽는다.
 · **후보 제약**: vLLM `guided_choice` 의 HF 대응이 없으므로 **후보 첫 토큰 id 집합**으로 좁힌다.
   여러 후보가 첫 토큰을 공유하면 **버킷**이 된다(x163 과 같은 성질·버킷 수를 함께 출력).
 · **부정통제**: 같은 j 를 두 번 돌린다. greedy·같은 입력이므로 **완전 동일**해야 한다 —
   다르면 계기가 흔들린 것이지 결과가 아니다([[57]]).
 · attention 엔트로피(진짜 order parameter)는 **이 판에 넣지 않는다** — GQA·RoPE 를 손으로
   재현하면 조용히 틀리기 쉽다. 별도로 짧은 프롬프트에서 `output_attentions` 와 대조해
   계기를 먼저 검증한 뒤 붙인다.

실행: CUDA_VISIBLE_DEVICES=0 py -3 x165_layer_resolution.py <MODEL_PATH> [JLIST]
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
import x157_entrainment_lambda as P                            # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402


def build_prompts(js):
    a2 = load_domain_a2("banking_knowledge")
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[P.TASK] + "\n\n" + X.QUESTION
    choices = [l.strip().split(":")[0].strip() for l in table.splitlines() if l.startswith("  ")]
    msgs = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), P.TASK)
    out = []
    for j in js:
        pre = ("Here is a customer-service conversation so far.\n\n"
               + Y.render(msgs[:j]) + "\n\n") if j else ""
        out.append((j, pre + base))
    return out, choices, len(msgs)


def entropy(ps):
    tot = sum(ps) or 1.0
    return -sum((p / tot) * math.log(p / tot) for p in ps if p > 0)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    js = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,13,26,27").split(",")]

    tok = AutoTokenizer.from_pretrained(path)
    # ⚠32B 는 **GPTQ-Int8 을 HF 로 못 읽는다**(gptqmodel/optimum 미설치 · 설치하면 torch 2.7→2.13,
    #   transformers 4.51→5.14 로 공유 env 를 갈아엎고, `gptqmodel<3` 은 빌드 실패). 그래서 bf16 을
    #   GPU+CPU 오프로드로 올린다(`T2_LENS_DEVMAP=auto`). **동등하다고 가정하지 않는다** —
    #   Int8 랜드마크(j=26 gold · j=27 붕괴)를 최종 층이 재현하는지로 교란을 **잰다**.
    dm = os.environ.get("T2_LENS_DEVMAP", "cuda:0")
    kw = {}
    if dm == "auto":
        # ⚠상한을 안 주면 accelerate 가 GPU 를 45.6/47.4GiB 까지 채우고, 그 뒤 `lm_head` 가중치를
        #   올릴 1.45GiB 가 없어 OOM 으로 죽는다(2026-08-09 실측). 활성화 + 가중치 스왑 자리를 남긴다.
        kw["max_memory"] = {0: os.environ.get("T2_LENS_GPU_MEM", "36GiB"), "cpu": "300GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=dm, **kw)
    model.eval()
    L = model.config.num_hidden_layers
    print("model=%s  layers=%d  hidden=%d" % (path, L, model.config.hidden_size))

    prompts, choices, ntraj = build_prompts(js)
    # 후보 → 첫 토큰 id (공백 선행형도 함께 본다 — 채팅 템플릿 뒤에서는 보통 선행 공백이 없다)
    first = {}
    for c in choices:
        ids = tok.encode(c, add_special_tokens=False)
        if ids:
            first.setdefault(ids[0], []).append(c)
    ids = sorted(first)
    gold_id = next((i for i in ids if any(c.startswith(P.GOLD_HEAD) for c in first[i])), None)
    print("후보 %d → 첫-토큰 버킷 %d · gold 버킷 id=%s (%s)"
          % (len(choices), len(ids), gold_id, first.get(gold_id)))
    print("궤적 %d 메시지" % ntraj)

    norm = model.model.norm
    head = model.lm_head

    for j, prompt in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            # 모델 자신의 logits 는 안 쓴다(렌즈를 우리가 돌린다). 전 위치 logits 는 3348×152k
            # ≈ 2GB 낭비이므로 마지막 한 자리만 계산시킨다.
            try:
                out = model(**enc, output_hidden_states=True, logits_to_keep=1)
            except TypeError:
                out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states                      # (L+1) × [1, seq, hidden]
        print("\n=== j=%d · 토큰 %d ===" % (j, enc["input_ids"].shape[1]))
        print("%-6s %10s %9s %9s  %s" % ("층", "p(gold)", "H(nats)", "후보질량", "argmax"))
        lstar, prev_ok = None, False
        for li in range(1, len(hs)):
            # ⚠오프로드 시 `norm`/`lm_head` 의 파라미터는 **meta** 에 있다. 거기로 h 를 미리 옮기면
            #   `Cannot copy out of meta tensor` 로 죽는다(2026-08-09 실측) — accelerate 훅이
            #   호출 시점에 실행 디바이스로 옮겨 주므로 **그대로 넘긴다**.
            h = hs[li][0, -1, :]
            with torch.no_grad():
                logits = head(norm(h)).float().cpu()
            sub = logits[ids]
            p = torch.softmax(sub, dim=-1)
            # ★계기 검정: 후보 집합이 **어휘 전체 질량 중 몇 %**를 갖는가. 이 값이 작으면
            #   "후보들 사이의 혼합"은 우리 재정규화가 만든 그림이고, 중간 층은 그저 아직
            #   출력 기저로 읽히지 않는 것이다(logit lens 의 알려진 한계). 숨기지 않고 낸다.
            cmass = float(torch.softmax(logits, dim=-1)[ids].sum())
            top = int(torch.argmax(p))
            pg = float(p[ids.index(gold_id)]) if gold_id in ids else float("nan")
            ok = (ids[top] == gold_id)
            # ℓ* = gold 가 argmax 가 되어 **끝까지 유지되는** 최초 층
            if ok and not prev_ok:
                lstar = li
            if not ok:
                lstar = None
            prev_ok = ok
            if li % max(1, L // 12) == 0 or li >= L - 3 or li <= 2:
                print("%-6d %10.4f %9.3f %9.4f  %s%s"
                      % (li, pg, entropy([float(x) for x in p]), cmass,
                         first[ids[top]][0], "  ✓" if ok else ""))
        print("  ⇒ ℓ* = %s / L=%d   (%s)"
              % (lstar if lstar else "없음(끝까지 gold 아님)", L,
                 "%.2f" % (lstar / L) if lstar else "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
