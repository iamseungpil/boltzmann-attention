#!/usr/bin/env python3
"""X8-(b) triage 프로브 — formalize 기능 2종(user_acts·slots)의 실패 원인 3분기 (2026-07-30).

`EXPERIMENT_PLAN_PATENT_PAPERS_2026_07_30.md` §1 X8 절차:
  ⑴ 선언-강제 시험 (arm A → arm C 에서 실패가 **소멸**하면 = 인터페이스-유발)
  ⑵ 잔존 사례의 정보-맞춘 격리 replay ([[18]] · 해소되면 = 부하)
  ⑶ 잔존 = 능력/경계 → learn 표적 스펙 등재

arms:
  A  = prompt-only 자유서술 (선언 강제 없음·느슨한 파싱)
  C  = 선언 봉투(guided_json) + **결정론 검증기** + 위반 시 regen
       검증기 술어(전부 **닫힌** 술어·[[22]]):
         V1 acts ⊆ 허용 8종
         V2 acts 비어있지 않음
         V3 slots 각 항목이 발화의 **축자 부분문자열**  ← 날조 차단
         V4 중복 slot 없음
  Actx = A + 직전 에이전트 턴 문맥 ([[18]] B_fullctx 근사·⑵용)

★[[03b]]: gold는 사람이 만든 `x8_gold_labels.jsonl`이다. 이 스크립트는 gold를 만들지 않는다.
★[[30]] 구멍 교정: X4·X5가 stdout만 내서 원시 소실 → **항상 rows 파일로 쓴다**.

용법: py -3 x8b_triage_probe.py --arms A,C,Actx --seeds 3 --out rows.jsonl
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
PORTS = {"7B": ("http://localhost:8142/v1", "Qwen/Qwen2.5-7B-Instruct"),
         "32B": ("http://localhost:8140/v1", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")}

ACTS = ["CONTROL", "CLOSE", "REFUSE", "REPORT", "PROVIDE", "REQUEST", "ASK", "CONFIRM"]

# 라벨 규약(X8_GOLD_LABEL_PROTOCOL §2·§3)의 프롬프트판 — gold와 **같은** 규약을 모델에도 준다
# (규약을 안 주고 재면 규약 불일치를 능력 결손으로 오귀속한다).
RUBRIC = """You classify a single customer utterance from a banking support chat.

TASK 1 — acts: which speech acts does this utterance perform? Choose ALL that apply from:
  CONTROL  the utterance is ONLY a control token (###STOP###/###TRANSFER###/###OUT-OF-SCOPE###)
  CLOSE    signals completion / no further action needed
  REFUSE   declines to supply requested info, or rejects the proposed path
  REPORT   reports the outcome of an action THE CUSTOMER executed themselves
  PROVIDE  supplies one or more operand values
  REQUEST  asks the agent to perform an action (look up, file, transfer, apply)
  ASK      asks for information only (an answer, not a state change)
  CONFIRM  affirms a proposed action

TASK 2 — slots: the operand VALUES this utterance supplies, copied VERBATIM.
  INCLUDE ids (txn_/dsp_/user ids, tool names), emails, phone numbers, addresses, person
    names, money amounts, percentages, durations, credit scores, incomes, enum choices
    (card names, Yes/No), status values.
  EXCLUDE field NAMES ("phone number", "promo/offer ID") - naming a field is not a value.
  EXCLUDE emphasis ("eighth time", "physical flyer") - emphasis is not a value.
  EXCLUDE the customer's own counts/claims ("four friends", "two bonuses").
  EXCLUDE ranking criteria ("lowest annual fee", "best bonus").
  Note: markdown bold in the utterance marks emphasis AND field names too, so bold is NOT
  the answer - decide by whether the span is an operand value."""

ENVELOPE = {
    "type": "object",
    "properties": {
        "acts": {"type": "array", "items": {"type": "string", "enum": ACTS}, "minItems": 1},
        "slots": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["acts", "slots"],
    "additionalProperties": False,
}


def call(base, model, msgs, seed, max_tokens=400, guided=None):
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.0, "seed": seed}
    if guided:
        body["guided_json"] = guided
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"] or ""


def verify(pred, text):
    """arm C 결정론 검증기 — 닫힌 술어만([[22]]). 반환 위반 목록."""
    v = []
    acts = pred.get("acts")
    slots = pred.get("slots")
    if not isinstance(acts, list) or not acts:
        v.append("V2_empty_acts")
        acts = acts if isinstance(acts, list) else []
    if any(a not in ACTS for a in acts):
        v.append("V1_unknown_act")
    if not isinstance(slots, list):
        v.append("V3_slots_not_list")
        slots = []
    norm = re.sub(r"\s+", " ", text)
    for s in slots:
        if not isinstance(s, str) or re.sub(r"\s+", " ", s) not in norm:
            v.append("V3_not_verbatim")
            break
    if len(slots) != len(set(map(str, slots))):
        v.append("V4_dup_slot")
    return v


def parse_free(out):
    """arm A 느슨한 파싱: JSON 블록이 있으면 그걸, 없으면 라인 휴리스틱."""
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return {"acts": d.get("acts") or d.get("act") or [],
                        "slots": d.get("slots") or d.get("values") or []}, False
        except Exception:
            pass
    acts = [a for a in ACTS if re.search(rf"\b{a}\b", out)]
    slots = re.findall(r"[-*]\s*[\"']?([^\"'\n]{2,60})[\"']?\s*$", out, re.M)
    return {"acts": acts, "slots": [s.strip() for s in slots]}, True


def score(pred, gold_acts, gold_slots, text):
    ga, gs = set(gold_acts), set(gold_slots)
    pa = {a for a in (pred.get("acts") or []) if isinstance(a, str)}
    ps = {str(s).strip() for s in (pred.get("slots") or []) if str(s).strip()}
    return {
        "acts_exact": pa == ga,
        "acts_tp": len(pa & ga), "acts_fp": len(pa - ga), "acts_fn": len(ga - pa),
        "slots_exact": ps == gs,
        "slots_tp": len(ps & gs), "slots_fp": len(ps - gs), "slots_fn": len(gs - ps),
        "slot_fabricated": sum(1 for s in ps
                               if re.sub(r"\s+", " ", s) not in re.sub(r"\s+", " ", text)),
        "pred_acts": sorted(pa), "pred_slots": sorted(ps),
    }


def prf(rows, k):
    tp = sum(r[f"{k}_tp"] for r in rows)
    fp = sum(r[f"{k}_fp"] for r in rows)
    fn = sum(r[f"{k}_fn"] for r in rows)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0), tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7B", choices=list(PORTS))
    ap.add_argument("--arms", default="A,C,Actx")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--regen-cap", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(_SIM, "x8_triage_rows.jsonl"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    samp = [json.loads(l) for l in open(os.path.join(_SIM, "x8_sample_utterances.jsonl"),
                                        encoding="utf-8")]
    gold = {json.loads(l)["sample_id"]: json.loads(l)
            for l in open(os.path.join(_SIM, "x8_gold_labels.jsonl"), encoding="utf-8")}
    if args.limit:
        samp = samp[:args.limit]
    base, model = PORTS[args.model]
    arms = args.arms.split(",")
    print(f"model={args.model} arms={arms} seeds={args.seeds} n={len(samp)}")

    rows = []
    for arm in arms:
        for seed in range(args.seeds):
            for r in samp:
                g = gold[r["sample_id"]]
                ctx = ""
                if arm == "Actx" and r.get("prev_agent"):
                    ctx = f"\n=== PRECEDING AGENT TURN ===\n{r['prev_agent'][:1200]}\n"
                user = (RUBRIC + ctx + "\n=== CUSTOMER UTTERANCE ===\n" + r["text"]
                        + ("\n\nAnswer as JSON: {\"acts\": [...], \"slots\": [...]}"
                           if arm != "C" else ""))
                msgs = [{"role": "user", "content": user}]
                regen, viol_hist = 0, []
                try:
                    if arm == "C":
                        out = call(base, model, msgs, seed, guided=ENVELOPE)
                        pred = json.loads(out)
                        v = verify(pred, r["text"])
                        while v and regen < args.regen_cap:
                            regen += 1
                            viol_hist.append(list(v))
                            fb = ("Your previous answer violated: " + ", ".join(v)
                                  + ". Every slot must be copied VERBATIM from the utterance. "
                                    "Re-answer.")
                            out = call(base, model, msgs + [
                                {"role": "assistant", "content": out},
                                {"role": "user", "content": fb}], seed, guided=ENVELOPE)
                            pred = json.loads(out)
                            v = verify(pred, r["text"])
                        loose = False
                    else:
                        out = call(base, model, msgs, seed)
                        pred, loose = parse_free(out)
                        v = verify(pred, r["text"])
                except Exception as e:
                    rows.append({"arm": arm, "seed": seed, "sample_id": r["sample_id"],
                                 "error": str(e)[:200]})
                    print(f"  [err] {arm} s{seed} {r['sample_id']}: {str(e)[:90]}")
                    continue
                sc = score(pred, g["acts"], g["slots"], r["text"])
                rows.append(dict(arm=arm, seed=seed, sample_id=r["sample_id"],
                                 day=r["day"], task_id=r["task_id"], pos=r["pos"],
                                 gold_acts=g["acts"], gold_slots=g["slots"],
                                 residual_viol=v, regen_used=regen,
                                 viol_history=viol_hist, loose_parsed=loose, **sc))
            done = [x for x in rows if x.get("arm") == arm and x.get("seed") == seed
                    and "error" not in x]
            if done:
                ae = sum(x["acts_exact"] for x in done) / len(done)
                se = sum(x["slots_exact"] for x in done) / len(done)
                print(f"  {arm} seed{seed}: acts_exact={ae:.2f} slots_exact={se:.2f} "
                      f"n={len(done)}")

    with open(args.out, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"\n[saved] {args.out}  ({len(rows)} rows)")

    print("\n" + "=" * 72)
    print("arm 요약 (⑴선언-강제 시험)")
    print("=" * 72)
    for arm in arms:
        ok = [x for x in rows if x.get("arm") == arm and "error" not in x]
        if not ok:
            continue
        aP, aR, aF, *_ = prf(ok, "acts")
        sP, sR, sF, stp, sfp, sfn = prf(ok, "slots")
        print(f"{arm:5s} n={len(ok):3d}  acts: exact={sum(x['acts_exact'] for x in ok) / len(ok):.2f} "
              f"F1={aF:.2f}   slots: exact={sum(x['slots_exact'] for x in ok) / len(ok):.2f} "
              f"P={sP:.2f} R={sR:.2f} F1={sF:.2f}")
        print(f"        날조 slot {sum(x['slot_fabricated'] for x in ok)}개 · "
              f"잔존 위반 {sum(1 for x in ok if x['residual_viol'])}건 · "
              f"regen {sum(x['regen_used'] for x in ok)}회 · "
              f"느슨파싱 {sum(1 for x in ok if x.get('loose_parsed'))}건")
        vc = Counter(v for x in ok for vh in x.get("viol_history") or [] for v in vh)
        if vc:
            print(f"        regen 유발 위반: {dict(vc)}")
    print("\n⚠[[08]]: arm 집계로 결론 금지 — 다음 단계는 per-case 교차표(어느 발화가 어느 arm서 "
          "고쳐졌나)와 잔존 사례 정독이다.")


if __name__ == "__main__":
    main()
