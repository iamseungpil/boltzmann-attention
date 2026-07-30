#!/usr/bin/env python3
"""X8-(f) triage 프로브 v2 — 개정 9라벨 + 변동 측정 + 최소-스팬 검증기 arm (2026-07-30).

v1(`x8b_triage_probe.py`) 대비 3가지 수정:

  ① **개정 규약 9라벨**(`x8_gold_labels_v2.jsonl`) — `ASK`→`REQ_INFO`(우리 ASK와 이름 충돌 해소),
     `REQUEST`→`REQ_ACT`+`ESCALATE`(이관은 전용 결정론 경로를 타므로 별도). v1 §1-b 재채점이
     이 경계 모호가 정확도 6~10pp를 깎았음을 실측했다.
  ② **`temperature>0`** — v1은 temp 0 + seed 3 이라 144조합 중 138이 **예측 완전동일**해서 변동
     정보가 0이었다(호출 2/3 낭비). 이제 시드가 실제 변동을 준다.
  ③ **arm D 신설 = 타입 선언 + 형식-적합 검증기** — v1 축 (b) 직격.
     v1 arm C의 V3는 "축자 부분문자열"만 봐서 `'Annual income is $95,000'`도 **통과시킨다**.
     모델이 `{value, type}`을 선언하게 하고 검증기가 **type 패턴 fullmatch**를 요구하면
     최소-스팬이 강제된다. 이건 **형식** 판정이라 닫힌 술어이고([[22]]), **의도 판정이 아니다**
     — 의도(어느 act인가)는 v1 자기정정 2의 교훈대로 **LLM 몫**으로 남긴다.
     타입 어휘는 도메인-일반 **값 형식**(id/email/phone/money/percent/number/date/duration)이라
     도메인 리터럴이 아니다. 열린 형식(name/enum/other)은 **패턴 검사 면제**(정직한 범위 제한).

arms: A(prompt-only) · C(선언 봉투+축자 검증기) · D(선언+**타입**+형식 검증기) · Actx(A+직전 턴)

용법: py -3 x8f_triage_probe_v2.py --model 7B --arms A,C,D --seeds 4 --temp 0.7
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
PORTS = {"7B": ("http://localhost:8142/v1", "Qwen/Qwen2.5-7B-Instruct"),
         "14B": ("http://localhost:8143/v1", "Qwen/Qwen2.5-14B-Instruct"),
         "32B": ("http://localhost:8140/v1", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")}

ACTS = ["CONTROL", "CLOSE", "REFUSE", "REPORT", "PROVIDE",
        "REQ_ACT", "ESCALATE", "REQ_INFO", "CONFIRM"]

# 값 **형식** 타입 — 도메인-일반(은행/소매/항공 무관). fullmatch 패턴이 있는 것만 검증 대상.
TYPES = {
    "id":       r"[a-z]{2,4}_[0-9a-f]{6,}",
    "email":    r"[\w.+\-]+@[\w\-]+\.\w{2,}",
    "phone":    r"\+?\d[\d\s\-()]{7,}\d",
    "money":    r"[-+]?\$[\d,]+(?:\.\d{1,2})?",
    "percent":  r"[-+]?\d+(?:\.\d+)?\s?%",
    "number":   r"[-+]?[\d,]+(?:\.\d+)?",
    "duration": r"\d+(?:\s?[-–]\s?\d+)?\s*(?:business\s+)?(?:day|days|month|months|year|years|hour|hours|week|weeks)",
    # 열린 형식 — 패턴 검사 면제(honest scoping)
    "name": None, "address": None, "enum": None, "other": None,
}

RUBRIC = """You classify a single customer utterance from a banking support chat.

TASK 1 - acts: which speech acts does this utterance perform? Choose ALL that apply:
  CONTROL   the utterance is ONLY a control token (###STOP###/###TRANSFER###/###OUT-OF-SCOPE###)
  CLOSE     signals completion / no further action needed
  REFUSE    declines to supply requested info, or rejects the proposed path
  REPORT    reports the outcome of an action THE CUSTOMER executed themselves
  PROVIDE   supplies one or more operand values
  REQ_ACT   asks the agent to perform an action (look up, file, apply, fix)
  ESCALATE  asks to be transferred/escalated to a human agent
  REQ_INFO  asks for information only (an answer, not a state change)
  CONFIRM   affirms a proposed action
Note ESCALATE is separate from REQ_ACT. An utterance may carry both (e.g. "do X, or else
transfer me"). Most real utterances carry TWO OR MORE acts - do not stop at one.

TASK 2 - slots: the operand VALUES this utterance supplies, copied VERBATIM.
  INCLUDE ids, emails, phone numbers, addresses, person names, money amounts, percentages,
    durations, credit scores, incomes, enum choices (card names, Yes/No), status values.
  EXCLUDE field NAMES ("phone number", "promo/offer ID") - naming a field is not a value.
  EXCLUDE emphasis ("eighth time", "physical flyer") - emphasis is not a value.
  EXCLUDE the customer's own counts/claims ("four friends").
  EXCLUDE ranking criteria ("lowest annual fee").
  Markdown bold marks emphasis and field names too, so bold is NOT the answer.
  **Give the SHORTEST span that is the value itself**: for "Annual income is $95,000" the
  slot is "$95,000", NOT "Annual income is $95,000"."""

TYPE_NOTE = """
Each slot must also carry a "type" from: id, email, phone, money, percent, number,
duration, name, address, enum, other. The value must be EXACTLY a value of that type -
no surrounding words, no labels, no punctuation."""

ENV_C = {
    "type": "object",
    "properties": {
        "acts": {"type": "array", "items": {"type": "string", "enum": ACTS}, "minItems": 1},
        "slots": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["acts", "slots"], "additionalProperties": False,
}
ENV_D = {
    "type": "object",
    "properties": {
        "acts": {"type": "array", "items": {"type": "string", "enum": ACTS}, "minItems": 1},
        "slots": {"type": "array", "items": {
            "type": "object",
            "properties": {"value": {"type": "string"},
                           "type": {"type": "string", "enum": sorted(TYPES)}},
            "required": ["value", "type"], "additionalProperties": False}},
    },
    "required": ["acts", "slots"], "additionalProperties": False,
}


def call(base, model, msgs, seed, temp, guided=None, max_tokens=520):
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens,
            "temperature": temp, "seed": seed}
    if guided:
        body["guided_json"] = guided
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


def norm(t):
    return re.sub(r"\s+", " ", str(t)).strip()


def verify(acts, slots, text, typed):
    """결정론 검증기 — 닫힌 술어만. slots = [str] 또는 [{value,type}]."""
    v = []
    if not isinstance(acts, list) or not acts:
        v.append("V2_empty_acts")
        acts = acts if isinstance(acts, list) else []
    if any(a not in ACTS for a in acts):
        v.append("V1_unknown_act")
    nt = norm(text)
    vals = []
    for s in (slots or []):
        if typed:
            if not isinstance(s, dict) or "value" not in s or "type" not in s:
                v.append("V6_bad_slot_shape")
                continue
            val, ty = str(s["value"]), s["type"]
            vals.append(val)
            pat = TYPES.get(ty, "SENTINEL")
            if pat == "SENTINEL":
                v.append("V7_unknown_type")
            elif pat and not re.fullmatch(pat, val.strip(), re.I):
                v.append("V5_type_mismatch")          # ★최소-스팬 강제
        else:
            vals.append(str(s))
    for val in vals:
        if norm(val) not in nt:
            v.append("V3_not_verbatim")
            break
    if len(vals) != len(set(map(norm, vals))):
        v.append("V4_dup_slot")
    return sorted(set(v)), vals


def parse_free(out):
    m = re.search(r"\{.*\}", out, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                sl = d.get("slots") or d.get("values") or []
                sl = [x.get("value") if isinstance(x, dict) else x for x in sl]
                return (d.get("acts") or []), sl
        except Exception:
            pass
    return [a for a in ACTS if re.search(rf"\b{a}\b", out)], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7B")
    ap.add_argument("--arms", default="A,C,D,Actx")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--regen-cap", type=int, default=2)
    ap.add_argument("--out", default="")
    # ★2026-07-30 야간: 8140은 Y1 본런 점유 → 32B 스케일 축은 GPU1의 8141로 돌린다(포트만 교체).
    ap.add_argument("--base_url", default="", help="PORTS 기본 endpoint 덮어쓰기(예: http://localhost:8141/v1)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    out_path = args.out or os.path.join(_SIM, f"x8v2_rows_{args.model}.jsonl")

    samp = [json.loads(l) for l in open(os.path.join(_SIM, "x8_sample_utterances.jsonl"),
                                        encoding="utf-8")]
    gold = {json.loads(l)["sample_id"]: json.loads(l)
            for l in open(os.path.join(_SIM, "x8_gold_labels_v2.jsonl"), encoding="utf-8")}
    base, model = PORTS[args.model]
    if args.base_url:
        base = args.base_url.rstrip("/")
    arms = args.arms.split(",")
    print(f"model={args.model} endpoint={base} arms={arms} seeds={args.seeds} "
          f"temp={args.temp} n={len(samp)}")

    rows = []
    for arm in arms:
        for seed in range(args.seeds):
            for r in samp:
                g = gold[r["sample_id"]]
                typed = (arm == "D")
                ctx = (f"\n=== PRECEDING AGENT TURN ===\n{r.get('prev_agent', '')[:1200]}\n"
                       if arm == "Actx" else "")
                user = (RUBRIC + (TYPE_NOTE if typed else "") + ctx
                        + "\n=== CUSTOMER UTTERANCE ===\n" + r["text"]
                        + ("\n\nAnswer as JSON: {\"acts\": [...], \"slots\": [...]}"
                           if arm in ("A", "Actx") else ""))
                msgs = [{"role": "user", "content": user}]
                env = ENV_D if typed else (ENV_C if arm == "C" else None)
                regen, hist = 0, []
                try:
                    if env is not None:
                        o = call(base, model, msgs, seed, args.temp, guided=env)
                        d = json.loads(o)
                        acts, sl = d.get("acts") or [], d.get("slots") or []
                        v, vals = verify(acts, sl, r["text"], typed)
                        while v and regen < args.regen_cap:
                            regen += 1
                            hist.append(v)
                            fb = ("Your previous answer violated: " + ", ".join(v)
                                  + ". Slots must be copied VERBATIM and be the SHORTEST span "
                                    "that is the value itself, matching the declared type. "
                                    "Re-answer.")
                            o = call(base, model, msgs + [{"role": "assistant", "content": o},
                                                          {"role": "user", "content": fb}],
                                     seed, args.temp, guided=env)
                            d = json.loads(o)
                            acts, sl = d.get("acts") or [], d.get("slots") or []
                            v, vals = verify(acts, sl, r["text"], typed)
                    else:
                        o = call(base, model, msgs, seed, args.temp)
                        acts, sl = parse_free(o)
                        v, vals = verify(acts, sl, r["text"], False)
                except Exception as e:
                    rows.append({"arm": arm, "seed": seed, "sample_id": r["sample_id"],
                                 "error": str(e)[:200]})
                    continue
                ga, gs = set(g["acts"]), set(g["slots"])
                pa = {a for a in acts if isinstance(a, str)}
                ps = {norm(x) for x in vals if norm(x)}
                gsn = {norm(x) for x in gs}
                rows.append(dict(
                    arm=arm, seed=seed, endpoint=base, model=model,
                    sample_id=r["sample_id"], day=r["day"],
                    task_id=r["task_id"], pos=r["pos"],
                    gold_acts=sorted(ga), gold_slots=sorted(gs),
                    pred_acts=sorted(pa), pred_slots=sorted(ps),
                    acts_exact=(pa == ga),
                    acts_tp=len(pa & ga), acts_fp=len(pa - ga), acts_fn=len(ga - pa),
                    slots_exact=(ps == gsn),
                    slots_tp=len(ps & gsn), slots_fp=len(ps - gsn), slots_fn=len(gsn - ps),
                    residual_viol=v, regen_used=regen, viol_history=hist))
            ok = [x for x in rows if x.get("arm") == arm and x.get("seed") == seed
                  and "error" not in x]
            if ok:
                print(f"  {arm} s{seed}: acts_exact={sum(x['acts_exact'] for x in ok) / len(ok):.2f} "
                      f"slots_exact={sum(x['slots_exact'] for x in ok) / len(ok):.2f}")

    with open(out_path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"\n[saved] {out_path} ({len(rows)} rows)")

    print("\n" + "=" * 74)
    print(f"arm 요약 (모델 {args.model} · temp {args.temp} · seeds {args.seeds})")
    print("=" * 74)
    for arm in arms:
        ok = [x for x in rows if x.get("arm") == arm and "error" not in x]
        if not ok:
            continue
        per_seed = []
        for s in range(args.seeds):
            sr = [x for x in ok if x["seed"] == s]
            if sr:
                per_seed.append(sum(x["acts_exact"] for x in sr) / len(sr))
        stp = sum(x["slots_tp"] for x in ok)
        sfp = sum(x["slots_fp"] for x in ok)
        sfn = sum(x["slots_fn"] for x in ok)
        P = stp / (stp + sfp) if stp + sfp else 0
        R = stp / (stp + sfn) if stp + sfn else 0
        print(f"{arm:5s} n={len(ok):3d} acts_exact={sum(x['acts_exact'] for x in ok) / len(ok):.3f} "
              f"(시드별 {['%.2f' % x for x in per_seed]} 폭={max(per_seed) - min(per_seed):.2f})")
        print(f"      slots P={P:.2f} R={R:.2f} F1={2 * P * R / (P + R) if P + R else 0:.2f} "
              f"· 잔존위반 {sum(1 for x in ok if x['residual_viol'])} · regen {sum(x['regen_used'] for x in ok)}")
        vc = Counter(v for x in ok for h in x["viol_history"] for v in h)
        if vc:
            print(f"      regen 유발: {dict(vc)}")
    print("\n⚠[[08]]: 집계로 결론 금지 — per-case 교차표는 x8c 포렌식으로.")


if __name__ == "__main__":
    main()
