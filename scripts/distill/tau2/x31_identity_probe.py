#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x31: 상인 지시-동일성 **격리 프로브** (무료·로컬 vLLM·gold 0).

질문(사용자 2026-08-01): *"약어표는 스케일이 배워야 할 상식·지식을 formal하게 넣어주는 것인가?"*
⇒ 판정 기준 4번(설계 논의): **격리 프로브가 표 내용을 스스로 재현하면** 지식은 모델에 있고 표는
   *검사 가능성*만 더한 것(=검증 어휘)이다. 재현 못 하면 그만큼은 지식 보충이다.

프로토콜([[18]] 정보-맞춤 · C265 순서-통제):
  - 입력 = 정책 제외-불릿 이름 1개 + **카탈로그 merchant_name 전량(138)**. 표는 절대 안 보여준다.
  - 출력 = 같은 업체를 가리키는 카탈로그 항목의 JSON 배열(없으면 빈 배열).
  - **2 arm = 카탈로그 정순 / 역순** — 답이 순서의 함수면 '앎'이 아니다(C265 실증).
  - 채점 = A2 저작 표와의 집합 일치(gold 미열람·표는 정책×카탈로그+세계지식으로 저작된 것).
"""
import argparse, json, os, re, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8140/v1")
ap.add_argument("--model", default="openai/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
ap.add_argument("--out", default="/home/woori/scratch/x30run/x31_probe.json")
A = ap.parse_args()

import tau2.agent.llm_agent as la                        # noqa: E402
from tau2.data_model.message import UserMessage          # noqa: E402
from tau2.utils.utils import DATA_DIR                    # noqa: E402

DOM = os.path.join(str(DATA_DIR), "tau2", "domains", "banking_knowledge")
A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
QP = next(t for t in A2["scaffold_get_tools"]
          if t["name"] == "get_reward_discrepancies")["variants"]["ratefix"]["isolate"]["quote_pin"]
TBL = QP["policy_group_rows"]
# 상인-불릿 키만(그룹 제목 제외) = 저작 시 판단이 실제로 들어간 자리
GROUPWORDS = ("Merchants", "Retailers", "Markets", "Subscriptions", "excluded")
KEYS = [k for k in TBL if not any(w in k for w in GROUPWORDS)]

merch = set()


def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "merchant_name" and isinstance(v, str):
                merch.add(v.strip())
            else:
                walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


walk(json.load(open(os.path.join(DOM, "db.json"), encoding="utf-8")))
CAT = sorted(merch)
print(f"프로브 키 {len(KEYS)}개 · 카탈로그 {len(CAT)}종 · arm 2(정순/역순)")

PROMPT = (
    "A bank's reward policy document lists an excluded merchant by this name:\n\n"
    "    POLICY NAME: {name}\n\n"
    "Below is the complete list of merchant names as they appear on customer transaction records "
    "at this bank. Transaction records often write a business's name differently from how the "
    "policy writes it.\n\n{catalog}\n\n"
    "Which of the listed transaction merchant names refer to the SAME business that the policy "
    "name refers to? Include every one that does. If none of them is that business, answer with an "
    "empty list. Do not include businesses that merely have a similar-looking name or operate in a "
    "similar industry.\n\n"
    "Reply with EXACTLY one JSON array of strings copied verbatim from the list, and nothing else.")


def parse(txt, cat):
    out, i = [], 0
    txt = txt or ""
    while i < len(txt):
        if txt[i] != "[":
            i += 1
            continue
        for j in range(len(txt), i, -1):
            try:
                v = json.loads(txt[i:j])
            except Exception:
                continue
            if isinstance(v, list):
                out = [str(x) for x in v]
                i = len(txt)
                break
        else:
            i += 1
            continue
        break
    valid = {m.lower(): m for m in cat}
    return sorted({valid[x.strip().lower()] for x in out if x.strip().lower() in valid})


kw = {"api_base": A.base, "api_key": "dummy", "temperature": 0.0,
      "max_tokens": 2048, "timeout": 1200.0, "num_retries": 1}
RES = {}
for arm, cat in (("fwd", CAT), ("rev", list(reversed(CAT)))):
    RES[arm] = {}
    for n, key in enumerate(KEYS, 1):
        p = PROMPT.format(name=key, catalog="\n".join("- " + m for m in cat))
        try:
            um = UserMessage(role="user", content=p)
        except TypeError:
            um = UserMessage(content=p)
        try:
            r = la.generate(model=A.model, tools=None, messages=[um], call_name="x31", **kw)
            got = parse(getattr(r, "content", None) or "", CAT)
        except Exception as e:
            print(f"  [{arm}] {key}: 실패 {e!r}")
            got = None
        RES[arm][key] = got
        exp = sorted(TBL[key])
        mark = "=" if got is not None and got == exp else "≠"
        print(f"  [{arm} {n:2d}/{len(KEYS)}] {key:24s} 모델={got} {mark} 표={exp}")

# ── 채점 ──────────────────────────────────────────────────────────────────────
def score(arm):
    ex = im = 0
    tp = fp = fn = 0
    for k in KEYS:
        got, exp = RES[arm].get(k), set(TBL[k])
        if got is None:
            continue
        g = set(got)
        ex += (g == exp)
        im += (g != exp)
        tp += len(g & exp); fp += len(g - exp); fn += len(exp - g)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    return ex, im, tp, fp, fn, prec, rec


print("\n" + "=" * 72)
for arm in ("fwd", "rev"):
    ex, im, tp, fp, fn, pr, rc = score(arm)
    print(f"[{arm}] 집합 정확일치 {ex}/{len(KEYS)} · 원소 TP{tp} FP{fp} FN{fn} · 정밀 {pr:.2f} 재현 {rc:.2f}")
stable = sum(1 for k in KEYS if RES["fwd"].get(k) == RES["rev"].get(k))
print(f"[순서 안정성] 정순==역순 {stable}/{len(KEYS)}  ⇒ {'안정(앎의 신호)' if stable >= len(KEYS)*0.9 else '불안정(위치 함수·C265형)'}")

print("\n[판별 케이스] 표 저작 판단이 접두 규칙과 갈렸던 자리:")
for k in ("Target", "Microsoft", "Dell", "Amazon", "LinkedIn Learning", "ThredUp", "Slack", "Zoom"):
    if k in RES["fwd"]:
        print(f"  {k:18s} fwd={RES['fwd'][k]} rev={RES['rev'][k]} 표={sorted(TBL[k])}")

json.dump({"keys": KEYS, "table": {k: TBL[k] for k in KEYS}, "result": RES},
          open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n→ {A.out}")
