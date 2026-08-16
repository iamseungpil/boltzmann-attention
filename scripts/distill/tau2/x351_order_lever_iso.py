# -*- coding: utf-8 -*-
r"""x351 — **순서 레버의 두 전제를 격리로 먼저 잰다**(⛔0번 규칙·[[62]]).

## 왜

t7305 전수 포렌식(C508)이 확정한 라이브 사실: **결정이 요구보다 먼저** 난다.
  · savings 결정 = turn 4~6 · 그 시점 손님 발화는 msg 1·3 뿐(=checking 요구 + 벤치의 미끼)
  · savings 요구가 실제로 도착하는 것은 msg 41~61(sim 별로 다름·아래 표는 궤적에서 읽었다)
그래서 "요구가 도착할 때까지 결정 서브를 부르지 않는다(대신 묻는다)"는 순서 레버가 후보다.
**짓기 전에** 그 레버가 살 수 있는 것이 있는지, 그리고 그 판단을 LLM 이 할 수 있는지를 잰다.

## 잰다 ① — 효용: 요구가 있었다면 서브가 gold 를 냈을까

    A_REF      결정 시점(turn 4~6)의 발화로 만든 인용 + 후보줄     ← 라이브 재현(기대 오답)
    B_ARRIVED  **요구가 도착한 뒤**의 발화로 만든 인용 + 후보줄     ← 순서 레버가 사는 조건
    D_NEG      다른 태스크의 요구 + 후보줄                        ← 부정통제([[57]])

인용은 라이브와 **같은 절차**(A2 `requirement_prompt` → LLM → 원문 `in` 검산)로 만들고,
본문은 라이브 축자 구성(A2 `doc_decide_prompt`·요구 머리·재료 몸)으로 짠다. 정규식 0.

## 잰다 ② — 판별: "이 축의 요구는 아직 없다"를 LLM 이 말할 수 있나

같은 두 시점의 발화에 **축을 명시한** 인용 요구를 준다(축 이름의 출처 = env 문서군 키이고,
축 선택은 라이브에서도 LLM 몫이다). 엔진은 인용의 **원문 존재만** 검산한다([[66]]·[[22]]).

    NOW      결정 시점 발화 → **빈 목록**이어야 맞다(그 축 요구가 아직 없다)
    ARRIVED  도착 뒤 발화   → **비어 있지 않아야** 맞다  ← 이것이 부정통제다(둘 다 비면 죽은 판별기)

## 판정 (사전 고정 · 결과보다 먼저 인쇄된다)

    ①효용   B_ARRIVED gold ≥6/8 ∧ A_REF gold ≤2/8 → 순서 레버가 살 것이 **있다**
            B ≈ A (둘 다 낮음)                     → 순서로는 못 산다 ⇒ **레버 폐기**(결손은 다른 이름)
            D_NEG gold ≥3                          → 계기 무효
    ②판별   NOW 빈 ≥6/8 ∧ ARRIVED 비지-않음 ≥6/8   → LLM 이 판별한다 ⇒ 엔진은 **검산만**(결정론 0)
            어느 한쪽 <6/8                          → 판별 불가 ⇒ 순서 레버는 ASK 로만(엔진 판단 금지)

실행(리모트·8141):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x351_order_lever_iso.py
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import t2_probe as P                                              # noqa: E402
import t2_search as TS                                            # noqa: E402

TAG = "bank_t7305_treat_20260817a"
AUX = "bank_t7305_treataux_20260817a"
TASK = "task_055"
NEG_TASK, NEG_SEED, NEG_MSG = "task_024", "1567", 1
GROUP = "savings_accounts"
NOW_CLOCK = "2025-11-14"
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"
A2DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2")
MARKS = {"SILVERPLUS": "Silver Plus", "GREEN": "Green Account", "GOLD": "Gold Account"}

# sim → (결정 turn, 요구가 **도착한** 손님 메시지 index). 궤적에서 읽었다(x350 전수 정독).
#   ⚠분석 판단이지 엔진 규칙이 아니다 — 레버는 이 표를 쓰지 않는다.
SIMS = {
    "1567":   (2, 46),
    "361454": (6, 41),
    "363271": (4, 45),
    "373753": (6, 41),
    "514337": (4, 54),
    "554706": (4, 16),
    "626729": (4, 61),
    "863145": (4, 53),
}

SCOPED = ("Below are the customer's own messages in this conversation.\n\n{messages}\n\n"
          "List the requirements the customer stated for the {group} the agent is about to "
          "recommend now. Copy each requirement VERBATIM from the messages above - do not "
          "paraphrase, do not add anything they did not say. If the customer has not stated any "
          "requirement for it yet, reply with an empty array. Reply with a JSON array of strings "
          "and nothing else.")


def a2_load():
    out = {}
    for name in ("banking_knowledge.settings.json", "banking_knowledge.specific.json"):
        p = os.path.join(A2DIR, name)
        if os.path.exists(p):
            out.update(json.load(io.open(p, encoding="utf-8")))
    return out


def disp(slug):
    return " ".join(w.capitalize() if w[:1].islower() else w
                    for w in str(slug).replace("_", " ").split())


def sim_of(tag, task, seed):
    for s in F.sims(tag):
        if F.task_id(s) == task and str(s.get("seed")) == str(seed):
            return s
    return None


def user_text(sim, upto_turn=None, upto_idx=None):
    """손님 발화를 **인덱스로 골라 통째로** 잇는다(추출 0). 라이브 `_users` 와 같은 규약."""
    out = []
    for i, m in enumerate(sim.get("messages") or []):
        if m.get("role") != "user":
            continue
        ti = m.get("turn_idx")
        if upto_turn is not None and not (ti is not None and int(ti) < int(upto_turn)):
            continue
        if upto_idx is not None and i > upto_idx:
            continue
        out.append(" ".join(str(m.get("content") or "").split()))
    return "\n\n".join(out)


def quotes(tpl, text, label="", **kw):
    """LLM formalize → 원문 존재 검산(`in`)만. 정규식 0·엔진 추출 0.

    ⚠**빈 목록의 두 뜻을 갈라 인쇄한다**(2026-08-17 1차 실행에서 걸린 계기 결함): `max_tokens`
      가 작으면 긴 대화에서 JSON 이 **잘려** 파싱이 깨지고, 그것이 조용히 `[]` 가 되어
      *"모델이 요구가 없다고 답했다"* 로 오독된다(어제 x345 절단 사고와 같은 부류).
      ⇒ 한도를 올리고, 파싱 실패는 **`FAIL`** 로 인쇄해 판별 집계에서 뺀다."""
    if not (tpl and text):
        return None
    raw = str((chat(tpl.format(messages=text[:8000], **kw), None, 0.0, 2000) or {}).get("content") or "")
    i, j = raw.find("["), raw.rfind("]")
    if i < 0 or j <= i:
        print("   ⚠[%s] JSON 경계 없음(절단 의심·응답 %d자) — 빈 목록으로 세지 않는다" % (label, len(raw)))
        return None
    try:
        rows = json.loads(raw[i:j + 1])
    except Exception as e:
        print("   ⚠[%s] JSON 파싱 실패(%s·응답 %d자) — 빈 목록으로 세지 않는다"
              % (label, type(e).__name__, len(raw)))
        return None
    return [q for q in rows if isinstance(q, str) and q and q in text]


def block(qs):
    return "Customer's stated request:\n" + "\n".join("- " + q for q in qs)


def main():
    a2 = a2_load()
    po = a2.get("policy_ontology") or {}
    corpus = {}
    for fn in sorted(os.listdir(DOCS)):
        d = json.load(io.open(os.path.join(DOCS, fn), encoding="utf-8"))
        corpus[str(d.get("id") or fn)] = str(d.get("content") or "")
    material, info = TS.material_for(a2, GROUP, now=NOW_CLOCK, corpus=corpus)
    if not material or not (info.get("dropped") or ()):
        print("재료/만료제거 실패 — 중단(계기 결함)")
        return 1
    cands = ", ".join(disp(x) for x in sorted((po.get("doc_index") or {}).get(GROUP) or ()))
    cand_line = str(po.get("decide_candidates_text")).format(candidates=cands)
    tpl = str(po.get("doc_decide_prompt") or "")
    rq = str(po.get("requirement_prompt") or "")
    if "{ask}" not in tpl or not rq:
        print("A2 템플릿 없음 — 중단(계기 결함)")
        return 1
    live = lambda ask: tpl.format(ask=ask, material=material)       # noqa: E731

    neg_sim = sim_of(AUX, NEG_TASK, NEG_SEED)
    neg = " ".join(str((neg_sim.get("messages") or [])[NEG_MSG].get("content") or "").split())

    print("x351 · %s/%s · group=%s · 재료 %d자(문서 %d·뺀 것 %d)"
          % (TAG, TASK, GROUP, len(material), info.get("kept", 0), len(info.get("dropped") or ())))
    print("판정(사전 고정): ①효용 B_ARRIVED gold ≥6/8 ∧ A_REF ≤2/8 → 순서 레버가 살 것이 있다 · "
          "B≈A → 레버 폐기 · D_NEG gold ≥3 → 계기 무효 / ②판별 NOW 빈 ≥6/8 ∧ ARRIVED 비지-않음 ≥6/8 "
          "→ LLM 이 판별(엔진은 검산만) · 한쪽 <6/8 → 판별 불가(ASK 로만)\n")

    res = {"payoff": {}, "discrim": {}}
    for seed, (turn, arr_idx) in sorted(SIMS.items()):
        sim = sim_of(TAG, TASK, seed)
        if sim is None:
            print("[%s] 궤적 없음 — 건너뜀(계기 결함 가능)" % seed)
            continue
        t_now = user_text(sim, upto_turn=turn)
        t_arr = user_text(sim, upto_idx=arr_idx)
        print("── seed %s · 결정 turn=%d · 요구 도착 msg=%d (발화 now %d자 / arrived %d자)"
              % (seed, turn, arr_idx, len(t_now), len(t_arr)))
        q_now = quotes(rq, t_now, "q_now/%s" % seed)
        q_arr = quotes(rq, t_arr, "q_arr/%s" % seed)
        # ② 판별 — 축을 명시한 인용 요구(빈 목록이 정답인 쪽 / 아닌 쪽)
        d_now = quotes(SCOPED, t_now, "d_now/%s" % seed, group=disp(GROUP))
        d_arr = quotes(SCOPED, t_arr, "d_arr/%s" % seed, group=disp(GROUP))
        cnt = lambda x: ("FAIL" if x is None else len(x))            # noqa: E731
        res["discrim"][seed] = {"now_n": cnt(d_now), "arrived_n": cnt(d_arr),
                                "now": (d_now or [])[:3], "arrived": (d_arr or [])[:3]}
        print("   인용(now %s개 / arrived %s개) · **판별**(축 명시: now %s개 / arrived %s개)"
              % (cnt(q_now), cnt(q_arr), cnt(d_now), cnt(d_arr)))
        for q in (d_arr or [])[:2]:
            print("      arrived 인용 예: %s" % q[:140])
        arms = [("A_REF", live(block(q_now) + "\n\n" + cand_line))]
        if q_arr:
            arms.append(("B_ARRIVED", live(block(q_arr) + "\n\n" + cand_line)))
        r = P.run("x351-%s" % seed, {"tag": TAG, "task": TASK, "cut": turn, "sim": seed, "base": ""},
                  arms, MARKS, "(판정은 전 sim 합산 후·위 문구 그대로)", "", None, 8, 3, det=True)
        res["payoff"][seed] = {k: {m: v[m][0] for m in MARKS} for k, v in (r or {}).items()}

    # 부정통제 1회(입력이 sim 과 무관하게 동일하므로 반복은 같은 계산이다)
    rn = P.run("x351-DNEG", {"tag": TAG, "task": TASK, "cut": 0, "sim": "-", "base": ""},
               [("A_REF", live(cand_line)), ("D_NEG", live(block([neg]) + "\n\n" + cand_line))],
               MARKS, "(부정통제)", "", None, 8, 3, det=True)
    res["payoff"]["DNEG"] = {k: {m: v[m][0] for m in MARKS} for k, v in (rn or {}).items()}

    print("\n" + "=" * 96)
    print("① 효용 — 축별 gold(`Silver Plus`) 적중 sim 수")
    a_gold = sum(1 for s, v in res["payoff"].items()
                 if s != "DNEG" and v.get("A_REF", {}).get("SILVERPLUS", 0) > 0)
    b_gold = sum(1 for s, v in res["payoff"].items()
                 if s != "DNEG" and v.get("B_ARRIVED", {}).get("SILVERPLUS", 0) > 0)
    n = len([s for s in res["payoff"] if s != "DNEG"])
    print("   A_REF(결정 시점) %d/%d · B_ARRIVED(요구 도착 후) %d/%d · D_NEG %s"
          % (a_gold, n, b_gold, n, res["payoff"].get("DNEG", {}).get("D_NEG")))
    print("   → %s" % ("순서 레버가 살 것이 **있다**" if (b_gold >= 6 and a_gold <= 2)
                       else "미결/폐기 — 위 사전 문구로 판정하라"))
    print("② 판별 — 축 명시 인용")
    now_ok = [v for v in res["discrim"].values() if v["now_n"] != "FAIL"]
    arr_ok = [v for v in res["discrim"].values() if v["arrived_n"] != "FAIL"]
    now_empty = sum(1 for v in now_ok if v["now_n"] == 0)
    arr_full = sum(1 for v in arr_ok if v["arrived_n"] > 0)
    nfail = sum(1 for v in res["discrim"].values()
                if "FAIL" in (str(v["now_n"]), str(v["arrived_n"])))
    print("   NOW 빈 목록 %d/%d · ARRIVED 비지-않음 %d/%d · 계기 FAIL %d sim(집계 제외)"
          % (now_empty, len(now_ok), arr_full, len(arr_ok), nfail))
    print("   → %s" % ("LLM 이 판별한다 ⇒ 엔진은 검산만" if (now_empty >= 6 and arr_full >= 6)
                       else "판별 불가 ⇒ 순서 레버는 ASK 로만(엔진 판단 금지)"))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                       "reports", "facet_rft_2026", "x351_order_lever_iso.json")
    with io.open(os.path.normpath(out), "w", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print("저장: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
