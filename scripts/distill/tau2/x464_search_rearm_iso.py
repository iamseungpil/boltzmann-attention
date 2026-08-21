# -*- coding: utf-8 -*-
r"""x464 — **016형 축-소진 재무장** 격리 A/B (`T2_SEARCH_REARM`·2026-08-21·[[62]] 정보-맞춘)

## 왜 (정본 `T7336_FORENSIC_016_2026_08_21.md` §레버 대조·§처방 1)
t7336 016: SEARCH_AGENT 가 Silver 특정([26]) **이전**에 Bronze 재료만 2회(263+254자) 배달한 채
두 축(business_credit_cards·credit_cards)을 잠갔고, [26]·[30] 생성 턴의 재요청 2회는
*"요청 축 … 모두 처리됨 — 침묵"* — 요건 문서 `doc_credit_cards_silver_rewards_card_011`
(*"spend at least $750 within 60 days"*)의 전달 경로가 **구조적으로 소멸**했다(그 sim 의 유일한
KB 채널·모델 자체 검색 0회). 결정점 [30]에서 모델은 요건의 **존재까지 옳게 추론**하고도 구체
수치 없이 transfer 를 택했다(user-sim 발동 조건 ③ *"Told you a specific dollar amount"* 미충족).

## 무엇을 재나 ([[62]] 정보-맞춘 격리 — Silver 특정 직후 **실제 결정점 문맥** 복원)
    A_current    현행(침묵) — t7336 016 궤적을 결정점까지 축자 복원, 재료 없음
    B_repaired   재무장 — 같은 문맥 + `T2_SEARCH_REARM` 이 배달하는 **문서 델타**(신규 계열의
                 선언 id 를 정확히 집은 본문·만료 제외 포함)
종점(닫힌 술어만·[[59]]ⓐ):
    said_750        답에 "750" 축자 — user-sim 조건 ③(구체 금액)의 실측 결손 축
    said_60day      "60 day"/"60-day" 축자
    said_spend_min  "spend at least" 축자
    transfer        "transfer" 축자 (조기 이관 잔존 여부)
⚠`submit_transaction` 도달은 user-sim(openrouter gpt-5.2·**유료**) 재실행 없이는 직접 못 잰다
  — [[09]]. 조건 ①은 이미 충족·②는 절반·③만 0 이었으므로(포렌식 축자) **③ 충족(said_750)을
  선행조건 프록시**로 보고하고, 성적 확정은 본런 reward 재실행에서만 한다([[69]]).
⚠[[70]] 판정 의무: ⑴A/B 짝 표 ⑵단일 태스크임을 명기(부호표는 이 축 1행) ⑶판 것 = 문맥 +N자
  (배달 부피)·지연 — 함께 인쇄한다.

## 엔진 배선 선(先)검증 (LLM 0·무료)
본 측정 전에 `t2_gate_patch._rearm_subjects` 를 복원 문맥에 **실물 호출**해 (군, 신규 계열)이
정확히 열리는지 찍는다 — 재무장 술어는 결정론이라 이 검증에는 GPU 도 LLM 도 필요 없다.
`--wiring-only` 로 이 단계만 돌 수 있다.

## [[71]] 격리 서브에이전트 계약 — 4문 답
  1) 기능 하나인가 — 하나다. 각 팔의 서브(=결정점 재생성)는 **다음 에이전트 발화 생성** 하나만
     한다. 채점은 엔진의 닫힌 substring 대조로 바깥에서 한다.
  2) 재료가 A2/A3 선언에서 읽혀 나왔나 — 그렇다. B 의 델타 문서 id 는 A3
     `policy_ontology.doc_index[군][계열]` 선언에서만 나온다(`t2_search.docs_for` 경유·이 스크립트에
     문서 id 리터럴 0). 복원 문맥은 실제 궤적 축자다. gold·`reward_info` 는 열지 않는다([[23]]).
  3) 전달이 선언된 id → 정확 집기인가 — 그렇다. `material_for(subjects=신규계열)` 가 선언 id 를
     corpus/파일에서 **정확히** 집는다(`x448.Sandbox` 환경 우선·bm25/embedding 0 — 유사도 검색은
     baseline 이지 우리 방식이 아니다).
  4) 엔진이 해석·선택·순위를 하지 않는가 — 안 한다. 재무장 술어는 닫힌 집합 대조
     (`groups_in` 정본)이고, 배달은 신규 계열 **전부**(순위 0)이며, 답을 고르는 것은 끝까지
     모델이다(argmax·최댓값·지목 문장 0).

## [[62]] 4문
  ①결손 측정 = t7336 016 궤적 34msg + 로그 252라인 전수(포렌식 정본·[S]) ②격리에서 재료가
  닿으면 모델이 쓰는 것은 기측정(x248 8/8·x335b 24/24) — 이 프로브는 **이 결정점**에서 그것을
  A/B 로 확정한다 ③사라지는 모델 판단 0(문서를 어떻게 쓸지는 모델 몫) ④엔진 출력에 순위·
  최댓값·정답 문장 0.

## 실행 (⚠지금 금지 — 유료 t7336 진행 중·freeze ON·리모트 접촉 금지. **t7336 종료 후**에만)
    # 리모트 · cwd=tau2-bench 루트 · vLLM Qwen2.5-32B-GPTQ-Int8 @ port 8141
    cd /home/woori/scratch/tau2-bench && \
    PYTHONPATH=src:scripts/distill/tau2 PYTHONIOENCODING=utf-8 \
    python scripts/distill/tau2/x464_search_rearm_iso.py --port 8141
    # 배선만(LLM 0·GPU 불요): ... x464_search_rearm_iso.py --wiring-only
"""
import argparse
import io
import json
import os
import sys
import types
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_gate_patch as GP              # noqa: E402  피측정 술어(정본·사본 금지 [[67]])
import t2_search as TS                  # noqa: E402  전달 체인(정본)

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))

# t7336 로그 축자 복원값 (provenance = `bank_t7336_halfB_20260821b.log` [sim=task_016#s626729]):
#   axis1 `group=business_credit_cards … turn=2`  → `[T2_DOCDECIDE] → 'Business Bronze Rewards Card'`
#   axis2 `group=credit_cards … turn=22`          → `[T2_DOCDECIDE] → 'Bronze Rewards Card'`
# 이 값들은 **격리 복원 입력**이지 엔진 상수가 아니다 — 라이브에서는 `_record_served` 가 적는다.
D_RESULTS = ("/home/woori/scratch/tau2-bench/data/simulations/"
             "bank_t7336_halfB_20260821b/results.json")
D_SERVED = "business_credit_cards=Business Bronze Rewards Card|credit_cards=Bronze Rewards Card"
D_SERVED_AT = "business_credit_cards=2|credit_cards=22"

SYS = ("You are the support agent for this bank, continuing the live conversation below. "
       "Write your next message to the customer. Be specific and grounded in what the "
       "transcript and any delivered documents actually say.")


def parse_kv(s):
    out = {}
    for part in (s or "").split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_sim(path, task_suffix, trial):
    with io.open(path, encoding="utf-8") as f:
        d = json.load(f)
    sims = [s for s in (d.get("simulations") or [])
            if str(s.get("task_id")).split("_")[-1] == str(task_suffix)]
    sims.sort(key=lambda s: s.get("trial") or 0)
    if trial is not None:
        sims = [s for s in sims if s.get("trial") == trial]
    if not sims:
        raise SystemExit("task %s 의 sim 이 없다 (%s)" % (task_suffix, path))
    return sims[0]


def find_dp(msgs, marker):
    """결정점 위치 — marker 를 **처음 축자로 확정한 assistant 발화** 뒤, 다음 내용-발화 직전.

    016 실물: marker=[26] → 결정점=[30](최종 transfer 선택 발화). 닫힌 술어(축자 substring +
    역할·위치)만 쓴다 — 발화의 뜻은 판정하지 않는다([[59]]).
    """
    i_mark = next((i for i, m in enumerate(msgs)
                   if m.get("role") == "assistant"
                   and marker in str(m.get("content") or "")), None)
    if i_mark is None:
        raise SystemExit("marker %r 를 담은 assistant 발화가 없다 — 궤적이 예상과 다르다" % marker)
    j = next((j for j in range(i_mark + 1, len(msgs))
              if msgs[j].get("role") == "assistant"
              and str(msgs[j].get("content") or "").strip()
              and not msgs[j].get("tool_calls")), len(msgs))
    return i_mark, j


def render(msgs, per_tool=3000):
    out = []
    for m in msgs:
        r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
        if r == "assistant":
            for tc in (m.get("tool_calls") or []):
                out.append("[assistant -> tool call] %s(%s)"
                           % (tc.get("name"),
                              json.dumps(tc.get("arguments") or {}, ensure_ascii=False)[:400]))
            if c:
                out.append("[assistant] " + c)
        elif r == "tool":
            out.append("[tool output] " + c[:per_tool])
        elif r == "user":
            out.append("[customer] " + c)
    return "\n".join(out)


def chat(port, model, body, temperature, maxtok):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps({"model": model, "temperature": temperature, "max_tokens": maxtok,
                         "messages": [{"role": "system", "content": SYS},
                                      {"role": "user", "content": body}]}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return str(json.loads(r.read().decode("utf-8"))
                   ["choices"][0]["message"]["content"] or "")


def score(reply):
    low = " ".join(str(reply or "").split()).lower()
    return {"said_750": "750" in low,
            "said_60day": ("60 day" in low) or ("60-day" in low),
            "said_spend_min": "spend at least" in low,
            "transfer": "transfer" in low}


def corpus_or_docdir():
    """라이브와 같은 환경 우선([[25]]) — 안 되면 A3 코퍼스 디렉터리로 떨어진다."""
    try:
        import x448_index_vs_all_iso as IVA
        cor = TS.corpus_from_env(IVA.Sandbox().env)
        if cor:
            return cor, None
    except Exception as e:
        print("⚠샌드박스 코퍼스 실패(%r) — DOCDIR 폴백" % (e,))
    try:
        import x430_account_facts as FT
        return None, FT.DOCDIR
    except Exception as e:
        raise SystemExit("코퍼스 없음(샌드박스·DOCDIR 둘 다 실패): %r" % (e,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--results", default=D_RESULTS)
    ap.add_argument("--task", default="016")
    ap.add_argument("--trial", type=int, default=None, help="비우면 첫 sim (t7336 = trial 1)")
    ap.add_argument("--marker", default="Silver Rewards Card",
                    help="주제 확정 축자(포렌식 [26]) — 결정점 탐지와 기대-계열 도출에만 쓴다")
    ap.add_argument("--served", default=D_SERVED, help="군=배달된 DOCDECIDE 결정문(로그 축자)")
    ap.add_argument("--served-at", dest="served_at", default=D_SERVED_AT,
                    help="군=배달 시점 turn(로그 축자)")
    ap.add_argument("--now", default="2025-11-14",
                    help="궤적의 현재 날짜(log_verification 축자 시각의 날짜) — 만료 제거용")
    ap.add_argument("--n", type=int, default=8, help="temp 표본 수(det 1발은 항상 먼저)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--maxtok", type=int, default=700)
    ap.add_argument("--arms", default="A_current,B_repaired")
    ap.add_argument("--wiring-only", action="store_true",
                    help="엔진 배선 검증만(LLM 0·GPU 불요)")
    ap.add_argument("--out", default="x464_search_rearm_iso.json")
    a = ap.parse_args()

    # ── ① 궤적 복원 (축자·읽기 전용) ─────────────────────────────────────────────
    sim = load_sim(a.results, a.task, a.trial)
    msgs = sim.get("messages") or []
    i_mark, i_dp = find_dp(msgs, a.marker)
    ctx = msgs[:i_dp]
    print("=" * 100)
    print("x464 · task=%s trial=%s · msgs=%d · marker(주제 확정)=[%d] · 결정점=[%d]"
          % (sim.get("task_id"), sim.get("trial"), len(msgs), i_mark, i_dp))
    print("  확정 발화 축자: %s" % " ".join(str(msgs[i_mark].get("content") or "").split())[:160])

    # ── ② 엔진 배선 선검증 — `_rearm_subjects` 실물 호출 (LLM 0) ─────────────────
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"),
                 encoding="utf-8") as f:
        a2 = json.load(f)
    po = a2.get("policy_ontology") or {}
    served_dec = parse_kv(a.served)
    served_at = {k: int(v) for k, v in parse_kv(a.served_at).items()}
    gs = list(served_dec)                      # 로그의 침묵 라인과 같은 순서
    agent = types.SimpleNamespace()
    agent._t2_search_served = {g: GP._served_subjects(po, g, decided=c)
                               for g, c in served_dec.items()}
    agent._t2_search_served_at = dict(served_at)
    ns = [types.SimpleNamespace(role=m.get("role"), content=m.get("content"),
                                tool_calls=None, id=None) for m in ctx]
    g, new = GP._rearm_subjects(agent, po, gs, set(gs), ns)
    exp = sorted(s for s in (po.get("doc_index") or {}).get("credit_cards") or {}
                 if s != "_general_" and GP._slug_disp(s) == a.marker)
    print("\n[배선] _rearm_subjects → 군=%r 신규 계열=%r" % (g, new))
    print("[배선] 기배달(복원): %s" % {k: sorted(v) for k, v in agent._t2_search_served.items()})
    ok = bool(g) and bool(set(new or ()) & set(exp))
    print("[배선] 기대 계열(%s → %r) 포함: %s" % (a.marker, exp, "PASS" if ok else "**FAIL**"))
    if not ok:
        print("⛔배선 실패 — A/B 를 돌려도 B 팔이 레버 실물과 다르다. 여기서 멈춘다([[55]]).")
        return 1
    if a.wiring_only:
        return 0

    # ── ③ B 재료 = 레버 실물이 배달할 문서 델타 (선언 id → 정확 집기) ────────────
    corpus, docdir = corpus_or_docdir()
    mat, info = TS.material_for(a2, g, now=a.now, corpus=corpus, doc_dir=docdir,
                                subjects=new, general=False, windowed="none",
                                per_doc=TS.VERDICT_PER_DOC)
    print("\n[재료] group=%s subjects=%s → %d자 (문서 %d·뺀 것 %d: %s)"
          % (g, ",".join(new), len(mat), info["kept"], len(info["dropped"]),
             ",".join(info["dropped"])[:80]))
    if not mat:
        print("⛔재료 0자 — 코퍼스 배선부터 본다([[55]]). 중단.")
        return 1

    # ── ④ A/B 생성 — det 1발 + temp n발 ─────────────────────────────────────────
    base_body = ("# Conversation transcript (verbatim, oldest first)\n%s\n\n"
                 "# Task\nWrite the agent's next message to the customer.\n"
                 % render(ctx))
    b_body = ("# Conversation transcript (verbatim, oldest first)\n%s\n\n"
              "# Policy documents just delivered to you (verbatim)\n%s\n\n"
              "# Task\nWrite the agent's next message to the customer.\n"
              % (render(ctx), mat))
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    rows = []
    for arm in arms:
        body = b_body if arm == "B_repaired" else base_body
        temps = [0.0] + [a.temperature] * a.n
        print("\n── %s (문맥 %d자%s) ────────────────────────────────" % (
            arm, len(body), " · 델타 +%d자" % len(mat) if arm == "B_repaired" else ""))
        for k, t in enumerate(temps):
            try:
                rep = chat(a.port, a.model, body, t, a.maxtok)
            except Exception as e:
                print("  #%d t=%.1f EXC %r" % (k, t, e))
                continue
            sc = score(rep)
            rows.append(dict(sc, arm=arm, k=k, temp=t, chars=len(body),
                             reply=rep[:1200]))
            print("  #%d t=%.1f  750=%-5s 60day=%-5s spend_min=%-5s transfer=%-5s | %s"
                  % (k, t, sc["said_750"], sc["said_60day"], sc["said_spend_min"],
                     sc["transfer"], " ".join(rep.split())[:110]))

    # ── ⑤ 집계 — [[70]] 짝 표 + 판 것 ───────────────────────────────────────────
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"task": sim.get("task_id"), "trial": sim.get("trial"),
                   "i_marker": i_mark, "i_dp": i_dp, "group": g, "new_subjects": new,
                   "material_chars": len(mat), "material_info": info, "rows": rows},
                  f, ensure_ascii=False, indent=1)
    print("\n" + "=" * 100)
    print("%-12s %-8s %-8s %-10s %-9s %s" % ("팔", "750", "60day", "spend_min", "transfer", "n"))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        print("%-12s %-8s %-8s %-10s %-9s %d"
              % (arm,
                 "%d/%d" % (sum(1 for r in rs if r["said_750"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["said_60day"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["said_spend_min"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["transfer"]), len(rs)), len(rs)))
    print("\n[[70]] 병기: 단일 태스크(016) 단일 결정점 — 부호표는 이 축 1행이다. B 가 판 것 = "
          "문맥 +%d자(배달 부피·지연). said_750 은 user-sim 조건 ③의 프록시이고 성적 확정은 "
          "본런 reward 재실행에서만 한다([[69]])." % len(mat))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
