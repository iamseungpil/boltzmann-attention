# -*- coding: utf-8 -*-
r"""x465 — **033형(이관 절차 문서 미열람) 문서-전달 격리 A/B/N** (2026-08-22·[[62]] 정보-맞춘)

## 왜 (정본 `T7336_FORENSIC_033_2026_08_22.md`)
t7336 033(reward 0.0·MISSING×4): 모델이 KB 를 `grep -r 'transfer protocol'` 한 번만 하고
라인 파편 *"proceed directly with the transfer protocol"* 을 **즉시-일반-이관 허가로 오독** —
정의 문서의 1822→0218→transfer 3단 사슬(본문 축자 *"in this exact order"*·*"Do not skip steps"*)을
발견조차 못 한 채 2차 요청 시점에 일반 이관을 조기 실행했다. 문서 본문(cat)은 끝까지 열리지
않았다. t7328 도 동일 양상(x460: amiss 9/10) — 안정 실패 모드다.

## 무엇을 가르나 ([[62]] — 결손을 격리로 잰 뒤에만 레버)
결정점에서 **정의 문서 전문을 엔진이 배달**(C585 패턴·검색 0·지시가 재료 앞=C578)하면 모델이
사슬을 찾는가:
    "안 읽어서"  → B 에서 사슬 호출이 나온다 ⇒ 레버는 **전달(부하 축소)뿐**으로 산다
    "읽어도 오독" → B 에서도 일반-이관 ⇒ 경계 — 전달로는 안 닫히고, 처방은 다른 층이다
N_neg 이 A 와 같아야 "결정점에 뭐라도 찔러서"가 아니라 **문서 내용**이 원인이라 말할 수 있다([[57]]).

## 팔 (한 변수만 다르다 · 문맥은 t7336 033 궤적 축자 복원)
    A_asis     현행 그대로 — 조기-이관 결정점 직전까지 축자. 재현 확인(라이브=일반-이관)
    B_docfull  같은 문맥 + 마지막 도구 출력 끝에 **정의 문서 전문** 배달.
               문서 id = A3 `require_doc_before.tools` 멤버십(시도 도구는 궤적에서 옴) +
               정본 `t2_gate_patch._docs_naming`(코퍼스 도출·이 파일에 문서 id 리터럴 0).
               본문 = `x448.Sandbox` 의 `shell cat`(실물 환경·[[25]]) → 코퍼스 json 폴백.
               배달 집합은 도출 **전부**(선택·순위 0) · 헤더 한 줄이 재료 **앞**(C578).
    N_neg      같은 문맥 + 같은 채널·같은 자리에 **무내용 재촉 한 줄**([[57]] 부정통제)
⚠재생 인터페이스는 **실물 도구 스키마 + 실제 메시지 객체 + `la.generate`** (x459 관용구·C584
  교훈: 렌더-텍스트로 "다음 도구 이름을 말하라"는 더 약한 인터페이스라 라이브를 재현 못 한다).

## 채점 (닫힌 술어만·[[59]]ⓐ — 다음 tool-call 의 구조 사실)
    chain_<표적>      unlock/call_discoverable_agent_tool 호출이고 인자가 A3 선언 표적
                      (require_doc_before.tools − 시도 도구)을 담는다 — 표적별로 센다
    general_transfer  시도 도구(= 일반 이관)를 그대로 부른다 — 실패 모드 재현
    other_tool / no_tool  그 밖의 호출 / 호출 없음
어느 표적이 "옳은 첫 단계"인가는 여기서 판정하지 않는다 — 표적별 계수만 보고한다(gold 0·[[69]]:
성적 확정은 본런 reward 재실행에서만). n = det 1발 + temp 6발 = **7/팔 (≥6)**.

## [[71]] 격리 서브에이전트 계약 — 4문 답
  1) 기능 하나인가 — 하나다. 각 팔의 서브(=결정점 재생성)는 **다음 발화/호출 생성** 하나만 한다.
     채점은 엔진의 닫힌 이름·멤버십 대조로 바깥에서 한다.
  2) 재료가 A2/A3 선언에서 읽혀 나왔나 — 그렇다. B 의 문서 집합은 A3 `require_doc_before.tools`
     선언과 정본 `_docs_naming`(코퍼스에서 도구명 등장 문서 도출)에서만 나온다. 이 파일에 문서
     id·태스크별 떠먹이기 리터럴 0([[63]]). gold·`reward_info` 는 열지 않는다([[23]]).
  3) 전달이 선언된 id → 정확 집기인가 — 그렇다. `x448.Sandbox.cat`(shell cat)이 도출된 id 를
     **정확히** 집는다. bm25·embedding 0 — 유사도 검색은 baseline 이지 우리 방식이 아니다.
  4) 엔진이 해석·선택·순위를 하지 않는가 — 안 한다. 도출 집합 **전부** 배달(순위 0·절단만 표시),
     어느 프로토콜이 맞는지·무엇을 부를지는 끝까지 모델 몫(argmax·정답 문장 0).

## [[62]] 4문 답
  ① 결손을 격리로 쟀나 — 정본 포렌식이 쟀다(궤적 20msg + 레버 로그 171줄 전수·[S]): 결손 =
     KB 회수 깊이(본문 미열람) + grep 파편 오독. 이 프로브는 그 결손의 **경계**를 가른다.
  ② 격리에서 되면 레버는 전달뿐인가 — 그렇다. B 가 이기면 처방은 P1/P2(문서 id 이름 대기·전달)
     계열의 **부하 축소**뿐이고, 결정론기는 늘리지 않는다.
  ③ 사라지는 모델 판단 0 — 문서를 읽고 사슬을 찾아 무엇을 부를지는 모델이 한다.
  ④ 엔진 출력에 argmax·최댓값·"정답은 X" 0 — 배달 blob 은 코퍼스 축자 연결 + 헤더 한 줄뿐.

## [[70]] 병기 (판정 의무)
단일 태스크(033)·단일 결정점 — 부호표는 이 축 1행이다. B 가 판 것 = 문맥 +N자(배달 부피·지연).
격리 결과는 승격 근거가 아니라 경계 판정이다 — 라이브 효과는 본런 A/B([[69]] reward)가 판정.

## 실행 (⚠지금 금지 — 유료 t7336 진행 중·GPU 점유 금지·리모트 읽기 전용. **t7336 종료 후**에만)
    # 리모트 · cwd=tau2-bench 루트 · vLLM Qwen2.5-32B-GPTQ-Int8 @ port 8141
    cd /home/woori/scratch/tau2-bench && \
    PYTHONPATH=src:scripts/distill/tau2 PYTHONIOENCODING=utf-8 \
    python scripts/distill/tau2/x465_transfer_doc_iso.py --port 8141
    # 배선만(LLM 0·GPU 불요): ... x465_transfer_doc_iso.py --wiring-only
"""
import argparse
import copy
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                 # noqa: E402  정본 로더·ToolCall 흡수(사본 금지·[[67]])
import t2_gate_patch as GP              # noqa: E402  `_docs_naming` 정본 — 라이브와 같은 도출

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)

# t7336 실물 경로 (포렌식 정본과 동일·읽기 전용). 본런 무접촉 cp 사본이 있으면 --results 로 준다.
D_RESULTS = ("/home/woori/scratch/tau2-bench/data/simulations/"
             "bank_t7336_halfB_20260821b/results.json")

# discoverable 사슬의 호출 껍데기 — env 레지스트리 실재를 main 에서 검증한다(이름 발명 금지·[[55]]).
DISC_TOOLS = ("unlock_discoverable_agent_tool", "call_discoverable_agent_tool")

# 배달 헤더 — 지시 한 줄이 재료 **앞**(C578: 재료 뒤에 둔 지시는 부하가 덮는다).
# 어느 문서·어느 프로토콜이 맞는지는 말하지 않는다(열린 술어=모델 몫·[[22]]·require_doc_before._note 동형).
DELIVER_HEAD = ("[KB DELIVERY] Read the following before choosing your next action. These are, "
                "in full and verbatim, ALL knowledge-base documents that mention the tool %s.")

# 무내용 재촉([[57]]) — 같은 채널·같은 자리·정보 0. "인자 변화 없음" 통제.
NUDGE = "[NOTICE] Please continue with the customer's request."


def load_a3():
    """A3 정본 층을 읽는다(x431/x464 관용구). 선언을 읽을 뿐 고르지 않는다."""
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def load_sim(path, task_suffix, trial):
    """결과 파일에서 해당 태스크 sim 하나(x464 관용구·읽기 전용)."""
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


def find_dp(msgs, pdc_tools):
    """조기-이관 결정점 — require_doc_before 선언 도구를 **처음** 부른 assistant 발화.

    닫힌 술어(역할 + 도구명 멤버십)만 쓴다 — 발화의 뜻은 판정하지 않는다([[59]]).
    033 실물: 그 첫 호출 = 일반 이관 도구(2차 요청 시점·사슬 0/4·포렌식 turn 12).
    """
    for i, m in enumerate(msgs):
        if str(m.get("role") or "") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            nm = F.nameof(tc)
            if nm in pdc_tools:
                return i, nm
    raise SystemExit("require_doc_before 선언 도구의 호출이 궤적에 없다 — 궤적이 예상과 다르다")


def last_tool_idx(msgs):
    """배달 자리 = 결정점 직전 **마지막 도구 출력**(033 실물: grep 파편 라인 그 자리).

    라이브의 배달·표면화 채널과 같은 자리다(도구-결과 채널·x459 관용구) — 역할 구조를 깨지 않는다.
    """
    for i in range(len(msgs) - 1, -1, -1):
        if str(msgs[i].get("role") or "") == "tool" and str(msgs[i].get("content") or "").strip():
            return i
    return None


def inject(ctx, text):
    """문맥 사본의 마지막 도구 출력 끝에 텍스트를 덧붙인다 — 그 외 바이트 동일."""
    out = copy.deepcopy(ctx)
    i = last_tool_idx(out)
    if i is None:
        raise SystemExit("도구 출력이 없어 배달 자리가 없다 — 이 결정점에는 이 프로브를 못 쓴다")
    out[i]["content"] = str(out[i].get("content") or "") + NLC + NLC + text
    return out


def docs_full(t_attempt, docdir, sb):
    """정의 문서 **전부**의 전문 — 도출은 정본 `_docs_naming`, 집기는 Sandbox `shell cat`.

    도출 실패·집기 실패는 조용히 넘기지 않고 보고한다(x448 1차 실측: 2편이 조용히 안 배달됐었다).
    """
    ids = sorted(GP._docs_naming(t_attempt, docdir) or set())
    if not ids:
        raise SystemExit("`_docs_naming(%s)` 도출 0편 — 코퍼스 경로부터 본다([[55]]): %s"
                         % (t_attempt, docdir))
    blob, missing = sb.cat(ids)
    fell_back = []
    for did in list(missing):
        p = os.path.join(docdir, "%s.json" % did)
        if os.path.exists(p):
            dd = json.load(io.open(p, encoding="utf-8"))
            blob += NLC + NLC + "### %s — %s%s%s" % (did, str(dd.get("title") or ""), NLC,
                                                     str(dd.get("content") or ""))
            fell_back.append(did)
            missing.remove(did)
    return ids, blob, missing, fell_back


def to_objs(msgs):
    """궤적 dict → tau2 메시지 객체 — replay 와 **같은 변환**(x467 이 뷰 조립에 재사용·[[67]] 사본 금지).

    이미 객체인 항목은 그대로 통과한다. 반환 (객체 리스트, 변환 누락 수).
    """
    from tau2.data_model.message import UserMessage, AssistantMessage, ToolMessage, SystemMessage
    CLS = {"user": UserMessage, "assistant": AssistantMessage,
           "tool": ToolMessage, "system": SystemMessage}
    out, dropped = [], 0
    for m in msgs:
        if not isinstance(m, dict):
            out.append(m)
            continue
        C = CLS.get(str(m.get("role") or ""))
        if C is None:
            dropped += 1
            continue
        try:
            out.append(C(**dict(m)))
        except Exception:
            dropped += 1
    return out, dropped


def replay(msgs, tools, model, base, temperature):
    """실제 메시지 객체 + 실물 도구 스키마로 결정점을 재생한다.

    x459.replay 관용구 그대로(C584 교훈: 렌더-텍스트는 약한 인터페이스라 라이브를 재현 못 한다)
    — 델타는 temperature 인자 하나(x459 는 0.0 고정)와 문구 치환 없음(주입은 inject 가 미리 한다).
    반환 = (calls, text, dropped, prompt_tokens). 넷째 원소는 응답 `usage.prompt_tokens`(없으면 None)
    — 재생 문맥이 라이브 생성 뷰와 같은지 라이브 usage 와 대조하는 정보-맞춤 검산용(x467 리뷰 BLOCK②).
    """
    import tau2.agent.llm_agent as la
    out, dropped = to_objs(msgs)
    resp = la.generate(model="openai/%s" % model, tools=tools, messages=out,
                       call_name="x465_replay", api_base=base, api_key="dummy",
                       temperature=temperature)
    calls = [(F.nameof(t), F.argsof(t)) for t in (getattr(resp, "tool_calls", None) or [])]
    u = getattr(resp, "usage", None)
    pt = u.get("prompt_tokens") if isinstance(u, dict) else getattr(u, "prompt_tokens", None)
    return calls, str(getattr(resp, "content", None) or ""), dropped, pt


def classify(calls, t_attempt, targets):
    """다음 tool-call 의 닫힌 분류 — 사슬(표적별) > 일반-이관 > 기타 > 무호출.

    표적 이름은 전부 A3 선언에서 왔고(런타임 도출), 여기 리터럴 0.
    """
    chain_hits, general, other = [], False, False
    for nm, ag in calls:
        vals = " ".join(str(v) for v in (ag or {}).values())
        if nm in DISC_TOOLS:
            hit = [t for t in targets if t in vals]
            chain_hits += hit or ["?unknown"]
        elif nm == t_attempt:
            general = True
        elif nm:
            other = True
    if chain_hits:
        return "chain_" + chain_hits[0].rsplit("_", 1)[-1], chain_hits
    if general:
        return "general_transfer", []
    if other:
        return "other_tool", []
    return "no_tool", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--results", default=D_RESULTS)
    ap.add_argument("--task", default="033")
    ap.add_argument("--trial", type=int, default=None, help="비우면 첫 sim (t7336 = 단일)")
    ap.add_argument("--n", type=int, default=6, help="temp 표본 수 — det 1발이 늘 앞선다(합 n+1≥7)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--maxchars", type=int, default=90000, help="배달 상한(x448 동일·절단은 표시)")
    ap.add_argument("--arms", default="A_asis,B_docfull,N_neg")
    ap.add_argument("--wiring-only", action="store_true", help="배선 검증만(LLM 0·GPU 불요·[[55]])")
    ap.add_argument("--out", default="x465_transfer_doc_iso.json")
    a = ap.parse_args()

    # ── ① A3 선언 + 궤적 복원 (축자·읽기 전용) ──────────────────────────────────
    a2 = load_a3()
    pdc = tuple((a2.get("require_doc_before") or {}).get("tools") or ())
    if not pdc:
        raise SystemExit("A3 require_doc_before.tools 선언이 비어 있다 — 층 정합부터([[24]])")
    sim = load_sim(a.results, a.task, a.trial)
    msgs = sim.get("messages") or []
    i_dp, t_attempt = find_dp(msgs, pdc)
    ctx = msgs[:i_dp]
    targets = [t for t in pdc if t != t_attempt]
    i_tool = last_tool_idx(ctx)
    print("=" * 100)
    print("x465 · task=%s trial=%s · msgs=%d · 결정점=[%d] %s · 배달 자리=[%s]"
          % (sim.get("task_id"), sim.get("trial"), len(msgs), i_dp, t_attempt, i_tool))
    print("  A3 선언 표적(시도 도구 제외): %s" % ", ".join(targets))
    if i_tool is not None:
        frag = " ".join(str(ctx[i_tool].get("content") or "").split())
        print("  직전 도구 출력 꼬리(파편 확인): …%s" % frag[-180:])

    # ── ② 배선 선검증 (LLM 0) — 샌드박스·레지스트리·도출·집기 ────────────────────
    import x448_index_vs_all_iso as IVA
    sb = IVA.Sandbox()                                  # 실물 환경([[25]]·변이 없는 읽기 전용)
    tools = list(sb.env.get_tools() or [])
    names = {t.name for t in tools}
    need = set(DISC_TOOLS) | {t_attempt}
    if not need <= names:
        raise SystemExit("env 레지스트리에 없는 채점 도구: %r — 도구 지형부터([[55]])"
                         % sorted(need - names))
    docdir = os.environ.get("T2_KB_DOCS_DIR") or __import__("x430_account_facts").DOCDIR
    ids, blob, missing, fell_back = docs_full(t_attempt, docdir, sb)
    cut = len(blob) > a.maxchars
    blob = blob[:a.maxchars]
    print("[배선] env 도구 %d종(채점 %d종 실재) · `_docs_naming(%s)` → %d편 · 전문 %d자%s"
          % (len(tools), len(need), t_attempt, len(ids), len(blob), " ⚠절단" if cut else ""))
    for did in ids:
        print("       %s%s" % (did, "  (코퍼스 폴백)" if did in fell_back else ""))
    if missing:
        print("  ⚠집기 실패(샌드박스·코퍼스 모두 없음): %r — 조용히 넘기지 않는다" % (missing,))
    if a.wiring_only:
        print("[배선] wiring-only 종료 — LLM 0·GPU 0")
        return 0

    # ── ③ 팔 구성 — 한 변수(배달 텍스트)만 다르다 ───────────────────────────────
    arm_ctx = {"A_asis": ctx,
               "B_docfull": inject(ctx, (DELIVER_HEAD % t_attempt) + NLC + NLC + blob),
               "N_neg": inject(ctx, NUDGE)}
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    base = "http://localhost:%d/v1" % a.port
    print("  재생 인터페이스: 실물 도구 스키마 %d종 + 실제 메시지 객체 + la.generate (C584)"
          % len(tools))

    # ── ④ 재생 — det 1발 + temp n발 / 팔 ────────────────────────────────────────
    rows = []
    for arm in arms:
        cx = arm_ctx[arm]
        delta = sum(len(str(m.get("content") or "")) for m in cx) - \
            sum(len(str(m.get("content") or "")) for m in ctx)
        print(NLC + "── %s (배달 델타 +%d자) ────────────────────────────────" % (arm, delta))
        for k, t in enumerate([0.0] + [a.temperature] * a.n):
            try:
                calls, text, dropped, _pt = replay(cx, tools, a.model, base, t)
            except Exception as e:
                print("  #%d t=%.1f EXC %r" % (k, t, e))
                rows.append({"arm": arm, "k": k, "temp": t, "cat": "EXC", "err": repr(e)[:200]})
                continue
            cat, hits = classify(calls, t_attempt, targets)
            rows.append({"arm": arm, "k": k, "temp": t, "cat": cat, "chain_hits": hits,
                         "calls": [[nm, json.dumps(ag, ensure_ascii=False, default=str)[:200]]
                                   for nm, ag in calls],
                         "dropped_msgs": dropped, "delta_chars": delta,
                         "text": " ".join(text.split())[:300]})
            print("  #%d t=%.1f  %-16s calls=%s%s"
                  % (k, t, cat, ",".join(nm for nm, _a in calls) or "-",
                     ("  ⚠복원 누락 %d" % dropped) if dropped else ""))

    # ── ⑤ 집계 — 팔 × 닫힌 분류 + [[70]] 병기 ───────────────────────────────────
    cats = ["chain_" + t.rsplit("_", 1)[-1] for t in targets] + \
           ["general_transfer", "other_tool", "no_tool", "EXC"]
    print(NLC + "=" * 100)
    print("%-10s %s  (n/팔)" % ("팔", " ".join("%-16s" % c for c in cats)))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        print("%-10s %s  %d" % (arm, " ".join(
            "%-16s" % ("%d/%d" % (sum(1 for r in rs if r["cat"] == c), len(rs))) for c in cats),
            len(rs)))
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"task": sim.get("task_id"), "trial": sim.get("trial"), "i_dp": i_dp,
                   "t_attempt": t_attempt, "targets": targets, "delivered_ids": ids,
                   "delivered_chars": len(blob), "truncated": cut, "missing": sorted(missing),
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(NLC + "판정: A 가 general_transfer 를 재현해야 격리가 산다(아니면 [[55]] — 결과 사용 금지).")
    print("      B 에서 chain_* 이 나오고 N 이 A 와 같으면 **원인은 미전달**(레버=전달만 산다).")
    print("      B 도 general 이면 **읽어도 오독** — 전달로 안 닫히는 경계다([[62]]②).")
    print("[[70]] 병기: 단일 태스크(033)·단일 결정점 — 부호표는 이 축 1행. B 가 판 것 = 문맥 "
          "+%d자(배달 부피·지연). 성적 확정은 본런 reward 재실행에서만([[69]])." % len(blob))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
