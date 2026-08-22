# -*- coding: utf-8 -*-
r"""x485 — **004형 이관 결정점 격리 A/B/C/N** (2026-08-22·[[62]] 정보-맞춘·사용자 지시)

## 왜 (정본 `x484_004_variability.json`·71 sim 전수)
004 의 gold 는 액션 **하나**이고 성적을 정하는 것은 **`reason` 한 칸**이다:
`account_ownership_dispute` → PASS 22/22 · 다른 값 → FAIL 30/30 · 이관 호출 자체가 없음 → FAIL 19.
경로(검색 횟수·조회 순서)는 채점되지 않는다. 실패 49 의 분해 = **오선택 29 + 미실행 19 + 기타 1**.
⇒ 변동성은 "경로가 흔들린다"가 아니라 **결정점 하나의 두 결정이 매 런 새로 뽑힌다**는 뜻이다.

## 무엇을 가르나 ([[62]] — 레버 짓기 전에 결손을 격리로)
한 결정점에서 **두 축을 같이** 잰다:
    축 A(명명↔실행)  다음 턴에 **도구 호출을 내는가**, 아니면 산문으로 "이관했습니다" 하는가
    축 B(닫힌 enum)   낸다면 `reason` 을 무엇으로 고르는가
가르는 것:
    "안 읽어서(부하)"   → B_table 에서 옳은 코드가 나온다 ⇒ 레버는 **전달뿐**([[62]]②)
    "읽어도 못 고른다"  → B_table 도 A 와 같다 ⇒ **경계** — 전달로 안 닫힌다([[23]] 레버 없음)
N_neg 이 A 와 같아야 "결정점에 뭐라도 찔러서"가 아니라 **내용**이 원인이라 말할 수 있다([[57]]).

## 팔 (한 변수만 다르다)
    A_asis   결정점까지 궤적 축자 — **재현 확인**(라이브 continuation 과 대조·안 되면 [[55]])
    B_table  + **enum 정의 문서**만 배달. 도출 = 라이브 도구 스키마의 `reason` enum 리터럴을
             **포함하는 코퍼스 문서**(닫힌 술어·이 파일에 문서 id·코드값 리터럴 0·[[23]] gold 무접촉)
    C_full   + `_docs_naming` 도출 **전량** = 라이브 `T2_REQUIRE_DOC_DELIVER` 가 실제로 싣는 것
             ⇒ [[70]] 절충 근거: **표적 배달 ↔ 전량 배달**을 같은 자리에서 대조한다
    N_neg    + 무내용 재촉 한 줄([[57]] 부정통제·같은 채널·같은 자리)
배달 자리 = 문맥 **맨 끝의 user 메시지**(라이브와 동일: `work + fb` 의 비커밋 재생성 버퍼).
재생 인터페이스 = 실물 도구 스키마 + 실제 메시지 객체 + `la.generate`(C584 교훈·x465 관용구 재사용).

## 결정점 (닫힌 술어만·[[59]])
A2 정본의 `kind=="notice"` 게이트가 선언한 `notice_text` 가 assistant 발화에 들어간 **뒤**,
그 다음 assistant 발화. 즉 *고지·동의는 끝났고 이제 실행만 남은* 자리 — 라이브에서 우리
`T2_TERM_GRANT`(터미널 한 턴 추가)가 붙는 바로 그 지점이다. 문구는 **선언에서 읽는다**([[71]]②).

## 채점 (닫힌 술어·gold 무접촉)
    no_tool          호출 0 (축 A 실패)
    xfer:<reason>    이관 계열 호출 — `reason` 값 **그대로** 버킷(옳고 그름은 여기서 판정 안 한다)
    other_tool       그 밖의 호출
⚠어느 코드가 정답인지는 이 파일이 모른다·묻지 않는다([[69]] 성적은 본런 reward 에서만).
   판정은 바깥에서 **정책 근거**(코퍼스의 when-to-use 표)와 대조해 내린다([[23]]).

## [[71]] 4문
  1) 기능 하나 — 각 재생은 **다음 발화/호출 생성** 하나. 채점은 바깥에서 이름·값 대조.
  2) 재료는 선언에서 — 고지 문구=A2 `gates[kind=notice].notice_text` · enum=**라이브 도구 스키마** ·
     문서=코퍼스. 태스크별 떠먹이기·문서 id 리터럴 0([[63]]).
  3) 전달 = 선언된 id → 코퍼스 정확 집기(bm25·embedding 0).
  4) 엔진 해석·선택·순위 0 — 도출 집합 전부·헤더 한 줄·argmax 0·"정답은 X" 0.

## [[70]] 병기
B/C 가 파는 것 = 문맥 +N자(배달 부피·지연). 부호는 태스크별로 갈리므로 격리 결과는 **경계 판정**
이지 승격 근거가 아니다 — 라이브 효과는 본런 reward A/B 가 판정한다([[69]]).

## 실행
    cd /home/woori/scratch/tau2-bench && \
    PYTHONPATH=src:scripts/distill/tau2 PYTHONIOENCODING=utf-8 \
    /home/woori/venvs/seka_env/bin/python \
      /home/woori/workspace_common/boltzmann-attention-pi/scripts/distill/tau2/x485_transfer_reason_iso.py \
      --port 8141
    # 배선만(LLM 0·GPU 불요): ... --wiring-only
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

import t2_forensic as F                     # noqa: E402  정본 로더([[67]])
import t2_gate_patch as GP                  # noqa: E402  `_docs_naming` 정본(라이브와 같은 도출)
import x465_transfer_doc_iso as X465        # noqa: E402  재생·변환 관용구 재사용(사본 0·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)

XFER_FAMILY = ("transfer_to_human_agents", "initial_transfer_to_human_agent_0218",
               "initial_transfer_to_human_agent_1822",
               "emergency_credit_bureau_incident_transfer_1114")

DELIVER_HEAD = ("[KB DELIVERY] Read the following before choosing your next action. These are, "
                "in full and verbatim, the knowledge-base document(s) that define the reason "
                "codes accepted by %s.")
DELIVER_HEAD_FULL = ("[KB DELIVERY] Read the following before choosing your next action. These "
                     "are, in full and verbatim, ALL knowledge-base documents that mention %s.")
NUDGE = "[NOTICE] Please continue with the customer's request."
# D/E 팔의 머리 한 줄 — 뒤따르는 것이 **우리 정책층의 선언**임만 밝힌다(내용 저작 0).
ASK_HEAD = "[POLICY] The following requirement from the operating policy applies to this step."


def enum_priority(a3):
    """A3 정본의 `enum_priority` 선언 — F/G 팔의 재료. 이 파일에 문구 리터럴 0([[71]]②)."""
    e = (a3 or {}).get("enum_priority") or {}
    if not (e.get("general") and e.get("with_examples")):
        raise SystemExit("A3 enum_priority 선언이 없다/불완전하다 — 층부터([[24]])")
    return str(e["general"]), str(e["with_examples"])


def notice_gate(settings_path):
    """A2 정본의 **고지 게이트 선언**을 통째로 읽는다 — 이 파일에 문구 리터럴 0([[71]]②).

    반환 (notice_text, ask, gate_id). `ask` 는 D/E 팔의 재료다 — **축자 그대로** 쓰고
    문장을 고르거나 다듬지 않는다(엔진 저작 0·[[62]]④). 그 안에 우선순위 진술
    (*"pick the MOST SPECIFIC reason … generic … ONLY when no other option fits"*)이 들어 있다.
    """
    with io.open(settings_path, encoding="utf-8") as f:
        d = json.load(f)
    for g in (d.get("gates") or []):
        if str(g.get("kind") or "") == "notice" and g.get("notice_text"):
            return str(g["notice_text"]), str(g.get("ask") or ""), str(g.get("id") or "")
    raise SystemExit("A2 정본에 kind=notice 게이트의 notice_text 선언이 없다 — 층 정합부터([[24]])")


def find_dp(msgs, notice):
    """결정점 = 고지가 실린 assistant 발화 **다음의** assistant 발화 index.

    닫힌 술어만 — 역할 + 선언 문구의 부분문자열 포함. 발화의 뜻은 판정하지 않는다([[59]]).
    """
    i_notice = None
    for i, m in enumerate(msgs):
        if str(m.get("role") or "") == "assistant" and notice in str(m.get("content") or ""):
            i_notice = i
            break
    if i_notice is None:
        return None, None
    for j in range(i_notice + 1, len(msgs)):
        if str(msgs[j].get("role") or "") == "assistant":
            return j, i_notice
    return None, i_notice


def classify(calls):
    """닫힌 분류 — 이관이면 `reason` 값 그대로 버킷. 옳고 그름 판정 0."""
    for nm, ag in calls:
        ag = ag or {}
        inner = str(ag.get("agent_tool_name") or ag.get("user_tool_name") or "")
        target = inner or str(nm or "")
        if target in XFER_FAMILY:
            args = ag.get("arguments") if isinstance(ag.get("arguments"), dict) else ag
            return "xfer:%s" % str((args or {}).get("reason") or "-")
    if calls:
        return "other_tool"
    return "no_tool"


def live_next(msgs, j):
    """그 결정점에서 **라이브가 실제로 한 것** — 재현 확인의 대조군([[55]])."""
    m = msgs[j]
    calls = [(F.nameof(tc), F.argsof(tc)) for tc in (m.get("tool_calls") or [])]
    return {"calls": [[nm, json.dumps(ag, ensure_ascii=False, default=str)[:160]]
                      for nm, ag in calls],
            "cat": classify(calls),
            "text": " ".join(str(m.get("content") or "").split())[:200]}


def enum_values(tools, tool_name):
    """라이브 **도구 스키마**에서 enum 값 전부. 값 리터럴은 여기서 오지, 이 파일엔 없다."""
    for t in tools:
        if t.name != tool_name:
            continue
        try:
            sch = t.openai_schema
        except Exception:
            sch = getattr(t, "schema", None)
        props = (((sch or {}).get("function") or {}).get("parameters") or {}).get("properties") or {}
        for k, v in props.items():
            if isinstance(v, dict) and v.get("enum"):
                return k, [str(x) for x in v["enum"]]
    return None, []


def enum_docs(corpus, values, min_hits=3):
    """enum 값 리터럴을 **여럿 담은** 문서 = 그 enum 의 정의 문서 (닫힌 술어·유사도 0).

    `min_hits` 는 "우연히 한 값이 스친 문서"를 걷어내는 임계일 뿐 순위가 아니다 — 통과한
    문서는 **전부** 배달한다(선택·argmax 0·[[62]]④). 몇 편이 통과했는지 그대로 보고한다.
    """
    out = []
    for did, body in sorted((corpus or {}).items()):
        n = sum(1 for v in values if v in str(body or ""))
        if n >= min_hits:
            out.append((did, n))
    return out


def build_ctx(msgs, j, extra=None):
    """결정점까지의 문맥 + (있으면) 배달 user 메시지 하나 — 라이브 `work + fb` 와 같은 자리."""
    ctx = copy.deepcopy(msgs[:j])
    if extra:
        ctx.append({"role": "user", "content": extra})
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--tags", default="bank_x004_docoff2,bank_x004_base2",
                    help="원천 런 태그(쉼표) — 영속 gz 또는 리모트 라이브")
    ap.add_argument("--task", default="004")
    ap.add_argument("--per-cat", type=int, default=1, dest="per_cat",
                    help="라이브 분류당 원천 sim 수 — 관측된 국면을 고루 재현하기 위해서다")
    ap.add_argument("--max-src", type=int, default=4, dest="max_src", help="원천 sim 총 상한")
    ap.add_argument("--n", type=int, default=5, help="temp 표본/팔 — det 1발이 앞서 붙는다")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--maxchars", type=int, default=90000)
    ap.add_argument("--min-enum-hits", type=int, default=3)
    ap.add_argument("--arms", default="A_asis,B_table,C_full,N_neg",
                    help="D_priority(표+선언 ask) · E_ask_only(ask 만) · "
                         "F_tier(표+일반 티어 규칙·사례 0) · G_examples(표+사례 포함·고지절 제거)")
    ap.add_argument("--wiring-only", action="store_true")
    ap.add_argument("--out", default="x485_transfer_reason_iso.json")
    a = ap.parse_args()

    # ── ① 선언 읽기 (문구·enum·문서) ─────────────────────────────────────────────
    ntext, ask_text, gid = notice_gate(os.path.join(HERE, "a2", "banking_knowledge.settings.json"))
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8") as _f:
        ep_general, ep_examples = enum_priority(json.load(_f))
    print("=" * 100)
    print("x485 · 고지 게이트 %s 선언 문구: %r" % (gid, ntext[:70]))
    print("        선언된 ask %d자 (D/E 팔의 재료·축자)%s"
          % (len(ask_text), "" if ask_text else "  ⚠비어 있다 — D/E 불가"))

    import x448_index_vs_all_iso as IVA
    sb = IVA.Sandbox()
    tools = list(sb.env.get_tools() or [])
    names = {t.name for t in tools}
    tgt = next((t for t in XFER_FAMILY if t in names), None)
    if tgt is None:
        raise SystemExit("env 레지스트리에 이관 도구가 없다 — 도구 지형부터([[55]])")
    ekey, evals = enum_values(tools, tgt)
    if not evals:
        raise SystemExit("도구 스키마에 enum 이 없다 — B_table 의 도출 근거가 사라진다")
    import t2_search as TS
    corpus = TS.corpus_from_env(sb.env) or {}
    ed = enum_docs(corpus, evals, a.min_enum_hits)
    docdir = os.environ.get("T2_KB_DOCS_DIR") or __import__("x430_account_facts").DOCDIR
    full_ids = sorted(GP._docs_naming(tgt, docdir, corpus=corpus) or set())
    print("  도구 %s · enum 키 `%s` %d값 · 코퍼스 %d편" % (tgt, ekey, len(evals), len(corpus)))
    print("  B_table 도출(enum 값 %d개 이상 포함): %d편" % (a.min_enum_hits, len(ed)))
    for did, n in ed:
        print("       %s  (enum 값 %d개)" % (did, n))
    print("  C_full 도출(`_docs_naming`·라이브 레버와 동일): %d편" % len(full_ids))
    tset = {d for d, _ in ed}
    for did in full_ids:
        print("       %s%s" % (did, "   ← B_table 과 겹침" if did in tset else ""))
    if not ed:
        raise SystemExit("enum 정의 문서 도출 0편 — 임계(--min-enum-hits)부터 본다([[55]])")

    def deliver(ids):
        """선언된 id → **shell cat** 로 정확 집기([[71]]③·x465 와 같은 경로). 코퍼스는 폴백뿐."""
        blob, missing = sb.cat(list(ids))
        fell = []
        for did in list(missing):
            body = corpus.get(did)
            if body:
                blob += NLC + NLC + "### %s%s%s" % (did, NLC, str(body))
                fell.append(did)
                missing.remove(did)
        if missing:
            print("  ⚠집기 실패(샌드박스·코퍼스 모두 없음): %r — 조용히 넘기지 않는다" % (missing,))
        return blob[:a.maxchars], fell, sorted(missing)

    table_blob, t_fell, t_miss = deliver([d for d, _ in ed])
    full_blob, f_fell, f_miss = deliver(full_ids)
    print("  배달 부피: B_table %d자%s · C_full %d자%s"
          % (len(table_blob), (" (코퍼스 폴백 %d편)" % len(t_fell)) if t_fell else "",
             len(full_blob), (" (코퍼스 폴백 %d편)" % len(f_fell)) if f_fell else ""))

    # ── ② 원천 sim + 결정점 ────────────────────────────────────────────────────
    # 원천 선정은 **라이브가 그 결정점에서 한 것의 분류**로만 한다 — 관측된 국면을 고루 재현
    # 하려는 것이고, `reward`·gold 는 열지 않는다([[23]]). 분류는 위 `classify` 의 닫힌 술어다.
    # (초판은 "태그당 앞에서 nsims 개"였는데 그러면 통과 궤적만 뽑혀 실패 재현을 못 한다·[[18]].)
    cand = []
    for tag in [x.strip() for x in a.tags.split(",") if x.strip()]:
        for s in F.sims(tag, suffix=".results.json.gz"):
            if str(s.get("task_id")).split("_")[-1] != a.task:
                continue
            msgs = s.get("messages") or []
            j, i_n = find_dp(msgs, ntext)
            if j is None:
                continue
            cand.append({"tag": tag, "sim": s, "j": j, "i_notice": i_n,
                         # ★키에 태그를 넣는다 — 같은 시드가 여러 태그에 있어 simtag 만 쓰면
                         #   원천별 집계가 **합쳐진다**(1차 실측: 626729 가 두 태그에서 겹쳐 12/12 로 보였다).
                         "key": "%s/%s" % (tag, F.simtag(s) or str(s.get("id"))[-8:]),
                         "live": live_next(msgs, j)})
    per, srcs = {}, []
    for c in cand:
        k = c["live"]["cat"]
        if per.get(k, 0) >= a.per_cat or len(srcs) >= a.max_src:
            continue
        per[k] = per.get(k, 0) + 1
        srcs.append(c)
    print(NLC + "결정점 있는 후보 %d개 · 라이브 분류 분포: %s"
          % (len(cand), ", ".join("%s=%d" % (k, sum(1 for c in cand if c["live"]["cat"] == k))
                                  for k in sorted({c["live"]["cat"] for c in cand}))))
    if not srcs:
        raise SystemExit("결정점(고지 뒤 assistant)이 있는 sim 이 없다 — 태그·문구부터([[55]])")
    print(NLC + "원천 sim %d개" % len(srcs))
    for s in srcs:
        print("  %-20s %-14s 결정점=[%d] (고지=[%d]) · 라이브가 한 것: %-28s %s"
              % (s["tag"][:20], s["key"][-14:], s["j"], s["i_notice"], s["live"]["cat"],
                 ",".join(nm for nm, _ in s["live"]["calls"]) or "(호출 0)"))
    if a.wiring_only:
        print(NLC + "[배선] wiring-only 종료 — LLM 0·GPU 0")
        return 0

    # ── ③ 재생 ────────────────────────────────────────────────────────────────
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    base = "http://localhost:%d/v1" % a.port
    # ★D/E (2026-08-22 2차·[[18]] 구멍 메우기): 우리 게이트의 **선언된 `ask`** 는 비커밋으로
    #   나가 영속 궤적에 없다(C596) ⇒ 1차 A_asis 문맥에도 없었다. 그래서 *"우선순위 진술을 준
    #   조건"*은 아직 안 쟀다. D 는 B 에 그 진술을 **더한 것 하나만** 다르고(C578: 지시가 재료
    #   앞), E 는 진술만 준다(무엇이 사는지 분해). 문장 선별·다듬기 0 — 선언 축자 그대로다.
    extra = {"A_asis": None,
             "B_table": (DELIVER_HEAD % tgt) + NLC + NLC + table_blob,
             "C_full": (DELIVER_HEAD_FULL % tgt) + NLC + NLC + full_blob,
             "N_neg": NUDGE,
             "D_priority": (ASK_HEAD + NLC + ask_text + NLC + NLC
                            + (DELIVER_HEAD % tgt) + NLC + NLC + table_blob),
             "E_ask_only": ASK_HEAD + NLC + ask_text,
             # ★F/G (2026-08-22 3차·[[66]] 자기감사): D 의 이득이 **일반 규칙**에서 온 것인지
             #   **사례 열거**에서 온 것인지 갈리지 않았다. D 가 실은 게이트 ask 를 통째로 실었고
             #   그 안에 004 의 답을 이름으로 대는 예시가 있다(*"identity values do not match →
             #   an ownership-dispute reason"*·커밋 aa657c59 = 실패를 보고 쓴 처방). 그러면
             #   측정 대상이 사라진다([[62]]④). 두 팔이 그것을 가른다 — 둘 다 고지-발송 절은
             #   빠져 있어 1차 D 의 재고지 오염([[55]] 문구 모순)도 함께 제거된다.
             #     F_tier      일반 티어 규칙만 (사례 0·doc_042 축자·도메인 일반)
             #     G_examples  같은 자리에 사례 포함본 (D 의 후반부 축자)
             #   F 가 0 이고 G 만 산다면 이득의 정체는 떠먹이기이고 이 축은 우리 레버가 아니다.
             "F_tier": (ASK_HEAD + NLC + ep_general + NLC + NLC
                        + (DELIVER_HEAD % tgt) + NLC + NLC + table_blob),
             "G_examples": (ASK_HEAD + NLC + ep_examples + NLC + NLC
                            + (DELIVER_HEAD % tgt) + NLC + NLC + table_blob)}
    if not ask_text and ({"D_priority", "E_ask_only"} & set(arms)):
        raise SystemExit("선언된 ask 가 비어 D/E 를 만들 수 없다 — A2 층부터([[24]])")
    rows = []
    for s in srcs:
        msgs = s["sim"].get("messages") or []
        for arm in arms:
            ctx = build_ctx(msgs, s["j"], extra.get(arm))
            delta = len(extra.get(arm) or "")
            print(NLC + "── %s / %s (배달 +%d자) ─────────────────" % (s["key"][-14:], arm, delta))
            for k, t in enumerate([0.0] + [a.temperature] * a.n):
                try:
                    r = X465.replay(ctx, tools, a.model, base, t)
                except Exception as e:
                    print("  #%d t=%.1f EXC %r" % (k, t, e))
                    rows.append({"src": s["key"], "tag": s["tag"], "arm": arm, "k": k,
                                 "temp": t, "cat": "EXC", "err": repr(e)[:200]})
                    continue
                cat = classify(r.calls)
                rows.append({"src": s["key"], "tag": s["tag"], "arm": arm, "k": k, "temp": t,
                             "cat": cat, "delta_chars": delta, "dropped_msgs": r.dropped,
                             "prompt_tokens": r.prompt_tokens,
                             "calls": [[nm, json.dumps(ag, ensure_ascii=False, default=str)[:200]]
                                       for nm, ag in r.calls],
                             "text": " ".join(r.text.split())[:300]})
                print("  #%d t=%.1f  %-46s %s" % (k, t, cat,
                                                  ",".join(nm for nm, _ in r.calls) or "-"))

    # ── ④ 집계 ────────────────────────────────────────────────────────────────
    cats = sorted({r["cat"] for r in rows})
    print(NLC + "=" * 100)
    print("%-10s %s" % ("팔", " ".join("%-30s" % c[:30] for c in cats)))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        print("%-10s %s  (n=%d)" % (arm, " ".join(
            "%-30s" % ("%d/%d" % (sum(1 for r in rs if r["cat"] == c), len(rs)))
            for c in cats), len(rs)))
    print(NLC + "축 A(명명↔실행) — 호출을 낸 비율")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        acted = sum(1 for r in rs if r["cat"] not in ("no_tool", "EXC"))
        print("  %-10s %d/%d" % (arm, acted, len(rs)))
    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"task": a.task, "notice_gate": gid, "tool": tgt, "enum_key": ekey,
                   "enum_values": evals, "table_ids": [d for d, _ in ed], "full_ids": full_ids,
                   "table_chars": len(table_blob), "full_chars": len(full_blob),
                   "sources": [{"tag": s["tag"], "key": s["key"], "j": s["j"],
                                "live": s["live"]} for s in srcs],
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(NLC + "판정: A_asis 가 라이브 continuation 을 재현해야 격리가 산다(아니면 [[55]]·결과 폐기).")
    print("      B_table 에서 옳은 코드가 나오고 N_neg 이 A 와 같으면 원인은 **미전달**([[62]]②).")
    print("      B_table 도 A 와 같으면 전달로 안 닫히는 **경계**다 — 레버 없음으로 남긴다([[23]]).")
    print("      C_full ↔ B_table 차이가 [[70]] 절충의 근거다(전량 배달이 무엇을 파는가).")
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
