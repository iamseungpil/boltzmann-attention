# -*- coding: utf-8 -*-
"""X13 SU-MT — 대본-유저 다중턴 프로브 본체 (MT_PROBE_DESIGN rev3 · 2026-07-30 야간).

설계 = `reports/facet_rft_2026/MT_PROBE_DESIGN_2026_07_30.md`(rev3). 요지:
  · 유저 = **대본**(실 궤적의 user 발화 순서 재생) → user-sim 비용 0
  · 도구 = **진짜 tau2 env**(banking_knowledge·bm25) → 실 스키마·실 오류
  · 에이전트 = 로컬 32B(8141) → 비용 0
  ⇒ 장문·다중턴·실도구라는 배포-유사 3요소를 무료로 얻는다.

★arm(rev3 §3-2b 실측 반영):
  A_PROMPT    프롬프트로만 봉투 요구(도구 정상 제공)
  B_SAYGUIDED 말-채널(도구 미제공) 요청에만 guided_json — **행동 채널에 문법을 걸면 호출이 0이 된다**
  C_VERIFY    A + §1d 검증기 + 위반 시 regen 1회
  D_TWOPASS   1패스=행동(문법 없음) → 2패스=그 행동을 봉투로 형식화(문법·도구 미제공)

⚠채점 금지 목록(설계서 §2-1·§7): 태스크 pass 채점 금지 · ASK를 페널티로 채점 금지
  (대본 유저는 답하지 않는다) · 지평 초과율은 **1차 지표**로 보고 · 이탈 행은 별도 칸.

용법:
  py -3 x13_su_mt_probe.py --check-env            # V2: env 격리(get_db_hash 전후 대조)
  py -3 x13_su_mt_probe.py --smoke                # V3: 2케이스 × 1시드 × 4 arm
  py -3 x13_su_mt_probe.py --cases 12 --seeds 3 --out rows.jsonl
"""
import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import x13_su_mt_cases as _cases                                    # noqa: E402

BASE = os.environ.get("X13_BASE", "http://localhost:8141/v1")
MODEL = os.environ.get("X13_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
DOMAIN = "banking_knowledge"
RETRIEVAL = "bm25"          # ★비용 0 — openai_embeddings는 외부 호출이라 쓰지 않는다([[09]])
ARMS = ["A_PROMPT", "B_SAYGUIDED", "C_VERIFY", "D_TWOPASS"]

ENVELOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "turn_type": {"type": "string", "enum": ["ACT", "ASK", "CONFIRM", "INFORM", "DONE"]},
        "next_action": {"type": ["string", "null"]},
        "ask": {"type": ["object", "null"],
                "properties": {"slot": {"type": "string"},
                               "reason": {"type": "string", "enum": ["missing", "confirm"]}},
                "required": ["slot", "reason"]},
        "done_report": {"type": "array", "items": {
            "type": "object",
            "properties": {"kind": {"type": "string"}, "what": {"type": "string"},
                           "resolves": {"type": ["string", "null"]}},
            "required": ["kind", "what"]}},
        "prose": {"type": "string"},
    },
    "required": ["turn_type", "prose"],
}

# ★문구 강도 교정(2026-07-30 야간·실측): 초판 문구로는 **행동 턴에서 content가 빈 문자열**이었다
#   (도구를 부르면 봉투를 아예 안 낸다). 채널 제약인지 프롬프트 강도인지 격리 테스트한 결과
#   **강한 지시에서는 tool_calls와 봉투가 함께 나온다** ⇒ 채널 제약 아님·문구 문제.
#   초판 문구로 잰 준수율은 "배포 조건의 결손"이 아니라 **내 가이드의 결손**이었다([[42]] 동형).
ENVELOPE_GUIDE = (
    "\n\n<declaration>\nRULE: every message you send MUST contain a declaration envelope as a single "
    "JSON object in the message content. This is mandatory **even when you also call a tool in the "
    "same turn** — never leave the content empty. **Write the envelope FIRST, before any tool call** "
    "— text placed after a tool call is discarded by the server and does not count. Format:\n"
    '{"turn_type":"ACT|ASK|CONFIRM|INFORM|DONE","next_action":"<tool you are calling now or null>",'
    '"ask":{"slot":"<slot>","reason":"missing|confirm"} or null,'
    '"done_report":[{"kind":"<claim kind>","what":"<what you did>","resolves":null}],'
    '"prose":"<what you tell the customer>"}\n'
    "The declaration must match what you actually do this turn.\n</declaration>"
)


# ── 엔진 접점 ────────────────────────────────────────────────────────────────
def make_env():
    from tau2.domains.banking_knowledge.environment import get_environment
    return get_environment(retrieval_variant=RETRIEVAL)


def tool_schemas(env):
    out = []
    for t in env.get_tools():
        try:
            out.append(t.openai_schema)
        except Exception as e:
            print("  ⚠스키마 실패 %r: %r" % (getattr(t, "name", "?"), e), file=sys.stderr)
    return out


def system_prompt(env, arm):
    from tau2.agent.llm_agent import SYSTEM_PROMPT, AGENT_INSTRUCTION
    sp = SYSTEM_PROMPT.format(domain_policy=env.get_policy(), agent_instruction=AGENT_INSTRUCTION)
    return sp + (ENVELOPE_GUIDE if arm != "D_TWOPASS" else "")


LAST_USAGE = {}


def call_llm(msgs, seed, tools=None, guided=None, max_tokens=700):
    body = {"model": MODEL, "messages": msgs, "temperature": 0.0, "seed": seed,
            "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if guided is not None:
        # ⚠rev3 §3-2b: tools와 동시에 걸면 tool_calls가 0이 된다 — 호출측이 배타를 보장한다.
        assert not tools, "문법과 도구를 같은 호출에 걸지 말 것(행동 채널이 죽는다)"
        body["guided_json"] = guided
        body["guided_decoding_backend"] = "xgrammar"
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    # ★hermes 파서는 `<tool_call>` **뒤** 텍스트를 버린다(2026-07-30 실측: 호출 뒤에 봉투를 쓰면
    #   completion_tokens는 늘어나는데 content는 빈 문자열로 온다). "봉투 없음"과 "봉투가 잘림"을
    #   구분하려면 생성 토큰 수를 함께 봐야 한다.
    LAST_USAGE.clear()
    LAST_USAGE.update(d.get("usage") or {})
    return d["choices"][0]["message"]


# ── 봉투 파싱·검증(§1d 축소판·전부 닫힌 술어) ────────────────────────────────
def parse_env_obj(content):
    """봉투 추출. ★초판은 `첫 { ~ 마지막 }`를 통째로 잘라 파싱했다 — 산문 뒤에 봉투가 붙거나
    본문에 중괄호가 있으면 **오파싱**한다(원문 정독서 실제 확인). 이제 **균형 잡힌 중괄호 블록**을
    전부 찾아 그중 `turn_type`을 가진 첫 객체만 봉투로 인정한다(닫힌 술어)."""
    t = (content or "").strip()
    if not t or "{" not in t:
        return None
    t = t.replace("```json", "```")
    out = []
    for i, ch in enumerate(t):
        if ch != "{":
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(t)):
            c = t[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(t[i:j + 1])
                    break
    for blk in out:
        try:
            o = json.loads(blk)
        except Exception:
            continue
        if isinstance(o, dict) and "turn_type" in o:
            return o
    return None


def verify_envelope(env_obj, tool_calls):
    """§1d 부분 구현 — 선언과 실제 행동의 정합만 본다(닫힌 술어)."""
    v = []
    if env_obj is None:
        return ["NO_ENVELOPE"]
    tt = env_obj.get("turn_type")
    acted = bool(tool_calls)
    if tt == "ACT" and not acted:
        v.append("R1_ACT_WITHOUT_CALL")          # 선언만 하고 행동 안 함
    if tt in ("ASK", "INFORM", "DONE") and acted:
        v.append("R2_CALL_WITHOUT_ACT")          # 행동했는데 다르게 선언
    na = env_obj.get("next_action")
    if acted:
        names = [c["function"]["name"] for c in tool_calls]
        if na and na not in names:
            v.append("R4_NEXT_ACTION_MISMATCH")
    if tt == "ASK" and not env_obj.get("ask"):
        v.append("R13_ASK_WITHOUT_SLOT")
    return v


# ── 본 루프 ─────────────────────────────────────────────────────────────────
def run_case(case, arm, seed, K=40):
    env = make_env()
    tools = tool_schemas(env)
    msgs = [{"role": "system", "content": system_prompt(env, arm)}]
    script, si = case["script"], 0
    msgs.append({"role": "user", "content": script[si]}); si += 1

    row = {"case": case["sim"], "task_id": case["task_id"], "source": case["source"],
           "arm": arm, "seed": seed, "turns": 0, "tool_calls": 0, "tool_errors": 0,
           "envelopes": 0, "envelope_ok": 0, "violations": [], "regens": 0,
           "horizon_hit": False, "script_exhausted": False, "deviated": False,
           # ★rev4 §10-5: 지평 초과 여부보다 **대본을 얼마나 진행했는지**가 비교 가능한 양이다.
           "script_len": len(script), "script_consumed": 1,
           # ★rev4 §10-6 부수관찰: 한 턴에 도구를 몰아 부르는 형태(multi-act)를 계측한다.
           "max_calls_in_turn": 0, "multiact_turns": 0,
           # ★파서-드롭 의심 계측: 도구를 불렀고 content는 비었는데 생성 토큰이 호출분보다 많은 턴.
           "drop_suspect_turns": 0, "gen_tokens": 0,
           "prose_chars": 0, "endpoint": BASE, "model": MODEL}

    for _ in range(K):
        row["turns"] += 1
        am = call_llm(msgs, seed, tools=tools)
        gen = int(LAST_USAGE.get("completion_tokens") or 0)
        row["gen_tokens"] += gen
        tcs = am.get("tool_calls") or []
        content = am.get("content") or ""
        if tcs and not content.strip():
            need = sum(len(c["function"]["name"]) + len(c["function"]["arguments"] or "")
                       for c in tcs) / 3.0          # 호출을 싣는 데 필요한 대략 토큰
            if gen > need + 20:
                row["drop_suspect_turns"] += 1

        # ── 선언 채널 ──
        env_obj = None
        if arm == "D_TWOPASS":
            # 2패스: 방금 한 행동을 봉투로 형식화(도구 미제공 → 문법 적용 가능)
            # ★프롬프트 누출 교정(2026-07-31 실측): 초판 설명문에 `tool_calls=[...]`가 들어 있었고
            #   모델이 **값 대신 키 이름 `tool_calls`를 복사**했다(또 `KB_search`를 `SEARCH_KB`로
            #   지어냈다). 그 상태로 잰 R4 불일치 107건은 **내 프롬프트의 산물**이지 선언 불충실이
            #   아니다 ⇒ 이름을 **축자 복사**하도록 명시하고 호출이 없으면 null을 요구한다.
            _names = [c["function"]["name"] for c in tcs]
            desc = ("This is the turn you just produced.\n"
                    "Message text: %r\n"
                    "Tools you actually called (exact names): %s\n\n"
                    "Emit the declaration envelope for that turn. For next_action, copy ONE name "
                    "VERBATIM from the list above (do not invent or abbreviate); if the list is "
                    "empty, use null."
                    % (content[:800], ", ".join(_names) if _names else "(none)"))
            fm = call_llm(msgs + [{"role": "user", "content": desc}], seed,
                          guided=ENVELOPE_SCHEMA, max_tokens=400)
            env_obj = parse_env_obj(fm.get("content"))
        elif arm == "B_SAYGUIDED" and not tcs:
            fm = call_llm(msgs + [{"role": "user",
                                   "content": "Emit the declaration envelope for this turn."}],
                          seed, guided=ENVELOPE_SCHEMA, max_tokens=400)
            env_obj = parse_env_obj(fm.get("content"))
        else:
            env_obj = parse_env_obj(content)

        if env_obj is not None:
            row["envelopes"] += 1
            row["prose_chars"] += len(str(env_obj.get("prose") or ""))
        viol = verify_envelope(env_obj, tcs)
        if not viol:
            row["envelope_ok"] += 1
        row["violations"] += viol

        if arm == "C_VERIFY" and viol:
            row["regens"] += 1
            fb = ("Your declaration violated: " + ",".join(viol) +
                  ". Re-emit so the declaration matches what you actually do.")
            am = call_llm(msgs + [{"role": "assistant", "content": content},
                                  {"role": "user", "content": fb}], seed, tools=tools)
            tcs = am.get("tool_calls") or []
            content = am.get("content") or ""

        # ── 행동 채널: 실제 env 실행 ──
        if tcs:
            row["max_calls_in_turn"] = max(row["max_calls_in_turn"], len(tcs))
            if len(tcs) > 1:
                row["multiact_turns"] += 1
            msgs.append({"role": "assistant", "content": content, "tool_calls": tcs})
            for c in tcs:
                row["tool_calls"] += 1
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except Exception:
                    args = {}
                try:
                    res = env.use_tool(name, **args)
                    out = str(res)
                    # ★env는 예외 대신 "Error: ..." 문자열을 돌려준다(V2 진단서 확인).
                    #   예외만 세면 실패가 조용히 성공으로 계상된다.
                    if out.strip().startswith("Error"):
                        row["tool_errors"] += 1
                except Exception as e:
                    out = "Error: %s" % (e,)
                    row["tool_errors"] += 1
                msgs.append({"role": "tool", "tool_call_id": c.get("id") or name,
                             "content": out[:20000]})
            continue                                   # 도구 턴 = 유저 발화 투입 안 함

        # 말 턴 → 대본 다음 발화 투입(설계서 §2-1-2)
        msgs.append({"role": "assistant", "content": content})
        if si >= len(script):
            row["script_exhausted"] = True
            break
        msgs.append({"role": "user", "content": script[si]}); si += 1
        row["script_consumed"] = si
    else:
        row["horizon_hit"] = True

    row["script_rate"] = round(row["script_consumed"] / max(1, row["script_len"]), 3)
    row["viol_counts"] = {v: row["violations"].count(v) for v in set(row["violations"])}
    row.pop("violations")
    row["db_hash"] = env.get_db_hash()
    return row


# ── V2 / V3 ─────────────────────────────────────────────────────────────────
def _a_real_user_id(env):
    """DB에서 실제 user id 하나. ★래퍼 구조 주의 — 최상위 컬렉션이 `{'data': {...}}` 형태다."""
    d = env.tools.db.model_dump()
    u = d.get("users") or {}
    inner = u.get("data") if isinstance(u, dict) and "data" in u else u
    return next(iter(inner)) if isinstance(inner, dict) and inner else None


def check_env_isolation():
    """V2 — 케이스별 새 env가 초기 DB 상태에서 시작하는지(get_db_hash 대조).

    ★양성 대조가 필수다: 변형이 해시를 **실제로 움직이는지** 먼저 보인다. 초판은 존재하지 않는
    user_id를 넘겨 도구가 `Error: User with ID 'data' not found`를 돌려줬고, 해시가 안 변한 채
    "PASS"가 찍혔다 — **대조가 죽은 검사는 격리를 증명하지 못한다**([[08]]).
    """
    print("=== V2: env 인스턴스 격리 ===")
    e1 = make_env(); h0 = e1.get_db_hash()
    uid = _a_real_user_id(e1)
    print("  초기 해시            %s" % h0[:32])
    print("  대조용 user id       %s" % uid)
    res = e1.use_tool("change_user_email", user_id=uid, new_email="probe_v2@example.com")
    print("  변형 결과            %s" % str(res).split("\n")[0][:60])
    h1 = e1.get_db_hash()
    ctrl = (h1 != h0)
    print("  변형 후 해시(같은 env) %s  %s" % (h1[:32], "변함 ✅" if ctrl else "★불변 = 양성대조 실패"))
    e2 = make_env(); h2 = e2.get_db_hash()
    print("  새 env 해시            %s  %s" % (h2[:32], "초기 복귀 ✅" if h2 == h0 else "❌오염"))
    ok = ctrl and (h2 == h0)
    print("  ⇒ V2 %s" % ("PASS" if ok else
                         ("FAIL — 양성대조 실패(검사 무효)" if not ctrl else "FAIL — 케이스 간 오염")))
    return ok


def smoke(seeds=1, n=2, K=6):
    print("=== V3: 스모크 (케이스 %d × 시드 %d × arm %d · K=%d) ===" % (n, seeds, len(ARMS), K))
    cases = sorted(_cases.load_cases(), key=lambda c: -c["tool_chars"])[:n]
    rows = []
    for c in cases:
        for arm in ARMS:
            for s in range(seeds):
                try:
                    r = run_case(c, arm, s, K=K)
                except Exception as e:
                    print("  ✗ %s/%s: %r" % (c["sim"], arm, e))
                    continue
                rows.append(r)
                print("  %s %-12s turns=%d calls=%d err=%d env=%d/%d regen=%d %s%s"
                      % (r["case"], arm, r["turns"], r["tool_calls"], r["tool_errors"],
                         r["envelope_ok"], r["turns"], r["regens"],
                         "지평초과 " if r["horizon_hit"] else "",
                         "대본소진" if r["script_exhausted"] else ""))
    # 발화 확인(설계서 V3 기준): arm별 개입이 실제로 일어났나
    print("\n  arm별 개입 실발화:")
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            print("    %-12s 행 0 ⚠" % arm); continue
        print("    %-12s 봉투 %d/%d턴 · regen %d · 도구호출 %d"
              % (arm, sum(r["envelopes"] for r in rs), sum(r["turns"] for r in rs),
                 sum(r["regens"] for r in rs), sum(r["tool_calls"] for r in rs)))
    ok = bool(rows) and all(sum(r["tool_calls"] for r in rows if r["arm"] == a) > 0 for a in ARMS)
    print("\n  ⇒ V3 %s (전 arm이 도구를 실제로 호출했는가 = 행동 채널 생존)"
          % ("PASS" if ok else "FAIL"))
    return rows, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-env", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--arms", default="", help="쉼표 구분 arm 부분집합(예: D_TWOPASS)")
    ap.add_argument("--max-user-turns", type=int, default=14,
                    help="대본이 이보다 길면 제외(지평 안에 못 끝냄). ★버린 수를 로그한다(무음 절단 금지)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if args.check_env:
        sys.exit(0 if check_env_isolation() else 1)
    if args.smoke:
        rows, ok = smoke()
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("[saved] %s" % args.out)
        sys.exit(0 if ok else 1)

    allc = _cases.load_cases()
    dropped = [c for c in allc if c["n_user"] > args.max_user_turns]
    pool = [c for c in allc if c["n_user"] <= args.max_user_turns]
    cases = sorted(pool, key=lambda c: -c["tool_chars"])[:args.cases]
    # ★무음 절단 금지: 무엇을 왜 뺐는지 남긴다.
    print("케이스 풀 %d → 대본 >%d턴 제외 %d건 → 남은 %d 중 장문 상위 %d 선택"
          % (len(allc), args.max_user_turns, len(dropped), len(pool), len(cases)))
    print("  제외된 긴 대본(유저턴 상위): %s"
          % sorted((c["n_user"] for c in dropped), reverse=True)[:12])
    print("  선택 케이스 유저턴: %s" % [c["n_user"] for c in cases])
    print("  서로 다른 task %d · 서로 다른 궤적 %d"
          % (len({c["task_id"] for c in cases}), len({c["sim"] for c in cases})))
    print("본런: 케이스 %d · 시드 %d · arm %d · K=%d → 총 %d런"
          % (len(cases), args.seeds, len(ARMS), args.K, len(cases) * args.seeds * len(ARMS)))
    global ARMS
    if args.arms:
        ARMS = [a for a in args.arms.split(",") if a]
        print("  ★arm 부분집합 실행: %s" % ARMS)
    out = args.out or "x13_su_mt_rows.jsonl"
    # ★행 즉시 영속([[30]] 결과소실 방지) — 중간에 죽어도 완료분은 남는다.
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for c in cases:
            for arm in ARMS:
                for s in range(args.seeds):
                    try:
                        r = run_case(c, arm, s, K=args.K)
                    except Exception as e:
                        print("  ✗ %s/%s/s%d: %r" % (c["sim"], arm, s, e), flush=True)
                        continue
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    n += 1
                    print("  [%d] %s %-12s s%d turns=%d calls=%d 대본 %d/%d(%.0f%%) 봉투 %d%s"
                          % (n, r["case"], arm, s, r["turns"], r["tool_calls"],
                             r["script_consumed"], r["script_len"], 100 * r["script_rate"],
                             r["envelopes"], " 지평초과" if r["horizon_hit"] else ""),
                          flush=True)
    print("[saved] %s (%d행)" % (out, n))
    print("⚠판정은 별도 도구로([[08]]) — 태스크 pass 채점 금지·ASK 페널티 금지·지평 초과율 1차 보고.")


if __name__ == "__main__":
    main()
