# -*- coding: utf-8 -*-
"""x733 — 048 **객체별 판정** 격리 프로브 (§17 이 처방하고 미측정으로 남긴 자리).

§17 축자: *"격리 프로브 형태는 '카드 4장의 상태를 주고 **각각** 무엇을 할지' —
A: 한 장씩 따로 물음 ↔ B: 넷을 한 번에. … 재고 나서 논한다."* 그 측정을 그대로 채운다([[62]]).

재는 것: 라이브가 실제로 받은 재료(손님 발화 + 도구 출력)를 주면 모델이 **동종 객체 4개에
서로 다른 쓰기 집합**을 붙일 수 있나. 048 의 잔여 결손은 MISSING 0 · WRONGARG 2 —
`log_credit_card_closure_reason_4521` 를 gold 가 남기지 않는 두 카드에 추가로 찍은 것이다.

규율:
  · 서브는 gold 를 보지 않는다([[23]]). **어휘조차** gold 가 아니라 우리가 실행한 호출 이름에서
    뽑는다 — 048 은 MISSING 0 이므로 gold ⊆ done 이고, 누수 없이 gold 를 덮는다.
  · 엔진은 **비교만** 한다([[10]]/[[52]]). 정답을 만들지 않는다.
  · 카드 목록은 궤적 텍스트가 아니라 **db.json(환경 권위)** 에서 읽는다([[59]] 패턴매칭 금지).

팔:
  A_EACH   장당 1콜 (전체 재료 + 그 카드 하나만 물음)   ← [[65]] 이 예측하는 자리
  B_ALL    4장을 한 콜에                                  ← 라이브와 같은 모양
  C_STRIP  도구 출력 제거(손님 발화만) · **부정통제**([[57]])

사용: x733_048_percard_probe.py <base_url> <model> <tag> [반복]
"""
import io
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2_forensic as F

TASK = "task_048"
USER_ID = "e3f4a5b6c7"
DB = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/db.json"
TOOL_CAP = 700


def cards_from_db(db_path, user_id):
    """환경 권위. 궤적을 뜯지 않는다."""
    d = json.load(open(db_path, encoding="utf-8"))
    rows = [v for v in d["credit_card_accounts"]["data"].values()
            if str(v.get("user_id")) == user_id]
    return sorted(rows, key=lambda r: r["account_id"])


def materials(sim, with_tools=True):
    out = []
    for m in sim.get("messages") or []:
        r, c = m.get("role"), str(m.get("content") or "").strip()
        if not c:
            continue
        if r == "user":
            out.append("CUSTOMER: " + c)
        elif r == "tool" and with_tools:
            out.append("TOOL RESULT: " + c[:TOOL_CAP])
    return "\n\n".join(out)


def gold_per_card(sim, ids):
    """gold 행을 **카드 id 값으로 묶기만** 한다 — 판정은 하지 않는다."""
    d = F.mutation_diff(sim)
    out = {i: set() for i in ids}
    for g in d.get("gold") or []:
        for v in (g.get("args") or {}).values():
            if isinstance(v, str) and v in out:
                out[v].add(g.get("name"))
    return out


TRUNC = {"n": 0}


def profile_env(model):
    """모델 프로필(`model_profiles/<model>.env`)의 선언을 읽는다 — 값을 코드에 적지 않는다.

    라이브가 쓰는 짝을 그대로 물려받아야 **정보-맞춘 격리**가 된다([[18]]). Q3.8 선언:
    `T2_JUDGE_MAX_TOKENS=8192` · `T2_PROBE_THINK_BUDGET=4096` (본응답은 8192/4096).
    모델을 바꾸면 프로필만 바뀌고 이 프로브는 그대로다.
    """
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_profiles",
                      model.replace("/", "__") + ".env")
    out = {}
    if os.path.exists(fn):
        for line in io.open(fn, encoding="utf-8"):
            line = line.strip()
            if not line.startswith("export "):
                continue
            kv = line[len("export "):].split("#", 1)[0].strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    else:
        print("  ⚠프로필 없음: %s — 서버 기본값으로 돈다" % fn)
    return out


def limits(model):
    """(max_tokens, thinking_token_budget) — 선언 우선, 없으면 사고예산은 상한의 절반."""
    e = profile_env(model)
    mt = int(e.get("T2_JUDGE_MAX_TOKENS") or e.get("T2_AGENT_MAX_TOKENS") or 8192)
    tb = e.get("T2_PROBE_THINK_BUDGET") or e.get("T2_THINK_BUDGET")
    tb = int(tb) if tb else think_budget(mt)
    if tb and tb >= mt:
        tb = think_budget(mt)          # 예산이 상한과 같으면 답이 전손된다(t2_run_gated 실측)
    return mt, tb


def think_budget(cap):
    """사고 예산 = **상한의 절반**(하한 256 · 반드시 상한 미만).

    정본은 `t2_run_gated._think_budget` 인데 **중첩 함수라 import 가 안 된다** — 정책만 옮긴다.
    축자(t2_run_gated.py:600 부근 · 2026-08-31): *"생성 순서가 [사고 …] → [답] 이라, 상한이
    사고 도중에 걸리면 **답이 통째로 사라진다** … 상한을 올리는 대신 **사고에만 예산**을 걸어
    답 자리를 반드시 남긴다."* 그리고 *"예산이 상한과 같으면 답 전손은 시간 문제"*.
    ⚠선행 실측 *"486토큰에서 답이 바뀐다"* 가 있으므로 더 조이지 않는다.
    """
    cap = int(cap or 0)
    return max(256, cap // 2) if cap > 512 else None


def ask(base, model, prompt, schema, timeout=900, max_tokens=None, tb=None):
    if max_tokens is None:
        max_tokens, tb = limits(model)
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "d", "schema": schema}}}
    _tb = tb if tb else think_budget(max_tokens)
    if _tb:
        body["thinking_token_budget"] = _tb
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    ch = (d.get("choices") or [{}])[0] or {}
    txt = (ch.get("message") or {}).get("content") or ""
    fr = ch.get("finish_reason")
    try:
        return json.loads(txt)
    except Exception:
        # ★계기는 회수돼야 존재한다([[30]]). 이 프로브의 첫 판이 전 팔 빈 배열로 나온 이유가
        #   `finish_reason=length`(사고가 상한 전량 소진·content None)였는데 조용히 삼켰다([[82]]).
        TRUNC["n"] += 1
        print("  ⚠응답 없음 finish_reason=%s content=%dB usage=%s"
              % (fr, len(txt), (d.get("usage") or {}).get("completion_tokens")), flush=True)
        return None


HEAD = ("You are reviewing a COMPLETED bank support conversation. Below are the customer's "
        "messages and the results of every tool the agent ran (these include the bank's own "
        "policy documents and account records).\n\n"
        "The customer holds these credit card accounts:\n%s\n\n"
        "Decide, PER ACCOUNT, which write actions the bank SHOULD have performed on that "
        "account, according to the bank's policies as shown in the tool results. "
        "Choose only from this list of action names:\n%s\n"
        'If no write action should have been performed on an account, answer ["none"].\n\n'
        "=== CONVERSATION MATERIALS ===\n%s\n=== END ===\n")

ONE = "\nAnswer for THIS account only: %s\n"
ALL = "\nAnswer for ALL of the accounts listed above.\n"


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8141/v1"
    model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.8-27B-FP8"
    tag = sys.argv[3] if len(sys.argv) > 3 else "bank_x731_qB_t3prime_20260901_1621"
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    sim = None
    for s in F.sims(tag):
        if s.get("task_id") == TASK:
            sim = s
            break
    if sim is None:
        print("sim 없음: %s / %s" % (tag, TASK))
        return 2

    rows = cards_from_db(DB, USER_ID)
    ids = [r["account_id"] for r in rows]
    desc = "\n".join("- %s | %s | balance %s" % (r["account_id"], r.get("card_type"),
                                                 r.get("current_balance")) for r in rows)
    vocab = sorted({t.get("name") for t in F.attempted_mutations(sim) if t.get("name")})
    vocab_txt = "\n".join("- " + v for v in vocab)
    gold = gold_per_card(sim, ids)

    MT, TB = limits(model)
    print("한도(프로필 선언): max_tokens=%s · thinking_token_budget=%s" % (MT, TB))

    mat_full = materials(sim, True)
    mat_strip = materials(sim, False)
    print("재료: full=%d자 strip=%d자 · 어휘 %d개 · 카드 %d장"
          % (len(mat_full), len(mat_strip), len(vocab), len(ids)))
    print("gold(카드별):")
    for i in ids:
        print("   %-22s %s" % (i, sorted(gold[i]) or ["none"]))

    enum = vocab + ["none"]
    sch_one = {"type": "object", "required": ["actions"], "properties": {
        "actions": {"type": "array", "items": {"type": "string", "enum": enum}}}}
    sch_all = {"type": "object", "required": ["cards"], "properties": {
        "cards": {"type": "array", "items": {"type": "object",
                  "required": ["account_id", "actions"], "properties": {
                      "account_id": {"type": "string", "enum": ids},
                      "actions": {"type": "array",
                                  "items": {"type": "string", "enum": enum}}}}}}}

    def norm(a):
        s = set(a or [])
        s.discard("none")
        return s

    score = {}
    for arm in ("A_EACH", "B_ALL", "C_STRIP"):
        hits = {i: 0 for i in ids}
        preds = {i: [] for i in ids}
        for rep in range(reps):
            if arm == "A_EACH":
                for i in ids:
                    r = ask(base, model, HEAD % (desc, vocab_txt, mat_full) + ONE % i,
                            sch_one, max_tokens=MT, tb=TB)
                    if r is None:
                        preds[i].append("무응답")
                        continue
                    p = norm(r.get("actions"))
                    preds[i].append(sorted(p))
                    hits[i] += 1 if p == gold[i] else 0
            else:
                mat = mat_full if arm == "B_ALL" else mat_strip
                r = ask(base, model, HEAD % (desc, vocab_txt, mat) + ALL, sch_all, max_tokens=MT, tb=TB)
                if r is None:
                    for i in ids:
                        preds[i].append("무응답")
                    print("  %s rep%d 무응답" % (arm, rep), flush=True)
                    continue
                got = {c.get("account_id"): norm(c.get("actions"))
                       for c in (r.get("cards") or [])}
                for i in ids:
                    p = got.get(i, set())
                    preds[i].append(sorted(p))
                    hits[i] += 1 if p == gold[i] else 0
            print("  %s rep%d 완료" % (arm, rep), flush=True)
        score[arm] = (hits, preds)

    print("\n=== 결과 (카드별 정확일치 / %d회) ===" % reps)
    for arm in ("A_EACH", "B_ALL", "C_STRIP"):
        hits, preds = score[arm]
        tot = sum(hits.values())
        print("%-8s 합계 %2d/%d" % (arm, tot, reps * len(ids)))
        for i in ids:
            print("   %-22s %d/%d  예측=%s" % (i, hits[i], reps, preds[i]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
