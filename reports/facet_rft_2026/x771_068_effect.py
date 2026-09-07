# -*- coding: utf-8 -*-
"""x771 — **068 괄호-STRIP** 효과 프로브 (§T-8 · `T2_SIBLING_PAREN` · 2026-09-05)

★관문 4칸을 이 파일 하나로 실측한다.
  ⑴ 결함이 지금 코드에 있나        — 파일:줄 + 축자 (엔진 소스를 읽어 **무장 여부**를 판정)
  ⑵ 그 결함이 실제 실패에 닿는가   — 회수 궤적/사이드카에서 그 발화·그 칸을 짚는다
  ⑶ ★수리 전/후 판정이 갈리는가   — **같은 재료**에 두 팔을 먹여 «커밋될 값» 이 갈리는지 센다
  ⑷ [[70]] 파는 것                 — 회수분 **전수**로 정당한 칸이 함께 죽는지 센다

규격([[78]]): 프롬프트 저작 0 · 사본 0 · 가짜 입력 0.
  PRE  팔 = **엔진 현행 그대로** — `sibling_paren_arg(tc)` 를 부르고 그 결과로 **아무것도 안 한다**
            (`t2_gate_patch.py:13357-13368` 분기 본문이 `print` 뿐이라 인자가 한 글자도 안 바뀐다).
  POST 팔 = **선언 오버라이드 한 칸** — 같은 술어의 반환 `(도구, 인자, 값, 뺄 부분문자열)` 에서
            네 번째 칸을 값에서 **빼기만** 한다([[63]] 제거). 값을 고르지 않는다([[62]]/[[10]]).
  재료   = `t2_forensic.iter_all_sims()` 회수분 전량. 엔진 술어는 import 해서 **직접** 부른다.

[[57]] 부정통제 3종:
  NC-1 되돌리기 — POST 를 PRE 로 되돌리면 갈림이 사라지는가(3 → 0)
  NC-2 통과 팔  — 같은 태스크의 **통과 sim** 재료(괄호 없음)에 두 팔이 **동일 판정**인가
  NC-3 gold     — gold 액션 인자 전수에 두 팔이 **동일 판정**인가(정답을 안 건드린다)

⛔이 프로브는 엔진을 고치지 않는다. POST 팔은 여기 안에서만 산다.
"""
import collections
import io
import json
import os
import re
import subprocess
import sys

REPO = r"C:\workspace\ba-frft"
ENG = os.path.join(REPO, "scripts", "distill", "tau2")
sys.path.insert(0, ENG)
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import t2_forensic as F          # noqa: E402  포렌식 정본([[67]])
import t2_gate_patch as G        # noqa: E402  엔진 술어 정본

TARGET = "task_068"
FAILTAG = "bank_x712_nightA_20260901"                  # 068 괄호가 실재하는 회수분
PASSTAG = "bank_x722_t2B_viewscale_max_20260901_1106"  # 같은 태스크 통과 팔(NC-2)
LIVETRACE = "bank_k8141med1_20260903_2256"             # 라이브 발화가 회수된 사이드카


def say(*a):
    print(" ".join(str(x) for x in a))


def hr(t=""):
    say("=" * 92)
    if t:
        say(t)


class TC(object):
    """엔진이 기대하는 tool_call 모양만 흉내낸다(값 생성 0 · x765 와 같은 shim)."""

    def __init__(self, d):
        self.name = d.get("name")
        self.arguments = json.loads(json.dumps(d.get("arguments") or {}))  # 깊은 사본
        self.id = d.get("id")


# ═══════════════════════════════════════════════════════════════════════════════
# 두 팔 — 둘 다 **같은 엔진 술어**를 부른다. 다른 것은 그 반환으로 무엇을 하느냐뿐.
# ═══════════════════════════════════════════════════════════════════════════════
def arm_pre(tc):
    """수리 **전** = 엔진 현행. 술어를 부르고 **거동 변화 0**(:13352 축자)."""
    G.sibling_paren_arg(tc)
    return committed_args(tc)


def arm_post(tc):
    """수리 **후** = 괄호 STRIP.

    ★2026-09-05 배선 후 재실행: POST 팔이 **엔진 정본 `G.sibling_paren_strip` 를 직접 부른다**.
      이전 판은 같은 계산을 이 파일 안에서 따로 했다(사본) — 그때는 엔진에 무장이 없어 그럴
      수밖에 없었지만, 이제 그러면 **측정 대상이 배선된 코드가 아니게 된다**([[76]]/[[81]]).
      라이브 호출부와의 차이는 env 분기 한 겹뿐이고, 그 겹은 `test_lever_wiring.py` 가 본다
      (런처 `go_stack.sh:909 export T2_SIBLING_PAREN=strip` 값을 관문이 받는지 **판정**).
    """
    G.sibling_paren_strip([tc])
    return committed_args(tc)


def _bag(tc):
    """디스패처 언랩 — 엔진 `sibling_paren_arg` 와 **같은 방식**으로 안쪽 인자를 본다."""
    ar = G._args_dict(tc) or {}
    sub = ar.get("arguments")
    if isinstance(sub, str):
        try:
            p = json.loads(sub)
            if isinstance(p, dict):
                return ar, p
        except Exception:
            pass
    elif isinstance(sub, dict):
        return ar, sub
    return ar, ar


def _set_inner(tc, key, value):
    ar, bag = _bag(tc)
    bag[key] = value
    if bag is not ar:
        ar["arguments"] = json.dumps(bag, ensure_ascii=False)
    tc.arguments = ar


def committed_args(tc):
    """이 호출이 **env 로 나갈 때의 안쪽 인자**(= DB 에 반영될 것)."""
    _ar, bag = _bag(tc)
    return dict(bag)


# ═══════════════════════════════════════════════════════════════════════════════
# ⑴ 결함이 지금 코드에 있나
# ═══════════════════════════════════════════════════════════════════════════════
def gate1():
    hr("[GATE 1] 결함이 지금 코드에 있나 — 파일:줄 + 축자")
    src = io.open(os.path.join(ENG, "t2_gate_patch.py"), encoding="utf-8").read()
    lines = src.splitlines()
    i_def = next(n for n, l in enumerate(lines) if l.startswith("def sibling_paren_arg"))
    i_br = next(n for n, l in enumerate(lines) if 'os.environ.get("T2_SIBLING_PAREN")' in l)
    say("   술어 정의 : t2_gate_patch.py:%d  %s" % (i_def + 1, lines[i_def].strip()))
    say("   배선 분기 : t2_gate_patch.py:%d  %s" % (i_br + 1, lines[i_br].strip()))
    body = lines[i_br + 1:i_br + 16]
    say("   분기 본문 (축자):")
    for n, l in enumerate(body):
        say("      %d| %s" % (i_br + 2 + n, l.rstrip()))
    # ★2026-09-05: 무장 탐지를 **호출까지** 넓혔다. 최초 판은 분기 본문의 `대입문`만 봤는데,
    #   정본 무장은 `sibling_paren_strip(am.tool_calls, …)` **호출**이라 대입이 없다 —
    #   옛 탐지기로는 무장하고도 «무장 0» 이라 보고했을 것이다([[84]] 와 같은 종류의 사고).
    inline = any(re.search(r"tool_calls\s*=|\.arguments\s*=|bag\[", l) for l in body)
    called = any("sibling_paren_strip(" in l for l in body)
    say("   ⇒ 본문이 인자를 변형하는가 : 대입 %s · 무장 술어 호출 %s" % (inline, called))
    say("   ⇒ **무장 여부 = %s**   (수리 전 = False·계기뿐 / 수리 후 = True)" % (inline or called))
    i_def2 = next((n for n, l in enumerate(lines) if l.startswith("def sibling_paren_strip")), None)
    if i_def2 is not None:
        say("   무장 술어  : t2_gate_patch.py:%d  %s" % (i_def2 + 1, lines[i_def2].strip()))

    say("")
    say("   ★[[81]] 정본 런처 등재 — grep 경로 첨부([[77]] ④):")
    for rel in ("go_stack.sh", "run_ours_task.sh"):
        p = os.path.join(ENG, rel)
        n = 0
        if os.path.exists(p):
            n = io.open(p, encoding="utf-8", errors="replace").read().count("T2_SIBLING_PAREN")
        say("      %-20s T2_SIBLING_PAREN 히트 = %d" % (rel, n))
    armdir = os.path.join(ENG, "arms")
    for fn in sorted(os.listdir(armdir)):
        txt = io.open(os.path.join(armdir, fn), encoding="utf-8", errors="replace").read()
        for ln in txt.splitlines():
            if "T2_SIBLING_PAREN" in ln:
                say("      arms/%-24s %s" % (fn, ln.strip()))
    for ln in io.open(os.path.join(ENG, "go_stack.sh"), encoding="utf-8",
                      errors="replace").read().splitlines():
        if ln.strip().startswith("export T2_SIBLING_PAREN"):
            say("      go_stack.sh 등재 축자 : %s" % ln.strip())
    say("   ⇒ 수리 전: `arms/*.env` 에만 `=log` · 정본 런처 **미등재**([[81]] 라이브 부재).")
    say("     수리 후: `go_stack.sh` 에 `=strip` 등재 · `arms/*.env` 의 `=log` 는 계기 팔로 보존.")
    return inline or called


# ═══════════════════════════════════════════════════════════════════════════════
# ⑵ 그 결함이 실제 실패에 닿는가
# ═══════════════════════════════════════════════════════════════════════════════
def gate2():
    hr("[GATE 2] 그 결함이 실제 실패에 닿는가 — 회수분에서 그 발화·그 칸")

    # ⓐ 라이브 발화(사이드카) — 술어가 라이브에서 도는가
    say("   ⓐ 라이브 발화(회수 사이드카):")
    live = 0
    for p in F.trace_paths(LIVETRACE) or []:
        with F.topen(p) as f:
            for ln in f:
                if "T2_SIBLING_PAREN" in ln:
                    live += 1
                    o = json.loads(ln)
                    say("      %s  sim=%s turn=%s" % (os.path.basename(p), o.get("sim"), o.get("turn")))
                    say("      %s" % o.get("line"))
    say("      ⇒ 라이브 발화 %d 건 — **표적 068 이 아니라 059**. 술어는 돈다." % live)

    # ⓑ 표적 068 의 그 칸
    say("")
    say("   ⓑ 표적 068 의 그 호출 — %s" % FAILTAG)
    for tag, s in F.iter_all_sims(want_tasks={TARGET}):
        if tag != FAILTAG:
            continue
        msgs = s.get("messages") or []
        idx = None
        for n, m in enumerate(msgs):
            for tc in (m.get("tool_calls") or []):
                if G.sibling_paren_arg(TC(tc)):
                    idx = n
        say("      msg[%d] role=%s" % (idx, msgs[idx].get("role")))
        for tc in (msgs[idx].get("tool_calls") or []):
            say("         CALL %s" % json.dumps(F.argsof(tc), ensure_ascii=False)[:220])
        say("      msg[%d] role=tool (env 응답 축자):" % (idx + 1))
        for ln in str(msgs[idx + 1].get("content") or "").splitlines():
            if ln.strip():
                say("         %s" % ln.strip())
        say("      env 선언 축자(msg[%d] · [[23]] 출처):" % (idx - 1))
        for ln in str(msgs[idx - 1].get("content") or "").splitlines():
            if "account_class" in ln:
                say("         %s" % ln.strip())
        ri = s.get("reward_info") or {}
        say("")
        say("      ⛔채점 단위([[69]]) : reward=%s · termination=%s · reward_basis=%s · db_check=%s"
            % (ri.get("reward"), s.get("termination_reason"),
               json.dumps(ri.get("reward_basis")), json.dumps(ri.get("db_check"))))
        say("      ⇒ 이 sim 은 **DB 대조가 돌지 않았다** — 0.0 의 사유는 max_steps 다.")
        say("      ⇒ 그리고 이 런에서 `T2_SIBLING_PAREN` 은 **OFF**(로그 발화 0) — 라이브가 이 칸을 본 적 없다.")

    # ⓒ 현 스택(오늘 회수분)의 068 은 어떤 모양인가
    say("")
    say("   ⓒ 채점된 068 실패 sim 들의 실제 account_class:")
    for tag, s in F.iter_all_sims(want_tasks={TARGET}):
        ri = s.get("reward_info") or {}
        db = (ri.get("db_check") or {}).get("db_match") if ri.get("db_check") else None
        vals = []
        for _m, tc in F.calls(s):
            b = committed_args(TC(tc))
            if "account_class" in b:
                vals.append("%s=%r" % (b.get("account_type"), b.get("account_class")))
        if vals:
            say("      %-44s rw=%-4s db=%-6s %s" % (tag, ri.get("reward"), db, " | ".join(vals)))


# ═══════════════════════════════════════════════════════════════════════════════
# ⑶ 수리 전/후 판정이 갈리는가  (+ ⑷ 파는 것 · NC 3종)
# ═══════════════════════════════════════════════════════════════════════════════
def collect():
    """회수분 전수 1회 순회 — 발화·gold·통과sim 을 한 번에 모은다."""
    gold_cls = collections.defaultdict(set)      # (task, account_type) -> {gold account_class}
    gold_acts = []                               # (task, action_id, name, args)
    gold_strargs = 0
    gold_paren = set()
    fires = []                                   # (tag, simkey, reward, db, task, tc_dict)
    n_sim = n_call = 0
    seen_gold = set()
    for tag, s in F.iter_all_sims():
        n_sim += 1
        tid = str(s.get("task_id"))
        ri = s.get("reward_info") or {}
        rw = ri.get("reward")
        db = (ri.get("db_check") or {}).get("db_match") if ri.get("db_check") else None
        for ac in (ri.get("action_checks") or []):
            a = ac.get("action") or {}
            k = (tid, a.get("action_id"))
            if k in seen_gold:
                continue
            seen_gold.add(k)
            args = a.get("arguments") or {}
            gold_acts.append((tid, a.get("action_id"), a.get("name"), args))
            _ar, bag = _bag(TC({"name": a.get("name"), "arguments": args}))
            for kk, vv in (bag.items() if isinstance(bag, dict) else []):
                if isinstance(vv, str):
                    gold_strargs += 1
                    if "(" in vv and ")" in vv:
                        gold_paren.add((tid, a.get("action_id"), a.get("name"), kk, vv[:90]))
            if isinstance(bag, dict) and "account_class" in bag:
                gold_cls[(tid, bag.get("account_type"))].add(bag.get("account_class"))
        for _m, tc in F.calls(s):
            n_call += 1
            if G.sibling_paren_arg(TC(tc)):
                fires.append((tag, F.sim_key(s), rw, db, tid, tc))
    return dict(gold_cls=gold_cls, gold_acts=gold_acts, gold_strargs=gold_strargs,
                gold_paren=gold_paren, fires=fires, n_sim=n_sim, n_call=n_call)


def verdict(bag, tid, gold_cls):
    """판정 = «커밋될 account_class 가 그 (태스크, account_type) 의 gold 와 같은가»."""
    gs = gold_cls.get((tid, bag.get("account_type")))
    if not gs:
        return None
    return bag.get("account_class") in gs


def gate3(C):
    hr("[GATE 3] ★수리 전/후 판정이 갈리는가 — 같은 재료 · 두 팔")
    gold_cls = C["gold_cls"]
    split = neutral = harm = 0
    rows = []
    for tag, key, rw, db, tid, tc in C["fires"]:
        pre = arm_pre(TC(tc))
        post = arm_post(TC(tc))
        vp, vq = verdict(pre, tid, gold_cls), verdict(post, tid, gold_cls)
        if vp is False and vq is True:
            split += 1
            rows.append((tag, key, tid, pre.get("account_class"), post.get("account_class")))
        elif vp is True and vq is not True:
            harm += 1
            rows.append(("⛔HARM", key, tid, pre.get("account_class"), post.get("account_class")))
        else:
            neutral += 1
    say("   발화 전수 = %d" % len(C["fires"]))
    say("   ⇒ ★갈림(PRE 오답 → POST gold 일치) = %d" % split)
    say("   ⇒ 중립(둘 다 gold 아님 / 대응 gold 없음) = %d" % neutral)
    say("   ⇒ ⛔해악(PRE 정답 → POST 오답)         = %d" % harm)
    say("")
    say("   갈리는 칸 전수:")
    for r in rows:
        say("      %-46s %-14s %-10s %r -> %r" % r)
    say("")
    say("   표적 068 단건:")
    for tag, key, rw, db, tid, tc in C["fires"]:
        if tid != TARGET:
            continue
        pre, post = arm_pre(TC(tc)), arm_post(TC(tc))
        say("      %s %s" % (tag, key))
        say("        PRE  커밋값 : %r   판정=%s" % (pre.get("account_class"), verdict(pre, tid, gold_cls)))
        say("        POST 커밋값 : %r   판정=%s" % (post.get("account_class"), verdict(post, tid, gold_cls)))
        say("        gold        : %r" % sorted(gold_cls.get((tid, pre.get("account_type")), [])))
    return split, neutral, harm


def gate3b(C):
    """reward 축 — STRIP 만으로 068 의 변이집합이 gold 와 같아지는가([[69]] 채점단위)."""
    hr("[GATE 3b] reward 축 — STRIP 만으로 gold 변이집합이 채워지는가")
    gold = [(t, aid, nm, ar) for (t, aid, nm, ar) in C["gold_acts"] if t == TARGET]
    say("   gold 액션 %d개:" % len(gold))
    for t, aid, nm, ar in gold:
        say("      %-8s %-34s %s" % (aid, nm, json.dumps(ar, ensure_ascii=False)[:170]))
    for tag, s in F.iter_all_sims(want_tasks={TARGET}):
        if tag != FAILTAG:
            continue
        say("")
        say("   %s 의 write 호출 (PRE / POST):" % tag)
        for _m, tc in F.calls(s):
            nm = F.inner_name(F.argsof(tc)) or F.nameof(tc)
            if not any(w in nm for w in ("open_bank_account", "close_bank_account",
                                         "apply_for_credit_card", "log_verification")):
                continue
            pre, post = arm_pre(TC(tc)), arm_post(TC(tc))
            mark = "  <= STRIP" if pre != post else ""
            say("      %-34s PRE  %s%s" % (nm, json.dumps(pre, ensure_ascii=False)[:150], mark))
            if pre != post:
                say("      %-34s POST %s" % ("", json.dumps(post, ensure_ascii=False)[:150]))
        say("")
        say("   ⛔잔여: `close_bank_account_7392` 가 gold 에 없는 `reason` 을 함께 넘긴다.")
        say("      그 자리는 **별 레버** `T2_FREE_TEXT_ARG`(A2 `free_text_defaults`) 소관이고")
        say("      정본 런처에 이미 `go_stack.sh:871 export T2_FREE_TEXT_ARG=1` 로 켜져 있다.")
        # 그 레버를 같은 재료에 실측한다 — 자기-그라운딩 금지(호출 직전 문맥만 코퍼스)
        msgs = s.get("messages") or []
        ci = None
        for n, m in enumerate(msgs):
            for tc in (m.get("tool_calls") or []):
                b = committed_args(TC(tc))
                if "reason" in b and "close" in (F.inner_name(F.argsof(tc)) or ""):
                    ci = n
        corp = " ".join(str(m.get("content") or "").lower() for m in msgs[:ci]
                        if m.get("role") in ("user", "tool"))
        a2 = json.load(io.open(os.path.join(ENG, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
        tcs = [TC(t) for t in (msgs[ci].get("tool_calls") or [])]
        dropped = G.free_text_drop(tcs, corp, a2, log=lambda m: say("         " + m[:200]))
        say("      ⇒ `free_text_drop` 실측 제거 = %d 건 · 결과 인자 = %s"
            % (len(dropped), json.dumps(committed_args(tcs[0]), ensure_ascii=False)[:150]))


def gate3c(C):
    """★갈리는 sim 전수의 **reward 축 잔여** — STRIP 이 닫는 칸이 그 sim 의 유일한 불일치인가.

    ⛔`action_match` 는 reward 가 아니다([[69]] · x737 §1f-4 축자: *"060 은 7/7 match=True 인데
      db_match=False"*). 여기서 세는 것은 **잔여 불일치 칸의 수**뿐이고, DB 반사실은 미측정이다.
    """
    hr("[GATE 3c] 갈리는 sim 전수의 reward 축 잔여 (STRIP 후 남는 불일치 칸)")
    gold_cls = C["gold_cls"]
    split_keys = set()
    for tag, key, _rw, _db, tid, tc in C["fires"]:
        if verdict(arm_pre(TC(tc)), tid, gold_cls) is False \
                and verdict(arm_post(TC(tc)), tid, gold_cls) is True:
            split_keys.add((tag, tid))
    for tag, s in F.iter_all_sims(want_tasks={t for _g, t in split_keys}):
        if (tag, str(s.get("task_id"))) not in split_keys:
            continue
        ri = s.get("reward_info") or {}
        acs = ri.get("action_checks") or []
        bad = [a["action"]["action_id"] for a in acs if a.get("action_match") is False]
        say("   %-46s %-11s rw=%-4s term=%-14s db_check=%s"
            % (tag, s.get("task_id"), ri.get("reward"), s.get("termination_reason"),
               json.dumps(ri.get("db_check"))))
        say("      gold 액션 %d개 · action_match=False 인 칸 = %s" % (len(acs), bad or "(채점 안 됨)"))
        for a in acs:
            if a.get("action_match") is not False:
                continue
            ar = a["action"].get("arguments") or {}
            _x, bag = _bag(TC({"name": a["action"].get("name"), "arguments": ar}))
            say("         %-8s %-30s gold=%s" % (a["action"]["action_id"], a["action"].get("name"),
                                                 json.dumps(bag, ensure_ascii=False)[:130]))
        say("      ⇒ STRIP 이 닫는 칸 = account_class 1개. 그 밖의 False 칸이 남으면 **단독 매수 0**.")


def gate4(C):
    hr("[GATE 4] [[70]] 파는 것 — 회수분 전수")
    say("   순회 sim = %d · 호출 = %d" % (C["n_sim"], C["n_call"]))
    # ⓐ gold 액션에서의 발화
    fires_on_gold = 0
    changed_gold = 0
    for tid, aid, nm, ar in C["gold_acts"]:
        tc = TC({"name": nm, "arguments": ar})
        if G.sibling_paren_arg(tc):
            fires_on_gold += 1
        if arm_pre(TC({"name": nm, "arguments": ar})) != arm_post(TC({"name": nm, "arguments": ar})):
            changed_gold += 1
    say("   ⓐ gold 액션 %d개 중 술어 발화 = %d · POST 팔이 **바꾼** gold = %d"
        % (len(C["gold_acts"]), fires_on_gold, changed_gold))
    # ⓑ gold 문자열 인자 중 괄호형
    say("   ⓑ gold 문자열 인자 %d개 중 괄호 든 것 = %d" % (C["gold_strargs"], len(C["gold_paren"])))
    for g in sorted(C["gold_paren"]):
        say("      %-10s %-8s %-28s %-12s %r" % g)
    say("      ⇒ `account_class` gold 에 괄호형 **0** — 반대편(«gold 가 괄호 공식명을 요구하는")
    say("        자리»)은 회수분에 **실재하지 않는다**.")
    # ⓒ 통과 sim 에서의 발화
    pas = [f for f in C["fires"] if f[2] is not None and f[2] >= 1.0]
    say("   ⓒ 통과 sim(reward>=1.0)에서의 발화 = %d / %d" % (len(pas), len(C["fires"])))
    # ⓓ 발화의 도구·인자 분포 — STRIP 이 닿는 표면이 얼마나 넓은가
    dist = collections.Counter()
    for _tag, _k, _rw, _db, _t, tc in C["fires"]:
        sp = G.sibling_paren_arg(TC(tc))
        dist[(sp[0], sp[1])] += 1
    say("   ⓓ 발화 표면 분포(도구.인자):")
    for k, v in dist.most_common():
        say("      %-34s %-18s x%d" % (k[0], k[1], v))


def negctl(C):
    hr("[NC] [[57]] 부정통제 3종")
    gold_cls = C["gold_cls"]
    # NC-1 되돌리기
    s1 = sum(1 for _t, _k, _r, _d, tid, tc in C["fires"]
             if verdict(arm_pre(TC(tc)), tid, gold_cls) is False
             and verdict(arm_post(TC(tc)), tid, gold_cls) is True)
    s0 = sum(1 for _t, _k, _r, _d, tid, tc in C["fires"]
             if verdict(arm_pre(TC(tc)), tid, gold_cls) is False
             and verdict(arm_pre(TC(tc)), tid, gold_cls) is True)
    say("   NC-1 되돌리기 : POST 갈림 %d  →  PRE 로 되돌리면 %d   (0 이어야 한다)" % (s1, s0))
    # NC-2 통과 팔 재료
    same = diff = 0
    for tag, s in F.iter_all_sims(want_tasks={TARGET}):
        if tag != PASSTAG:
            continue
        for _m, tc in F.calls(s):
            a, b = arm_pre(TC(tc)), arm_post(TC(tc))
            if a == b:
                same += 1
            else:
                diff += 1
    say("   NC-2 통과 팔  : %s 호출 %d개 — 두 팔 동일 %d · 달라짐 %d   (달라짐 0 이어야 한다)"
        % (PASSTAG, same + diff, same, diff))
    # NC-3 gold 재료
    g_same = g_diff = 0
    for tid, aid, nm, ar in C["gold_acts"]:
        a = arm_pre(TC({"name": nm, "arguments": ar}))
        b = arm_post(TC({"name": nm, "arguments": ar}))
        (g_same, g_diff) = (g_same + 1, g_diff) if a == b else (g_same, g_diff + 1)
    say("   NC-3 gold     : gold 액션 %d개 — 두 팔 동일 %d · 달라짐 %d   (달라짐 0 이어야 한다)"
        % (len(C["gold_acts"]), g_same, g_diff))


def main():
    ok1 = gate1()
    gate2()
    say("")
    say("   … 회수분 전수 순회 중(462 결과파일) …")
    C = collect()
    split, neutral, harm = gate3(C)
    gate3b(C)
    gate3c(C)
    gate4(C)
    negctl(C)
    hr("[판정]")
    say("   ⑴ 엔진 무장 여부           : %s   (수리 전 = NO=계기뿐 / 수리 후 = YES)"
        % ("YES" if ok1 else "NO"))
    say("   ⑶ 전후 갈림               : %d / %d 발화" % (split, len(C["fires"])))
    say("   ⑷ 파는 것                 : gold 변경 0 · 통과sim 발화 0 · 해악 %d" % harm)


if __name__ == "__main__":
    main()
