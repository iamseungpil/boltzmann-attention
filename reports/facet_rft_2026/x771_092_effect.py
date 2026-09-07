# -*- coding: utf-8 -*-
"""x771 - 092 효과 프로브 (`[OPERATOR-SCOPE]` 가 **아무 것도 수행하지 않는 unlock** 에서 발화)

★조사 명부는 092 를 «D14 계열» 로 배정했으나 기전이 다르다고 경고했다. 이 파일이 그 경고를
  실측으로 확정한다 - 092 의 우리-층 결함은 **재생성의 게이트 재진입**(D14)이 아니라
  `resolve_operator` 의 **발화 자리**다.

관문 4칸:
  (1) 결함이 지금 코드에 있나        - 파일:줄 + 축자 + [[77]] 검색 경로
  (2) 그 결함이 실제 실패에 닿는가   - 회수 log/fb/trace/results 축자 4중 교차
  (3) ★수리 전/후 판정이 갈리는가   - **엔진 정본 `resolve_write` 를 그대로 부른다**.
                                      PRE = 현재 소스 · POST = 같은 소스에 기계적 3-hunk 패치.
  (4) [[70]] 파는 것                 - 회수분 **전수** 계수

규격([[78]]): 프롬프트 저작 0 · 사본 0.
  두 팔 모두 **엔진 자신의 소스 바이트를 실행**한다(POST 는 str.replace 로 hunk 3개만 적용).
  진입점은 라이브가 부르는 그 함수다 - `t2_gate_patch.py:10300` 의
  `_rz.resolve_write(getattr(c,"name",None), _args_dict(c), state.messages, a2, self, la, UserMessage)`.
  선언 오버라이드는 **한 칸뿐**: `formalize_intent_tool` -> 회수된 라이브 값
  `close_debit_card_4721` (trace 축자
  `[T2_RESOLVE] operator-scope: 지목 대신 범위 표면화 (reset_debit_card_pin_6284, close_debit_card_4721)`).
  LLM 재질의 0.
  재료는 가짜 입력 0 - `sim_results/bank_lost2_viewmax2_20260903_1750` 회수분의 **실제 호출**.

[[57]] 부정통제 3종:
  NC-1 되돌리기 - PRE 팔을 같은 재료에 다시 먹이면 deny 가 돌아오는가
  NC-2 좁이     - POST 팔에 `call_tool = dispatch_tool` 을 주면 **여전히 deny** 인가
                  (= 수리가 끄기가 아니라 자리 옮기기라는 증거 · [[60]])
  NC-3 출처     - `a2.eplan.dispatch_tool` 을 지우면 POST 가 PRE 와 **판정 동일**인가
                  (= 갈림의 출처가 A2 선언이지 엔진 하드코딩이 아니라는 증거 · [[05]])
"""
import glob
import gzip
import io
import json
import os
import re
import subprocess
import sys
import time
import types
from collections import Counter

REPO = r"C:\workspace\ba-frft"
ENG = os.path.join(REPO, "scripts", "distill", "tau2")
SIMS = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
RUN = "bank_lost2_viewmax2_20260903_1750"
TASK = "task_092"
SIMTAG = "task_092#s626729"
DOMAIN = "banking_knowledge"

# 라이브 발사 환경([[30]] 계기는 회수돼야 존재 · go_stack.sh:588/713/816 축자)
os.environ.setdefault("T2_PROV_OURS", "1")
os.environ.setdefault("T2_SCOPE_ALL", "0")
os.environ.setdefault("T2_KEEP_DENY_BODY", "1")
sys.path.insert(0, ENG)

FAIL = []


def say(*a):
    print(" ".join(str(x) for x in a))


def chk(cond, label):
    say("   %s %s" % ("PASS" if cond else "**FAIL**", label))
    if not cond:
        FAIL.append(label)
    return cond


def _git(*args):
    return subprocess.run(["git", "-C", REPO] + list(args), capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def _gz(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="replace")


def load_sim():
    p = os.path.join(SIMS, RUN + ".results.json.gz")
    d = json.load(_gz(p))
    for s in d["simulations"]:
        if str(s.get("task_id")) == TASK:
            return s
    raise SystemExit("회수분에 %s 가 없다: %s" % (TASK, p))


def load_fb():
    p = os.path.join(SIMS, "fb_" + RUN + ".jsonl.gz")
    out = []
    for l in _gz(p):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("simtag") == SIMTAG:
            out.append(r)
    return out


def load_log():
    return _gz(os.path.join(SIMS, RUN + ".log.gz")).read()


def load_trace():
    out = []
    for l in _gz(os.path.join(SIMS, "trace_" + RUN + ".jsonl.gz")):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


# =========================================================================
# (1) 결함이 지금 코드에 있나
# =========================================================================
def gate1():
    say("=" * 92)
    say("[GATE 1] 결함이 지금 코드에 있나 - 파일:줄 + 축자")
    src = io.open(os.path.join(ENG, "t2_resolve.py"), encoding="utf-8").read()
    lines = src.split("\n")

    def find(pat, label):
        rx = re.compile(pat)
        for i, ln in enumerate(lines, 1):
            if rx.search(ln):
                say("   t2_resolve.py:%-5d %s" % (i, ln.strip()[:160]))
                return i, ln
        say("   **없음** %s" % label)
        FAIL.append("소스 앵커 부재: " + label)
        return None, ""

    n_sig, l_sig = find(r"^def resolve_operator\(", "resolve_operator 서명")
    say("     ^ 서명에 **호출 도구 이름(call_tool) 인자가 없다** - 발화 자리를 구분할 수 없다")
    n_ro, _ = find(r"^def resolve_operand\(", "resolve_operand 서명")
    n_pass, l_pass = find(r"^\s+return resolve_operator\(opspec, args_dict, msgs",
                          "resolve_operand -> resolve_operator 전달")
    say("   [MODE] %s" % ("**배선 후** — 아래 (a)(b) 는 「수리가 실렸나」를 묻는다"
                             if WIRED else "배선 전 — 아래 (a)(b) 는 「결함이 있나」를 묻는다"))
    # ★서명은 2줄이다(149-150) - 첫 줄만 보면 call_tool 을 놓친다.
    _sig2 = " ".join(lines[n_sig - 1:n_sig + 1])
    say("     (서명 2줄 이어읽기) %s" % _sig2.strip()[:200])
    chk(("call_tool" in _sig2.split("(", 1)[-1]) == WIRED,
        ("수리(a) resolve_operator 가 **호출 도구를 받는다**" if WIRED
         else "결함(a) resolve_operator 가 **호출 도구를 받지 않는다**"))
    chk(("call_tool=tool" in l_pass or "call_tool=tool" in "".join(lines[n_pass:n_pass + 2]))
        == WIRED,
        ("수리(b) resolve_operand 가 `tool` 을 **전달한다** (t2_resolve.py:%s)" % n_pass if WIRED
         else "결함(b) resolve_operand 는 `tool` 을 받아 놓고 **전달하지 않는다** "
              "(t2_resolve.py:%s)" % n_pass))

    n_w, l_w = find(r"if not _g\._is_effective_write\(", "write-only 침묵 술어")
    say("     ^ 판정 대상이 **operand(chosen)** 이다 - 실제로 실행하는 호출이 무엇인지는 안 본다")
    chk("_SUFFIX_RE.sub" in l_w and "chosen" in l_w,
        "결함(c) 되돌릴 수 없음 판정을 **operand** 로만 한다(호출 종류 무시)")

    import gate_interpreter as G
    a2 = G.load_domain_a2(DOMAIN)
    if a2 is None:
        raise SystemExit("A2 로드 실패: %s" % DOMAIN)
    ops = (a2 or {}).get("operands") or {}
    disp = ((a2 or {}).get("eplan") or {}).get("dispatch_tool")
    say("   a2(%s).operands 키 = %s" % (DOMAIN, sorted(ops)))
    say("   a2.eplan.dispatch_tool = %r  <- 선언은 **이미** 실행 자리를 구분해 두고 있다" % disp)
    chk(set(ops) >= {"unlock_discoverable_agent_tool", "call_discoverable_agent_tool"},
        "결함(d) A2 가 unlock/dispatch 에 **동일한** operator spec 을 건다")
    chk(bool(disp), "수리 재료 실재: a2.eplan.dispatch_tool 선언됨(엔진 리터럴 0 · [[05]])")

    say("   [[77]] 검색 경로 - 이 수리가 repo 에 없다는 근거:")
    for pat in ("T2_SCOPE_UNLOCK", "scope_call_kind", "call_tool", "operator_scope_at_write"):
        r = subprocess.run(["git", "-C", REPO, "grep", "-rn", pat, "--",
                            "scripts/", "*.sh"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        n = len([x for x in (r.stdout or "").split("\n") if x.strip()])
        say("     git grep -rn %-24r -- scripts/ *.sh  ->  %d 히트%s"
            % (pat, n, ("  (배선 후: 이 중 우리 수리 3줄 포함)"
                        if (WIRED and pat == "call_tool") else "")))
    lg = [x for x in _git("log", "--oneline", "-S", "dispatch_tool", "--",
                          "scripts/distill/tau2/t2_resolve.py").split("\n") if x.strip()]
    say("     git log -S dispatch_tool -- t2_resolve.py  ->  %d 커밋" % len(lg))
    return a2


# =========================================================================
# (2) 그 결함이 실제 실패에 닿는가
# =========================================================================
RX_SCOPE_LINE = re.compile(
    r"\[T2_RESOLVE\] operator-scope: 지목 대신 범위 표면화 \(([^,]+), ([^)]+)\)")
RX_DENY_LINE = re.compile(r"\[T2_RESOLVE\] deny tool=(\S+) arg=(\S+) reason=operator-scope")


def gate2(sim):
    say("")
    say("=" * 92)
    say("[GATE 2] 그 결함이 실제 실패에 닿는가 - 회수 log/fb/trace/results 4중 교차")
    log = load_log()
    fb = load_fb()
    tr = load_trace()

    hit_scope = RX_SCOPE_LINE.search(log)
    hit_deny = RX_DENY_LINE.search(log)
    say("   (a) %s.log.gz 축자" % RUN)
    for ln in log.split("\n"):
        if ("[T2_RESOLVE] operator-scope: 지목" in ln or "reason=operator-scope" in ln
                or "[T2_UNAVAIL] promised" in ln):
            say("      %s" % ln.strip()[:195])
    chk(bool(hit_scope) and bool(hit_deny), "라이브 발화 실재")
    chosen = hit_scope.group(1).strip() if hit_scope else None
    want = hit_scope.group(2).strip() if hit_scope else None
    call_tool = hit_deny.group(1) if hit_deny else None
    say("   => chosen=%r  want=%r  **발화 자리(호출 도구)=%r**" % (chosen, want, call_tool))
    chk(call_tool == "unlock_discoverable_agent_tool",
        "발화 자리가 **아무 것도 수행하지 않는 unlock** 이다")

    live_fb = [r for r in fb if r.get("kind") == "tool-deny"
               and "[OPERATOR-SCOPE]" in (r.get("text") or "")]
    say("   (b) fb_%s.jsonl.gz - 모델에 실제 배달된 문면 %d행" % (RUN, len(live_fb)))
    live_text = live_fb[0]["text"] if live_fb else None
    if live_text:
        say("      turn=%s len=%s" % (live_fb[0].get("turn"), live_fb[0].get("len")))
        say("      %s" % live_text[:240])
    for b in [r for r in fb if r.get("kind") == "tool-deny"
              and "[BLOCKED]" in (r.get("text") or "")]:
        say("      turn=%s %s" % (b.get("turn"), (b.get("text") or "")[:180]))
    for d in [r for r in fb if r.get("kind") == "reminder-assistant"
              and r.get("turn") == 72]:
        tail = " ".join((d.get("text") or "").split())[-160:]
        say("      turn=72 폐기된 초안 산문 꼬리: ...%s" % tail)
    chk(bool(live_text), "그 문면이 모델에게 배달됐다")

    say("   (c) trace_%s.jsonl.gz - 초안/재생성 계수" % RUN)
    draft = regen = None
    for d in tr:
        if d.get("mark") != "T2_GEN_TRACE" or d.get("turn") != 71:
            continue
        ln = d.get("line") or ""
        if "call=agent_response " in ln:
            draft = ln
        if "call=agent_response_unified_regen" in ln:
            regen = ln
    say("      초안   %s" % (draft or "(없음)"))
    say("      재생성 %s" % (regen or "(없음)"))
    n_draft = int(re.search(r"tool_calls=(\d+)", draft).group(1)) if draft else -1
    n_regen = int(re.search(r"tool_calls=(\d+)", regen).group(1)) if regen else -1
    chk(n_draft == 2 and n_regen == 1, "초안 2호출 -> 반려 -> 재생성 1호출 (호출 하나가 사라졌다)")

    say("   (d) %s.results.json.gz - 채점 축자([[69]] 채점단위 먼저)" % RUN)
    ri = sim["reward_info"]
    say("      reward=%s reward_basis=%s db_match=%s"
        % (ri["reward"], ri.get("reward_basis"), (ri.get("db_check") or {}).get("db_match")))
    for a in ri["action_checks"]:
        inner = (a["action"]["arguments"] or {}).get("agent_tool_name")
        if inner and "reset_debit_card_pin" in str(inner):
            say("      %s %-32s %-28s action_match=%s tool_type=%s"
                % (a["action"]["action_id"], a["action"]["name"], inner,
                   a["action_match"], a.get("tool_type")))
    blob = json.dumps(sim["messages"], ensure_ascii=False)
    n_final = len(re.findall(r'"agent_tool_name":\s*"reset_debit_card_pin_6284"', blob))
    say("      최종 궤적에서 reset_debit_card_pin_6284 를 인자로 쓴 호출 = **%d회**" % n_final)
    chk(n_final == 0, "반려 뒤 **재시도 0회** - gold 호출이 궤적에서 사라졌다")
    return chosen, want, call_tool, live_text


# =========================================================================
# 격리 재료 조립 - 회수분에서만 (저작 0)
# =========================================================================
class TC(object):
    def __init__(self, d):
        self.id = d.get("id")
        self.name = d.get("name")
        self.arguments = d.get("arguments") or {}
        self.requestor = d.get("requestor")


class M(object):
    def __init__(self, d):
        self.role = d.get("role")
        self.content = d.get("content")
        self.error = bool(d.get("error"))
        self.id = d.get("id")
        self.requestor = d.get("requestor")
        self.tool_calls = [TC(t) for t in (d.get("tool_calls") or [])] or None


class Tool(object):
    def __init__(self, name, description):
        self.name = name
        self.description = description


class AgentShim(object):
    """`_tool_scope`/`registry_names` 가 읽는 최소 표면. 설명 문자열은 **회수값**이다."""

    def __init__(self, tools):
        self.tools = tools


RX_SCOPE_PAIR = re.compile(
    r"'([a-z_]+_[0-9]{4})' = (.+?)(?=; '[a-z_]+_[0-9]{4}' = |\. Check which one)")


def recovered_scopes(live_text):
    """라이브 `[OPERATOR-SCOPE]` 문면에서 후보별 선언 범위를 **되읽는다**(저작 0)."""
    return dict(RX_SCOPE_PAIR.findall(live_text or ""))


# =========================================================================
# (3) 수리 전/후 판정이 갈리는가
# =========================================================================
HUNK_SIG_OLD = ("def resolve_operator(opspec, args_dict, msgs, agent=None, la=None, "
                "UserMessage=None,\n                     declared_required=None, a2=None):")
HUNK_SIG_NEW = ("def resolve_operator(opspec, args_dict, msgs, agent=None, la=None, "
                "UserMessage=None,\n                     declared_required=None, a2=None, "
                "call_tool=None):")

HUNK_FWD_OLD = ("        return resolve_operator(opspec, args_dict, msgs, agent, la, UserMessage,\n"
                "                                declared_required=_req, a2=a2)")
HUNK_FWD_NEW = ("        return resolve_operator(opspec, args_dict, msgs, agent, la, UserMessage,\n"
                "                                declared_required=_req, a2=a2, call_tool=tool)")

HUNK_GUARD_OLD = '            if os.environ.get("T2_SCOPE_ALL") != "1":\n'
HUNK_GUARD_NEW = (
    '            _disp = ((a2 or {}).get("eplan") or {}).get("dispatch_tool")\n'
    '            if (os.environ.get(\"T2_SCOPE_AT_DISPATCH_ONLY\", \"1\") == \"1\"\n'
    '                    and _disp and call_tool is not None\n'
    '                    and str(call_tool) != str(_disp)):\n'
    '                print("[T2_RESOLVE] operator-scope 침묵: call=%s 는 실행하지 않는다 - "\n'
    '                      "되돌릴 수 없는 자리는 dispatch=%s 뿐" % (call_tool, _disp),\n'
    '                      file=sys.stderr, flush=True)\n'
    '                return {"status": "ok"}\n'
    '            if os.environ.get("T2_SCOPE_ALL") != "1":\n')
def _engine_src():
    return io.open(os.path.join(ENG, "t2_resolve.py"), encoding="utf-8").read()


# ★배선 감지(2026-09-05): 수리가 엔진에 실린 뒤에도 **같은 갈림**을 재려면 팔을 뒤집어야 한다.
#   배선 전 = PRE 현재소스 / POST 정패치.   배선 후 = PRE 역패치 / POST 현재소스(=라이브 그 자체).
NL = chr(10)
MARK_GUARD = HUNK_GUARD_NEW.split(NL)[0] + NL
MARK_ANCHOR = HUNK_GUARD_OLD
WIRED = (MARK_GUARD in _engine_src()) and (HUNK_SIG_NEW in _engine_src())


def _exec_mod(src, name):
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(ENG, "t2_resolve.py")
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _forward_patch(src):
    for old in (HUNK_SIG_OLD, HUNK_FWD_OLD, HUNK_GUARD_OLD):
        if src.count(old) != 1:
            raise SystemExit("정패치 앵커가 유일하지 않다(%d): %r" % (src.count(old), old[:70]))
    src = src.replace(HUNK_SIG_OLD, HUNK_SIG_NEW)
    src = src.replace(HUNK_FWD_OLD, HUNK_FWD_NEW)
    return src.replace(HUNK_GUARD_OLD, HUNK_GUARD_NEW)


def _reverse_patch(src):
    """배선된 소스에서 수리를 **기계적으로 도려낸다**(주석 블록 포함 · 문면 의존 0).
    가드 시작 = MARK_GUARD 줄, 끝 = 그 뒤 첫 MARK_ANCHOR 줄. 그 사이를 통째로 삭제한 뒤
    서명/전달 hunk 를 되돌린다. 결과가 배선 전 소스와 **바이트 동일**임을 호출부가 검산한다."""
    for new in (HUNK_SIG_NEW, HUNK_FWD_NEW):
        if src.count(new) != 1:
            raise SystemExit("역패치 앵커가 유일하지 않다(%d): %r" % (src.count(new), new[:70]))
    i = src.index(MARK_GUARD)
    head = src[:i].rstrip(NL).split(NL)
    # 가드 바로 위에 붙은 우리 주석 블록도 함께 도려낸다(수리와 한 몸이다).
    while head and head[-1].lstrip().startswith("#") and "수리(092" not in head[-1]:
        head.pop()
    if head and "수리(092" in head[-1]:
        head.pop()
    j = src.index(MARK_ANCHOR, i)
    src = NL.join(head) + NL + src[j:]
    src = src.replace(HUNK_SIG_NEW, HUNK_SIG_OLD)
    src = src.replace(HUNK_FWD_NEW, HUNK_FWD_OLD)
    if src.count(HUNK_GUARD_OLD) != 1 or src.count(HUNK_SIG_OLD) != 1:
        raise SystemExit("역패치 후 원본 앵커 복원 실패")
    return src


def build_post_module():
    """수리가 **실린** 술어. 배선 후에는 라이브 소스 그 자체다([[67]] 사본 0)."""
    if WIRED:
        return _exec_mod(_engine_src(), "t2_resolve_post")
    return _exec_mod(_forward_patch(_engine_src()), "t2_resolve_post")


def build_pre_module():
    """수리가 **없는** 술어. 배선 후에는 라이브 소스를 역패치해 복원한다."""
    if WIRED:
        return _exec_mod(_reverse_patch(_engine_src()), "t2_resolve_pre")
    import t2_resolve as _m
    return _m


def check_roundtrip():
    """★역패치 충실성: 배선 소스를 역패치 -> 다시 정패치하면 **주석만 다른 원본**인가.
    같으면 PRE 팔이 「배선 전 그 소스」임이 기계적으로 보증된다(추정 0)."""
    if not WIRED:
        return None
    live = _engine_src()
    back = _forward_patch(_reverse_patch(live))
    nocomment = lambda t: NL.join(l for l in t.split(NL) if not l.lstrip().startswith("#"))
    return nocomment(back) == nocomment(live)



def gate3(a2, sim, chosen, want, call_tool, live_text):
    say("")
    say("=" * 92)
    say("[GATE 3] ★수리 전/후 판정이 갈리는가 - 같은 재료 · 엔진 정본 진입점 resolve_write()")

    msgs = [M(m) for m in sim["messages"][:72]]      # 초안 턴 직전까지(= 라이브 컨텍스트)
    scopes = recovered_scopes(live_text)
    say("   회수한 선언 범위(라이브 문면 되읽기) %d건: %s" % (len(scopes), sorted(scopes)))
    args_dict = {"agent_tool_name": chosen}
    say("   재료: msgs=%d(회수 궤적 0..71) · 호출 = %s(agent_tool_name=%r)"
        % (len(msgs), call_tool, chosen))

    PRE = build_pre_module()
    POST = build_post_module()
    _rt = check_roundtrip()
    if _rt is not None:
        chk(_rt, "역패치 충실성: 역패치->정패치가 라이브 소스로 **되돌아온다** (PRE 팔 = 배선 전 그 술어)")
    say("   팔 구성: %s" % ("배선 후 — PRE=역패치(수리 도려냄) · POST=**라이브 소스 그 자체**"
                          if WIRED else "배선 전 — PRE=현재소스 · POST=정패치"))

    calls = Counter()

    def _fake_formalize(*a, **k):
        calls["n"] += 1
        return want

    PRE.formalize_intent_tool = _fake_formalize
    POST.formalize_intent_tool = _fake_formalize
    say("   선언 오버라이드 1칸: formalize_intent_tool() -> %r (trace 회수값 · LLM 재질의 0)"
        % want)

    def run(mod, tool, a2_in, tag):
        ag = AgentShim([Tool(n, d) for n, d in scopes.items()])
        r = mod.resolve_write(tool, dict(args_dict), msgs, a2_in, ag, object(), object())
        say("   [%s] tool=%-30s -> status=%-5s reason=%s"
            % (tag, tool, r.get("status"), r.get("reason")))
        return r

    say("")
    say("   -- 팔 A/B (같은 재료 · 같은 호출) --")
    r_pre = run(PRE, call_tool, a2, "PRE  현재코드")
    r_post = run(POST, call_tool, a2, "POST 수리후 ")

    say("")
    say("   -- 충실성 검사 (격리가 라이브를 재현하는가) --")
    live_body = (live_text or "")
    if live_body.startswith("Error: "):
        live_body = live_body[len("Error: "):]
    same = (r_pre.get("feedback") or "").strip() == live_body.strip()
    say("      PRE  문면 len=%d" % len(r_pre.get("feedback") or ""))
    say("      라이브 문면 len=%d ('Error: ' 접두 제외)" % len(live_body))
    chk(same, "PRE 팔 문면이 라이브 사이드카와 **바이트 동일** - 격리가 라이브다")
    if not same:
        say("      PRE  : %r" % (r_pre.get("feedback") or "")[:220])
        say("      LIVE : %r" % live_body[:220])

    say("")
    say("   -- (3) 판정 --")
    flipped = (r_pre.get("status") == "deny" and r_post.get("status") == "ok")
    chk(r_pre.get("status") == "deny", "PRE = deny(operator-scope) - 라이브와 같다")
    chk(r_post.get("status") == "ok", "POST = ok - gold unlock 이 살아남는다")
    chk(flipped, "★판정이 갈린다 (deny -> ok)")

    say("")
    say("   -- [[57]] 부정통제 --")
    r_nc1 = run(PRE, call_tool, a2, "NC-1 되돌림")
    chk(r_nc1.get("status") == "deny", "NC-1 PRE 로 되돌리면 deny 가 **돌아온다**")

    disp = ((a2 or {}).get("eplan") or {}).get("dispatch_tool")
    r_nc2 = run(POST, disp, a2, "NC-2 dispatch")
    chk(r_nc2.get("status") == "deny",
        "NC-2 실행 자리(dispatch)에서는 POST 도 **여전히 deny** - 끄기가 아니라 자리 옮기기([[60]])")

    a2_nodisp = dict(a2)
    a2_nodisp["eplan"] = {k: v for k, v in (a2.get("eplan") or {}).items()
                          if k != "dispatch_tool"}
    r_nc3 = run(POST, call_tool, a2_nodisp, "NC-3 선언제거")
    chk(r_nc3.get("status") == "deny",
        "NC-3 A2 에서 dispatch_tool 선언을 빼면 POST = PRE - 갈림의 출처는 **선언**이다([[05]])")

    say("")
    say("   서브콜 재질의 횟수 = %d (LLM 0회 · 전부 회수값 반환)" % calls["n"])
    return flipped


# =========================================================================
# (4) [[70]] 파는 것 - 회수분 전수
# =========================================================================
CAMPAIGN_T = time.mktime((2026, 9, 3, 0, 0, 0, 0, 0, -1))


def gate4():
    say("")
    say("=" * 92)
    say("[GATE 4] [[70]] 파는 것 - 회수분 **전수** 계수")
    by_tool = Counter()
    by_tool_camp = Counter()
    nlogs = 0
    outcome = Counter()
    outcome_camp = Counter()
    lost_tasks_camp = Counter()
    for p in glob.glob(os.path.join(SIMS, "*.log.gz")):
        try:
            txt = _gz(p).read()
        except Exception:
            continue
        recent = os.path.getmtime(p) >= CAMPAIGN_T
        rows = []
        cur_sim = None
        for ln in txt.split("\n"):
            ms = re.search(r"\[sim=([^\]]+)\]", ln)
            if ms:
                cur_sim = ms.group(1)
            m1 = RX_SCOPE_LINE.search(ln)
            if m1:
                rows.append([cur_sim, m1.group(1).strip(), m1.group(2).strip(), None])
            m2 = RX_DENY_LINE.search(ln)
            if m2 and rows and rows[-1][3] is None:
                rows[-1][3] = m2.group(1)
        hits = [r for r in rows if r[3]]
        if not hits:
            continue
        nlogs += 1
        for _sim, _ch, _wa, _tool in hits:
            by_tool[_tool] += 1
            if recent:
                by_tool_camp[_tool] += 1
        rp = p[:-len(".log.gz")] + ".results.json.gz"
        if not os.path.exists(rp):
            continue
        try:
            dd = json.load(_gz(rp))
        except Exception:
            continue
        idx = {}
        for s in dd.get("simulations") or []:
            idx.setdefault(str(s.get("task_id")), []).append(s)
        for _sim, _ch, _wa, _tool in hits:
            tid = (_sim or "").split("#")[0]
            cand = idx.get(tid) or []
            if not cand:
                continue
            blob = json.dumps(cand[0].get("messages") or [], ensure_ascii=False)
            ran = ('"agent_tool_name": "%s"' % _ch) in blob
            key = "태움(끝내 실행)" if ran else "잃음(끝내 미실행)"
            outcome[(_tool, key)] += 1
            if recent:
                outcome_camp[(_tool, key)] += 1
                if not ran and _tool == "unlock_discoverable_agent_tool":
                    lost_tasks_camp[tid] += 1

    tot = sum(by_tool.values())
    say("   `[T2_RESOLVE] deny ... reason=operator-scope` 발화 - log %d개 · 총 %d회"
        % (nlogs, tot))
    for k, v in by_tool.most_common():
        say("      %-32s %4d회   (캠페인 2026-09-03+ %d회)" % (k, v, by_tool_camp[k]))
    unl = by_tool.get("unlock_discoverable_agent_tool", 0)
    unl_c = by_tool_camp.get("unlock_discoverable_agent_tool", 0)
    tot_c = sum(by_tool_camp.values())
    say("   => **수리가 침묵시키는 몫** = unlock 자리 %d/%d = %.1f%%  (캠페인 %d/%d = %.1f%%)"
        % (unl, tot, 100.0 * unl / max(tot, 1), unl_c, tot_c, 100.0 * unl_c / max(tot_c, 1)))
    say("   => **수리가 그대로 두는 몫** = dispatch 자리 %d/%d (캠페인 %d/%d)"
        % (by_tool.get("call_discoverable_agent_tool", 0), tot,
           by_tool_camp.get("call_discoverable_agent_tool", 0), tot_c))
    say("")
    say("   발화 뒤 그 operand 가 끝내 실행됐나 (results 조인 성공분만):")
    for k in sorted(outcome):
        say("      %-32s %-18s %d" % (k[0], k[1], outcome[k]))
    say("   => 「태움」 = 반려가 선택을 바꾸지 못하고 턴만 태운 몫")
    say("      (t2_resolve.py:254 축자 *\"61 중 49 는 끝내 실행됐다\"* 와 같은 자리)")
    say("   => 「잃음」 = 반려 뒤 그 도구가 궤적에서 사라진 몫 - 092 가 이 칸이다")
    say("   !! 조인 근사: 같은 태스크에 sim 이 여럿인 런은 **첫 sim** 으로만 대조한다(상한 아님).")
    say("")
    say("   캠페인(2026-09-03+) 만:")
    for k in sorted(outcome_camp):
        say("      %-32s %-18s %d" % (k[0], k[1], outcome_camp[k]))
    say("   캠페인 unlock-자리 「잃음」 태스크 분포 = %s"
        % dict(lost_tasks_camp.most_common(15)))
    say("      (036 은 엔진 자신의 주석이 이미 박제한 자리다 - t2_resolve.py:250-255 축자")
    say("       *\"036 실물: gold 가 요구하는 order_replacement_credit_card_7291 을 10회 반려했고,")
    say("       에이전트는 그것을 technical error 로 읽고 ... 그 gold 행이 통째로 MISSING\"*)")
    say("   !! 파는 것의 정직한 진술: unlock 자리의 **사전 경고 한 턴**을 판다.")
    say("      경고 자체는 사라지지 않고 dispatch(실행) 자리로 **한 걸음 미뤄진다**(NC-2 실측).")


def main():
    sim = load_sim()
    a2 = gate1()
    chosen, want, call_tool, live_text = gate2(sim)
    flipped = gate3(a2, sim, chosen, want, call_tool, live_text)
    gate4()
    say("")
    say("=" * 92)
    if FAIL:
        say("판정: **PROBE-FAIL / 판정불가** - 못 채운 칸 %d개" % len(FAIL))
        for f in FAIL:
            say("   - %s" % f)
    elif flipped:
        say("판정: **PROBE-PASS** - (1)(2)(3)(4) 4칸 + 부정통제 3종 전부 충족")
    else:
        say("판정: **PROBE-FAIL**")
    say("=" * 92)
    say("")
    say("!! 이 프로브가 **증명하지 않는 것**(과대주장 금지 · [[69]] 채점단위):")
    say("   092 의 reward_basis 는 ['DB'] 이고 채점은 DB 해시다. 이 수리가 직접 살리는 gold 는")
    say("   092_17 unlock_discoverable_agent_tool(reset_debit_card_pin_6284) = tool_type generic")
    say("   (비변이)이다. DB 를 가르는 MISSING 3건(092_13 close green · 092_14 order green ·")
    say("   092_18 reset pin evergreen)은 **거부 술어가 만들어 낼 수 없다** - P26 감사 B 가")
    say("   078/080 에 낸 반박과 같은 형태다.")
    say("   => 이 수리는 **092 의 reward 회복을 보장하지 않는다**. 보장하는 것은 하나뿐:")
    say("      gold 호출을 시도한 턴을 우리가 죽여 재시도를 0으로 만드는 경로가 닫힌다.")


if __name__ == "__main__":
    main()
