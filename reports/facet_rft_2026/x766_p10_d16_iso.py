# -*- coding: utf-8 -*-
"""P10 — D16([ACTION] 선언 불완결·referent 무구속) 격리. 팔 = **선언 한 칸**.

## 왜 이 프로브인가

설계서가 CONFIRMED 로 확정한 우리-층 결함 D16 축자(x737:57):

    «[ACTION] 선언 불완결(referent 무구속): 인자명만 주고 «누구의 user_id 인가»를 안 묶어
     유일한 §7-무장 fail 을 죽였다. CONFIRMED·선재(07-22~08-10 도입).
     수리 = env 축자 스니펫 동봉([[71]]/[[23]]/[[58]] 안전) · P10 격리 후 배선»

라이브 실물(회수·n=1 sim, task_010#s373753). 우리 층이 결정 창에서 실제로 발화한 문장의 꼬리:

    «... tell the customer in your reply to run submit_referral themselves with their
      details, then confirm the result. ... Arguments of submit_referral: user_id, account_type.»

그 직후 모델이 referent 를 스스로 메꿨다(축자 · msgs[27] raw_data/choices/0/message/reasoning_content):

    «The tool signature submit_referral(user_id, account_type). Which user_id? Could be
      referred person's user_id? ... Need infer.»

그리고 «친구의 user_id» 로 굳혔다(msgs[27]) → 손님이 «없다»(msgs[28]) → ###STOP###(msgs[32]).

## ★exit — 재정의된 것을 쓴다 (2026-09-05)

task_010 의 gold 두 번째 쓰기는 **유저-쓰기**다:

    {"action_id": "010_1", "requestor": "user", "name": "submit_referral",
     "arguments": {"user_id": "76ad9cc60e", "account_type": "Platinum Rewards Card"}}

dbdiff census 는 010 을 MISSING(ONLY-PRED 0 / ONLY-GOLD 2 / DIFF 0)으로 찍었다. 실패 단위는
«에이전트가 틀린 인자를 넣었다»가 아니라 «그 유저-쓰기가 일어나지 않았다»이다. 그러므로

    exit = «스니펫을 동봉하면 그 결정 창에서 유저가 본인 id 로 호출하게 되는가»

이고 «에이전트의 인자 정정»이 아니다.

### 이 프로브가 실제로 재는 것 / 대체한 것 (요건 3 · 정직 고지)

진짜 exit 은 **user-sim 의 쓰기**를 요구한다. 그런데 이 머신에 tau2 가 없고
(`import tau2` → ModuleNotFoundError), 회수 번들에는 **user-sim 의 system prompt·도구 스키마가
저장돼 있지 않다**(`info.user_info` 는 global guidelines 뿐 · `info.agent_info` 는 llm 이름뿐).
user-sim 을 세우려면 프롬프트를 저작해야 하는데 그것은 [[78]] 위반이라 **하지 않는다**.
그래서 두 단으로 쪼갠다.

  STAGE-1 (무료·오프라인·항상 실행) = **선언 문면 자체**를 잰다.
      «B 팔의 선언이 referent 를 실제로 묶는가 · 그 차이가 한 칸뿐인가 · 그 칸이 엔진의
        조용한 삼킴(:10562 `except Exception: pass`)에 죽지 않는가 · 사이드카 상한에 잘리지 않는가»
      ⇒ **유저 거동은 재지 못한다**. 이것은 exit 의 *전건*(우리 층이 무엇을 말했나)만 닫는다.

  STAGE-2 (`--live HOST:PORT` 일 때만) = **에이전트 hop 하나**를 잰다.
      회수된 결정 창(msgs[:27] + 초안 + 우리 피드백)을 그대로 되꽂고 재생성시켜,
      답문의 `user_id` 슬롯에 **무엇이 들어가는가**를 축자로 뽑는다.
      §7 은 «에이전트가 재시도 가능하다고 말하면 제출한다» 이므로 이 hop 이 유저-쓰기의
      **필요조건**이다(회수 실패는 정확히 여기서 죽었다 — 친구 id 요구 → user_stop).
      ⇒ 이것도 **충분조건이 아니다**. 유저-쓰기 자체는 라이브 런에서만 확정된다.

  ⇒ 요컨대 «유저가 본인 id 로 호출한다»는 이 프로브로 **확정되지 않는다**. STAGE-1 은
     전건의 문면을, STAGE-2 는 그 다음 hop 의 지목을 잰다. 나머지는 «모른다»([[77]]).

## 반증 조건 — refutation conditions (주장과 동시에 적는다 · [[77]])

  R1 STAGE-1 이 이렇게 관측되면 이 격리는 거짓이다: B 팔 산출에 env 축자가 없거나(=삼킴),
     A↔B 차이가 꼬리 한 곳이 아니거나, A 팔 산출이 라이브 축자와 바이트 불일치면 →
     격리가 **라이브와 다른 것을 재고 있다**. 그러면 «수리 = 스니펫 동봉» 은 지지되지 않는다.
  R2 STAGE-2 가 이렇게 관측되면 D16 수리안이 거짓이다: B 팔에서도 `user_id` 슬롯 filler 가
     본인 id 가 아닌 것(플레이스홀더·제3자)으로 나오면. A 팔과 B 팔의 filler 분포가 같아도
     같다(효과 0 · [[85]] 바닥 안).
  R3 [[70]] 반대편: B 의 기전이 referent 를 좁히면서 **정당한 제3자 지목까지** 죽이면
     (예: `submit_referral.account_type` = "you are referring someone" 이 사라지면) 순효과는
     음일 수 있다. §SIGN 이 그것을 같은 프로브에서 센다.
  R4 재료 0건이면 아무 주장도 하지 않는다 — «주장 금지» 를 찍고 exit 2.

## 선행 확인 (grep 경로 · [[74]])

  grep -n "Arguments of" scripts/distill/tau2/t2_gate_patch.py      → 히트 1 (:10560)
  grep -n "\[ACTION\]"   scripts/distill/tau2/t2_gate_patch.py      → 히트 5
  grep -n "axis_notes"   scripts/distill/tau2/a2/*.json a2/base/*.json
        → `user_action_arglist` 는 **선언 0건**(엔진 기본값이 발화한다)
  grep -rn "submit_referral" reports/facet_rft_2026/sim_results/*.results.json.gz (role=tool 필터)
        → env 발화 67건이 전부 오류문 2종 · KB 문서 0편 (x767 갈래)
  ls reports/facet_rft_2026/x76*.py · x737_next_run_plan_2026_09_04.md §1f-9 정독

## 재료 좌표 (전부 선언 또는 회수 · gold 는 채점에만)

  M1 A2 선언      : `gate_interpreter.load_domain_a2("banking_knowledge")` — **엔진 함수 호출**
  M2 엔진 조립부  : `t2_gate_patch.py` 의 `_ufb = str((a2 or {}).get("user_action_feedback")`
                    앵커부터 `except Exception:/pass` 까지 — 소스를 **잘라 그대로 exec** 한다
                    (재구현 0 · [[67]]). 외곽 = `apply_unified_regen`(:8425) → `unified`(:8685).
  M3 env 표면     : `a2/env_surface.json` → `banking_knowledge.tools.<target>`
                    `{"args": [...], "desc": "...Args:\n    user_id: Your user ID (the referrer)..."}`
  M4 라이브 축자  : `sim_results/fb_bank_010ctl_20260904_0007.jsonl.gz`
                    `{"kind":"reminder-user","turn":27,"simtag":"task_010#s373753","channel":"unified_regen"}`
                    (같은 turn 의 `"kind":"reminder-assistant"` = 재생성 대상 초안)
  M5 결정 창      : `sim_results/bank_010ctl_20260904_0007.results.json.gz`
                    `simulations[seed=373753].messages[:27]`
  M6 채점 키(gold): 같은 번들 `tasks[0].evaluation_criteria.actions[requestor=="user"]`
                    — **채점에만** 쓴다. 선언 저작에는 한 글자도 쓰지 않는다([[23]]).
  M7 표적 tool    : 타이핑하지 않는다. M2 가 만든 기본 템플릿의 `{tool}` 자리에 sentinel 을
                    넣어 얻은 앵커로 M4 축자에서 **역추출**한다.

## 철칙 자기감사

  [[78]] 프롬프트 저작 0. 팔은 선언 오버라이드 한 칸(`axis_notes.user_action_arglist`).
         STAGE-2 의 문맥은 전부 회수 축자를 되꽂은 것이고, 새로 쓴 지시문은 없다.
  [[71]] 결정 하나(그 창의 [ACTION] 문면) · 재료는 선언에서 읽음 · 엔진은 읽어 전달만.
  [[23]] 스니펫 출처 = env 도구 독스트링(M3). gold 미참조.
  [[62]] 결정론기는 «엔진 소스를 잘라 exec» 한 줄뿐. argmax·최댓값·«정답은 X» 없음.
  [[05]] ⚠B 팔의 A2 셀에는 **표적 도구의 env 축자가 리터럴로** 들어간다 — 측정용 팔이지
         출시형이 아니다. 출시형은 §WIRE 가 적는다(엔진이 이미 읽는 같은 dict 의 값 쪽).

실행:
    PYTHONIOENCODING=utf-8 py -3 reports/facet_rft_2026/x766_p10_d16_iso.py
    PYTHONIOENCODING=utf-8 py -3 reports/facet_rft_2026/x766_p10_d16_iso.py --live 10.0.0.151:8141 --n 4
"""
import argparse
import copy
import gzip
import io
import json
import os
import re
import sys
import textwrap
import urllib.request

REPO = r"C:\workspace\ba-frft"
TAU2 = os.path.join(REPO, "scripts", "distill", "tau2")
SR = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results")
ENGINE_FILE = os.path.join(TAU2, "t2_gate_patch.py")
ENV_SURFACE = os.path.join(TAU2, "a2", "env_surface.json")

DOMAIN = "banking_knowledge"
TAG = "bank_010ctl_20260904_0007"          # 회수 좌표 (M4/M5/M6)
SIMTAG = "task_010#s373753"                # 사이드카 simtag → task_id·seed 를 여기서 판다
FB_CHANNEL = "unified_regen"

# 엔진 조립부 앵커 (M2) — 문면이 아니라 **소스 좌표**다.
ANCHOR = '_ufb = str((a2 or {}).get("user_action_feedback")'
SENT_T = "\u0001TOOL\u0001"
SENT_A = "\u0001ARGS\u0001"
SIDECAR_CAP_DEFAULT = 4000                 # go_stack.sh:216 T2_FB_SIDECAR_TEXT_MAX

sys.path.insert(0, TAU2)
os.chdir(TAU2)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from gate_interpreter import load_domain_a2                      # noqa: E402  ← 엔진 함수


# ────────────────────────────────────────────────────────────────────────────
# 0. 엔진 조립부를 **잘라서 그대로 실행**한다 (재구현 0)
# ────────────────────────────────────────────────────────────────────────────
def engine_action_block():
    """`t2_gate_patch.py` 에서 [ACTION] 조립 블록을 앵커로 슬라이스 → dedent.

    독립 빌더 함수가 없다(외곽이 `unified(self, message, state)`·:8685). 그래서 **부를 이름**
    대신 **엔진의 바이트**를 부른다: 이 블록은 손으로 옮겨 적지 않고 파일에서 잘라 exec 한다.
    """
    src = io.open(ENGINE_FILE, encoding="utf-8").read().splitlines()
    starts = [i for i, l in enumerate(src) if ANCHOR in l]
    if len(starts) != 1:
        return None, None, None
    i = starts[0]
    j = i
    while j < len(src):
        if src[j].strip() == "pass" and src[j - 1].strip().startswith("except Exception"):
            break
        j += 1
    else:
        return None, None, None
    blk = textwrap.dedent("\n".join(src[i:j + 1]))
    return blk, i + 1, j + 1


class _ShimTool(object):
    """엔진이 읽는 모양만 흉내낸다 — 값 생성 0. 엔진은 `properties` 의 **키만** 읽는다."""

    def __init__(self, name, argnames):
        self.name = name
        self.openai_schema = {
            "type": "function",
            "function": {"name": name,
                         "parameters": {"type": "object",
                                        "properties": dict((a, {}) for a in argnames)}},
        }


class _ShimEnv(object):
    def __init__(self, tools):
        self._tools = list(tools)

    def get_user_tools(self):
        return list(self._tools)

    def get_tools(self):
        return []


class _ShimOrch(object):
    def __init__(self, env):
        self.environment = env


class _ShimSelf(object):
    def __init__(self, orch):
        self._t2_orch = orch


def build_action_text(block, a2, utgt, argnames):
    """엔진 블록을 그대로 실행해 `_ufb` 를 받는다. 우리가 넣는 것은 재료뿐."""
    ns = {"a2": a2, "_utgt": utgt,
          "self": _ShimSelf(_ShimOrch(_ShimEnv([_ShimTool(utgt, argnames)])))}
    exec(block, ns)                                              # noqa: S102 — 엔진 소스 그대로
    return ns.get("_ufb")


# ────────────────────────────────────────────────────────────────────────────
# 1. 회수 재료
# ────────────────────────────────────────────────────────────────────────────
def read_gz_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def sidecar_rows(path, simtag):
    out = []
    if not os.path.exists(path):
        return out
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("simtag") == simtag:
                out.append(r)
    return out


def declared_arg_lines(desc, argnames):
    """env 도구 독스트링의 `Args:` 블록에서 **선언된 줄을 축자로** 뽑는다.

    Google-style 독스트링 형식 파싱이다(도메인 패턴매칭 아님). 반환은 원문 그대로.
    ⚠`env_surface.json` 의 `desc` 는 200자에서 잘린다(실측) — 잘린 도구는 flag 로 표시한다.
    """
    lines, seen = [], {}
    for raw in (desc or "").splitlines():
        s = raw.strip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s*:\s*(.+)$", s)
        if not m:
            continue
        nm = m.group(1)
        if nm not in argnames or nm in seen:
            continue
        seen[nm] = True
        lines.append((nm, s))
    missing = [a for a in argnames if a not in seen]
    return lines, missing


def brace_escape(s):
    """`.format` 이 새 자리표시자로 오해하지 않도록 중괄호를 이중화.

    ⚠이 방어가 왜 필요한가: 엔진은 `_tpl_a.format(tool=..., args=...)` 로 kwargs 두 개만 넘긴다.
    선언에 새 자리표시자가 있으면 KeyError 가 나고 :10562 `except Exception: pass` 가 그것을
    **조용히 삼켜** `Arguments of ...` 문장이 통째로 사라진다(§NEG 에서 재현한다).
    """
    return s.replace("{", "{{").replace("}", "}}")


def flat_paths(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat_paths(v, prefix + "/" + str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flat_paths(v, prefix + "/[%d]" % i))
    else:
        out[prefix] = repr(obj)
    return out


def a2_diff_paths(a, b):
    fa, fb = flat_paths(a), flat_paths(b)
    keys = set(fa) | set(fb)
    return sorted(k for k in keys if fa.get(k) != fb.get(k))


# ────────────────────────────────────────────────────────────────────────────
# 2. STAGE-2 (라이브 · 옵션) — [[30]] 함정 6: 포트가 아니라 **모델 id 대조**
# ────────────────────────────────────────────────────────────────────────────
def served_model(base_url, timeout=30):
    with urllib.request.urlopen(base_url + "/models", timeout=timeout) as r:
        d = json.load(r)
    return [x.get("id") for x in (d.get("data") or [])]


def chat(base_url, model, messages, temperature, max_tokens, timeout=600):
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": temperature, "max_tokens": max_tokens}).encode("utf-8")
    req = urllib.request.Request(base_url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return (d["choices"][0]["message"].get("content") or "")


def to_openai(msgs):
    """회수 메시지를 OpenAI 채팅 모양으로 **되꽂는다**(새 문장 저작 0)."""
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("id") or "t",
                        "content": m.get("content") or ""})
        elif role == "assistant":
            e = {"role": "assistant", "content": m.get("content") or ""}
            tcs = m.get("tool_calls") or []
            if tcs:
                e["tool_calls"] = [{"id": t.get("id") or "t", "type": "function",
                                    "function": {"name": t.get("name") or "",
                                                 "arguments": json.dumps(t.get("arguments") or {})}}
                                   for t in tcs]
                e["content"] = e["content"] or None
            out.append(e)
        else:
            out.append({"role": "user", "content": m.get("content") or ""})
    return out


def slot_fillers(text, arg):
    """답문에서 `<arg>` 슬롯에 실제로 들어간 **축자 filler** 를 뽑는다.

    분류하지 않는다([[62]]) — 뽑아서 그대로 보여주고, 채점은 gold 키와의 **문자열 일치**로만
    한다(§LIVE). 열린 술어(«친구인가 본인인가»)는 사람이 축자를 보고 판정한다.
    """
    hits = []
    for m in re.finditer(re.escape(arg) + r"\s*[=:]\s*([^\s,\)\]\}\n]{1,60})", text):
        hits.append((m.group(1).strip("\"'`*"), text[max(0, m.start() - 100):m.end() + 60]))
    return hits


# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default=None, metavar="HOST:PORT",
                    help="STAGE-2 를 켠다. [[83]] 엔진 식별은 포트가 아니라 HOST:PORT.")
    ap.add_argument("--expect-model", default=None,
                    help="미지정이면 회수 번들의 info.agent_info.llm 을 쓴다([[30]] 함정 6).")
    ap.add_argument("--allow-model-mismatch", action="store_true")
    ap.add_argument("--n", type=int, default=4, help="팔당 표본 수")
    ap.add_argument("--temp", type=float, default=None, help="미지정이면 회수 llm_args 값")
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--cap", type=int, default=SIDECAR_CAP_DEFAULT)
    args = ap.parse_args()

    print("=" * 78)
    print("[x766-P10] D16 격리 — [ACTION] 선언의 referent 무구속. 팔 = 선언 한 칸.")
    print("=" * 78)

    # ── R4 재료 게이트 ───────────────────────────────────────────────────
    task_id = SIMTAG.split("#")[0]
    seed = int(SIMTAG.split("#s")[1])
    bundle_p = os.path.join(SR, TAG + ".results.json.gz")
    fb_p = os.path.join(SR, "fb_" + TAG + ".jsonl.gz")

    absent = [p for p in (ENGINE_FILE, ENV_SURFACE, bundle_p, fb_p) if not os.path.exists(p)]
    block, ln_a, ln_b = engine_action_block()
    a2 = load_domain_a2(DOMAIN)
    env_all = json.load(io.open(ENV_SURFACE, encoding="utf-8")) if os.path.exists(ENV_SURFACE) else {}
    env_dom = (env_all.get(DOMAIN) or {})
    rows = sidecar_rows(fb_p, SIMTAG) if os.path.exists(fb_p) else []

    if absent or block is None or not a2 or not env_dom or not rows:
        print("\n⛔ 주장 금지 — 재료 결손([[77]]).")
        for p in absent:
            print("   부재:", p)
        if block is None:
            print("   엔진 앵커 미해결:", ANCHOR)
        if not a2:
            print("   A2 미로드:", DOMAIN)
        if not env_dom:
            print("   env 표면 미로드:", DOMAIN)
        if not rows:
            print("   사이드카 항목 0:", SIMTAG)
        print("   ⇒ 이 상태에서 D16 에 대해 어떤 부호도 발화하지 않는다.")
        return 2

    print("\n[ENGINE] 조립부 = %s:%d-%d (앵커 슬라이스·재구현 0)" % (
        os.path.basename(ENGINE_FILE), ln_a, ln_b))
    print("         블록 %d줄 / %d자 — exec 로 그대로 실행" % (len(block.splitlines()), len(block)))

    # ── M7: 표적을 **타이핑하지 않고** 역추출 ────────────────────────────
    tmpl_fb = build_action_text(block, {}, SENT_T, [])
    if not tmpl_fb or SENT_T not in tmpl_fb:
        print("\n⛔ 주장 금지 — 엔진 기본 템플릿을 sentinel 로 회수하지 못했다.")
        return 2
    seg = tmpl_fb.split(SENT_T)
    rx = re.compile(re.escape(seg[0]) + r"([A-Za-z_][A-Za-z0-9_]*)" + re.escape(seg[1][:60]))

    live_rows = [r for r in rows
                 if r.get("kind") == "reminder-user" and r.get("channel") == FB_CHANNEL
                 and rx.search(r.get("text") or "")]
    if not live_rows:
        print("\n⛔ 주장 금지 — 회수 사이드카에 [ACTION] 발화 0건(%s)." % SIMTAG)
        return 2
    live = live_rows[-1]                                   # 결정 창 = 마지막 발화 turn
    live_text = live.get("text") or ""
    target = rx.search(live_text).group(1)
    live_line = live_text.split("\n")[0]

    tool_decl = ((env_dom.get("tools") or {}).get(target) or {})
    argnames = list(tool_decl.get("args") or [])
    if not argnames:
        print("\n⛔ 주장 금지 — env 선언에 '%s' 의 args 가 비었다." % target)
        return 2

    print("\n[M] 재료")
    print("    표적(역추출)  : %s   ← 타이핑 0, 엔진 템플릿 앵커로 M4 에서 추출" % target)
    print("    A2 action_tools 포함: %s" % (target in (a2.get("action_tools") or [])))
    print("    env 선언      : side=%s mutates=%s args=%s desclen=%d%s" % (
        tool_decl.get("side"), tool_decl.get("mutates"), argnames,
        len(tool_decl.get("desc") or ""),
        "  ⚠덤프 200자 절단 의심" if len(tool_decl.get("desc") or "") >= 200 else ""))
    print("    라이브 축자   : turn=%s len=%s sha=%s (first-line %d자)" % (
        live.get("turn"), live.get("len"), live.get("sha"), len(live_line)))
    print("    선언 현황     : axis_notes.user_action_arglist = %r  (미선언 ⇒ 엔진 기본값 발화)"
          % ((a2.get("axis_notes") or {}).get("user_action_arglist")))

    # ── 팔 A: 현행 선언 그대로 ──────────────────────────────────────────
    A_line = build_action_text(block, a2, target, argnames)

    # ── 기본 arglist 템플릿을 sentinel 로 회수(손으로 쓰지 않는다) ───────
    base_fb = build_action_text(block, {}, SENT_T, [])
    base_fb_args = build_action_text(block, {}, SENT_T, [SENT_A])
    tail_tmpl = base_fb_args[len(base_fb):].replace(SENT_T, "{tool}").replace(SENT_A, "{args}")

    # ── 팔 B: **선언 오버라이드 정확히 한 칸** ──────────────────────────
    #   내용 = 엔진이 이미 읽고 있는 그 dict 의 **값 쪽**(env 도구 독스트링 Args 줄) 축자.
    #   저작한 낱말 0 — 이어붙이는 것은 구두점뿐(" ", "; ").
    decl_lines, decl_missing = declared_arg_lines(tool_decl.get("desc"), argnames)
    if not decl_lines:
        print("\n⛔ 주장 금지 — env 선언에서 '%s' 의 인자별 축자를 못 뽑았다(desc 절단?)." % target)
        return 2
    ARGDOC = "; ".join(s for _, s in decl_lines)
    B_CELL = tail_tmpl + " " + brace_escape(ARGDOC)

    a2_B = copy.deepcopy(a2)
    a2_B.setdefault("axis_notes", {})["user_action_arglist"] = B_CELL
    B_line = build_action_text(block, a2_B, target, argnames)

    diff = a2_diff_paths(a2, a2_B)

    print("\n[ARM] 팔 (차이는 코드에서 한 줄이다)")
    print("    A_off : a2                       (오버라이드 0)")
    print("    B_on  : a2 + axis_notes.user_action_arglist = <기본 템플릿> + ' ' + <env 축자>")
    print("    선언 diff 경로 %d개: %s" % (len(diff), diff))
    print("    env 축자(동봉분): %r" % ARGDOC)
    if decl_missing:
        print("    ⚠선언 줄이 안 잡힌 인자: %s (절단·미기재)" % decl_missing)

    # ── G1: 격리 ↔ 라이브 동일성 ───────────────────────────────────────
    g1 = (A_line == live_line)
    print("\n[G1] A 팔 산출 == 라이브 축자 첫 줄 ?  %s  (%d자 vs %d자)"
          % ("YES" if g1 else "NO", len(A_line or ""), len(live_line)))
    if not g1:
        print("    ⛔ 격리가 라이브와 다른 것을 재고 있다(R1). 아래 부호는 신뢰 금지.")
        print("    ISO : %r" % (A_line or "")[:300])
        print("    LIVE: %r" % live_line[:300])

    # ── G2: 차이가 꼬리 한 곳인가 ──────────────────────────────────────
    g2 = bool(B_line) and B_line.startswith(A_line) and B_line[len(A_line):] == " " + ARGDOC
    print("\n[G2] B - A = 꼬리 한 곳 ?  %s" % ("YES" if g2 else "NO"))
    print("    A tail: %r" % A_line[-90:])
    print("    B tail: %r" % (B_line or "")[-260:])

    # ── NEG: 조용한 삼킴 부정통제 ([[57]] 계기 확인) ───────────────────
    bare = base_fb.replace(SENT_T, target)            # 꼬리가 통째로 사라진 모양
    bad_cell = tail_tmpl + " {argdoc}"
    a2_bad = copy.deepcopy(a2)
    a2_bad.setdefault("axis_notes", {})["user_action_arglist"] = bad_cell
    BAD_line = build_action_text(block, a2_bad, target, argnames)
    neg_ok = (BAD_line == bare)                       # 계기가 실제로 그 병을 잡아내는가
    print("\n[NEG] 자리표시자를 새로 만들면? (:10562 `except Exception: pass` 부정통제)")
    print("    셀 = %r" % bad_cell)
    print("    산출 길이 %d (A 팔 %d) → `Arguments of ...` 문장 소멸: %s"
          % (len(BAD_line or ""), len(A_line), "YES(조용히 삼켰다)" if neg_ok else "NO"))
    print("    ⇒ 부정통제 %s. B 팔이 자리표시자를 **안 쓰는** 이유가 이것이다 —"
          % ("성립(계기가 산다)" if neg_ok else "불성립 ⚠계기 의심"))
    print("      이 칸을 그렇게 짜면 문장이 조용히 사라져 측정이 무효가 된다.")
    b_swallowed = (B_line == bare)
    g3 = (ARGDOC in (B_line or "")) and not b_swallowed
    print("\n[G3] B 산출이 env 축자를 실제로 담는가 ?  %s   (B 팔 삼킴: %s)"
          % ("YES" if g3 else "NO", "YES" if b_swallowed else "NO"))

    # ── G4: 사이드카 상한 ──────────────────────────────────────────────
    grew = len(live_text) - len(A_line) + len(B_line or "")
    g4 = grew <= args.cap
    print("\n[G4] 사이드카 상한(T2_FB_SIDECAR_TEXT_MAX=%d · go_stack.sh:216)" % args.cap)
    print("    라이브 전문 %d자 → B 팔 적용 시 %d자 → 절단: %s"
          % (len(live_text), grew, "NO" if g4 else "YES ⚠계기 유실"))
    print("    ⚠사이드카 `len` 필드는 자르기 **전** 길이다 — 그 값으로 판정하면 안 된다.")

    # ── SIGN: [[70]] 부호표 ────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("[SIGN] [[70]] 부호표 — referent 를 좁히면 정당한 다른 지목이 함께 막히는가")
    print("-" * 78)
    print("    기전 정의: «그 도구의 **자기** 선언 줄을 축자로 이어 붙인다». 도구마다 다른 말을")
    print("    하므로 원리상 과잉일반화가 없다 — 아래는 그것을 user-실행 도구 전수로 보여 준다.")
    tools = (env_dom.get("tools") or {})
    at = set(a2.get("action_tools") or [])
    n_tool = n_line = n_trunc = n_none = 0
    third_party_kept = []
    shared = []                       # 표적과 **같은 이름의 인자**를 가진 도구들의 선언줄
    for nm in sorted(tools):
        v = tools[nm] or {}
        if v.get("side") != "user_tools":
            continue
        an = list(v.get("args") or [])
        if not an:
            continue
        n_tool += 1
        dl, dm = declared_arg_lines(v.get("desc"), an)
        trunc = len(v.get("desc") or "") >= 200
        n_trunc += 1 if trunc else 0
        n_line += len(dl)
        n_none += 1 if not dl else 0
        mark = "*" if nm == target else ("A" if nm in at else "·")
        print("  %s %-32s args=%-2d 선언줄=%d%s" % (mark, nm, len(an), len(dl),
                                                   "  ⚠desc 절단" if trunc else ""))
        shared.extend([(nm, s) for a, s in dl if a == argnames[0]])
        for a, s in dl:
            print("      %-22s | %s" % (a, s))
        if dm:
            print("      (줄 미확보: %s)" % dm)
        if nm == target:
            for a, s in dl:
                if a != an[0]:
                    third_party_kept.append((a, s))
    print("\n    합계: user-실행 도구 %d · 선언줄 %d · 줄 0개 도구 %d · desc 절단 %d"
          % (n_tool, n_line, n_none, n_trunc))
    print("    ★부호표 판정 재료:")
    print("      (+) 표적의 첫 인자에 referent 가 붙는다: %r"
          % (decl_lines[0][1] if decl_lines else None))
    print("      (−) 정당한 제3자 지목이 함께 막히는가 → 같은 스니펫의 나머지 줄을 본다:")
    for a, s in third_party_kept:
        print("          %-16s | %s" % (a, s))
    print("      ⇒ 제3자가 **다른 슬롯으로 이동**할 뿐이면 (−) 는 0이다.")
    print("        위 축자가 그 판정의 전부다 — 프로브는 부호를 대신 선언하지 않는다([[62]]).")
    print("\n    ⚠대안 팔의 과잉 계수 — 설계서 x737 이 예시로 든 doc_009 문면")
    print("      «Tell the customer to pass their own user_id» 는 **도구와 무관하게 같은 문장**을")
    print("      붙인다. 그것이 닿는 사정권 = '%s' 인자를 가진 도구 %d개, 그 선언줄을 전수로 편다:"
          % (argnames[0], len(shared)))
    self_bound = [(nm, s) for nm, s in shared
                  if re.sub(r"^\S+\s*(\([^)]*\))?\s*:\s*", "", s).lower().startswith("your")]
    other_bound = [(nm, s) for nm, s in shared if (nm, s) not in self_bound]
    print("      ┌ env 가 스스로 '본인' 이라 말하는 도구 %d개 (대안 팔이 참을 말한다)"
          % len(self_bound))
    for nm, s in self_bound:
        print("      │  %-30s | %s" % (nm, s))
    print("      └ env 가 '본인' 이라 말하지 **않는** 도구 %d개 (대안 팔은 여기서 env 를 넘어선다)"
          % len(other_bound))
    for nm, s in other_bound:
        print("         %-30s | %s" % (nm, s))
    print("      ⇒ 이 %d개가 [[70]] 의 «판 것» 후보다. B 팔(자기 선언줄 전달)은 도구마다"
          % len(other_bound))
    print("        다른 말을 하므로 이 칸이 원리상 0이다 — **측정이 아니라 구성에 의해** 0이다.")
    print("        ⚠구성에 의한 0 은 라이브 반증을 면제하지 않는다([[77]]).")

    # ── WIRE: 출시형 ───────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("[WIRE] 배선(출시형) — 이 프로브의 B 팔은 **측정용**이지 출시형이 아니다([[05]])")
    print("-" * 78)
    print("    B 팔은 표적 도구의 축자를 A2 셀에 리터럴로 넣는다 ⇒ 도메인/도구 특화 = 출시 금지.")
    print("    출시형은 엔진이 **이미 읽고 있는 같은 dict 의 값 쪽**을 읽어 전달만 한다:")
    print("      %s:10552-10553  `_pn = list(((_fn.get(\"parameters\") or {})"
          % os.path.basename(ENGINE_FILE))
    print("                       .get(\"properties\") or {}).keys())`   ← 키만 쓴다")
    print("      ⇒ 값 쪽 `properties[arg].get(\"description\")` 를 읽어 format kwarg 한 개 추가.")
    print("    ⛔전제(미확정·[[77]]): 라이브 banking `openai_schema` 가 per-arg description 을")
    print("      담는지 이 머신에서 확인 못 했다(`import tau2` → ModuleNotFoundError). 확인된 것은")
    print("      ⑴ 보관 덤프 specs/s1_inputs/telecom_tools_openai.json 이 담는다는 것,")
    print("      ⑵ a2/env_surface.json 의 desc 가 같은 문장을 담는다는 것뿐. 배선 전 리모트에서 1회 찍어라.")

    # ── STAGE-1 요약 ───────────────────────────────────────────────────
    ok1 = g1 and g2 and g3
    print("\n" + "=" * 78)
    print("[STAGE-1] %s — 잰 것: **선언 문면**. 못 잰 것: 유저 거동."
          % ("PASS" if ok1 else "FAIL"))
    print("  G1 라이브 동일성 %s · G2 한 칸 %s · G3 축자 실림 %s · G4 상한 %s"
          % ("O" if g1 else "X", "O" if g2 else "X", "O" if g3 else "X", "O" if g4 else "X"))
    print("  ⇒ exit «유저가 본인 id 로 호출하는가» 는 여기서 **확정되지 않는다**([[77]] 모른다).")
    print("=" * 78)

    if not args.live:
        print("\n[STAGE-2] 건너뜀 (무료 오프라인). 켜려면:")
        print("   py -3 %s --live HOST:PORT [--n 4]" % os.path.basename(__file__))
        print("   ⚠[[30]] 함정 6: 같은 포트를 다른 모델이 이어받는다 → /v1/models id 대조 필수.")
        print("     기대 모델은 회수 번들 info.agent_info.llm 에서 읽는다(타이핑 0).")
        return 0 if ok1 else 1

    # ── STAGE-2 ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[STAGE-2] 에이전트 hop 1 — 회수 결정 창을 되꽂고 재생성")
    print("=" * 78)
    bundle = read_gz_json(bundle_p)
    sims = [s for s in (bundle.get("simulations") or [])
            if s.get("task_id") == task_id and s.get("seed") == seed]
    if not sims:
        print("⛔ 주장 금지 — 번들에 %s seed=%d sim 이 잡히지 않는다." % (task_id, seed))
        return 2
    sim = sims[0]
    msgs = sim.get("messages") or []
    K = int(live.get("turn"))
    if not (0 <= K < len(msgs)) or msgs[K].get("role") != "assistant":
        print("⛔ 주장 금지 — turn=%s ↔ messages 인덱스 정렬을 확정 못 했다(len=%d, role=%s)."
              % (K, len(msgs), msgs[K].get("role") if 0 <= K < len(msgs) else None))
        print("   (사이드카 turn 이 messages 인덱스라는 일반 규칙은 성립하지 않는다 — 내용 대조 필요.)")
        return 2
    draft_rows = [r for r in rows if r.get("kind") == "reminder-assistant"
                  and r.get("turn") == K and r.get("channel") == FB_CHANNEL]
    draft = (draft_rows[-1].get("text") if draft_rows else None)
    print("  결정 창 = messages[:%d] + 초안(%d자) + 우리 피드백" % (K, len(draft or "")))
    print("  ⚠결손 고지: 회수 번들에 **에이전트 system prompt 가 없다**. 두 팔 모두 그것 없이")
    print("    돌린다 — 절대 수준은 라이브와 다르고, 짝지은 A/B 차이만 읽어야 한다.")

    ginfo = (bundle.get("info") or {}).get("agent_info") or {}
    expect = args.expect_model or str(ginfo.get("llm") or "")
    temp = args.temp
    if temp is None:
        temp = float(((ginfo.get("llm_args") or {}).get("temperature")) or 0.0)
    base = "http://%s/v1" % args.live
    try:
        ids = served_model(base)
    except Exception as e:
        print("⛔ %s 에 /v1/models 조회 실패: %r — 주장 금지." % (base, e))
        return 2
    short = expect.split("/")[-1]
    hit = [i for i in ids if short and short in i]
    print("\n  [[30]]함정6 모델 대조: served=%s / 기대=%s(회수 info.agent_info.llm) → %s"
          % (ids, expect, "MATCH" if hit else "MISMATCH"))
    if not hit and not args.allow_model_mismatch:
        print("  ⛔ 다른 모델이 그 포트를 이어받았다. 중단(--allow-model-mismatch 로 강제 가능).")
        return 2
    model = (hit or ids)[0]

    gold_user = [a for a in ((bundle.get("tasks") or [{}])[0]
                             .get("evaluation_criteria") or {}).get("actions", [])
                 if a.get("requestor") == "user" and a.get("name") == target]
    gold_uid = (gold_user[0].get("arguments") or {}).get(argnames[0]) if gold_user else None
    print("  채점 키(gold·채점 전용): %s.%s == %r" % (target, argnames[0], gold_uid))

    ctx = to_openai(msgs[:K])
    if draft:
        ctx = ctx + [{"role": "assistant", "content": draft}]
    B_text = live_text.replace(A_line, B_line, 1)
    if B_text == live_text:
        print("  ⛔ B 팔 치환 실패(라이브 전문에서 A 줄을 못 찾음) — 주장 금지.")
        return 2
    arms = {"A_off": live_text, "B_on": B_text}

    tally = {}
    for arm, feedback in arms.items():
        tally[arm] = {"SELF": 0, "OTHER": 0, "UNBOUND": 0, "NO_MENTION": 0, "ERR": 0}
        print("\n  ── %s (피드백 %d자) ──" % (arm, len(feedback)))
        for k in range(args.n):
            try:
                out = chat(base, model, ctx + [{"role": "user", "content": feedback}],
                           temp if k == 0 else max(temp, 0.7), args.max_tokens)
            except Exception as e:
                tally[arm]["ERR"] += 1
                print("    #%d ERROR %r" % (k, e))
                continue
            if target not in out:
                tally[arm]["NO_MENTION"] += 1
                print("    #%d NO_MENTION (%d자)" % (k, len(out)))
                continue
            fillers = slot_fillers(out, argnames[0])
            if not fillers:
                tally[arm]["UNBOUND"] += 1
                print("    #%d UNBOUND — 슬롯 filler 0" % k)
                continue
            for f, ctxs in fillers:
                if gold_uid and gold_uid in f:
                    lab = "SELF"
                elif re.fullmatch(r"[0-9a-f]{6,}", f):
                    lab = "OTHER"
                else:
                    lab = "UNBOUND"
                tally[arm][lab] += 1
                print("    #%d %-7s filler=%r" % (k, lab, f))
                print("        …%s…" % ctxs.replace("\n", " ")[:220])

    print("\n  [TALLY] (n=%d/팔)  %s" % (args.n, json.dumps(tally, ensure_ascii=False)))
    print("  ⚠이 표는 **에이전트의 지목**이다. 유저-쓰기(=reward 단위)가 아니다.")
    print("    UNBOUND/OTHER 가 B 에서도 남으면 R2(수리안이 그 창을 못 돌린다).")
    print("    A↔B 분포가 같으면 효과 0 — 그래도 유저-쓰기 자체는 라이브 런에서만 확정된다.")
    return 0 if ok1 else 1


if __name__ == "__main__":
    sys.exit(main())
