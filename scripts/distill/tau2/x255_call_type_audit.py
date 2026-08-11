# -*- coding: utf-8 -*-
r"""x254 — **우리가 도구 이름을 댈 때 부를 수 있는 형태로 대는가** (전수 감사 · 유료 0 · 모델 0).

## 왜 (사용자 지시 2026-08-11 · C418 의 일반화)

C418 은 099 에서 한 자리를 잡았다: 우리 `[ORDER]` 가 **발견형 도구를 도구 목록에 있는 것처럼**
이름 댄다. 그런데 이 env 의 호출 방식은 **네 가지**이고 우리는 그 구분을 문구에서 하지 않는다.
사용자 지적: *"도구 호출 타입을 정하고 타입에 따라 부르는 방식을 다르게 하게 명시해야 하지 않나."*

이 감사가 묻는 것은 하나다 — **우리가 이름을 댄 매 자리에서, 그 이름을 그렇게 부를 수 있는가.**

## 호출 타입 (전부 env API 도출 · 도메인 리터럴 0)

  T1 직접        에이전트 도구 목록에 있음               → 그대로 호출
  T2 에이전트-발견형  `env.tools.get_discoverable_tools()`     → unlock 뒤 디스패처로만
  T3 손님-발견형   `env.user_tools.get_discoverable_tools()` → 에이전트가 넘기고 손님이 실행
  T4 손님-기본     `env.user_tools.tools` − T3              → 손님이 실행(에이전트 호출 불가)

## 무엇을 세는가

`fb_*.jsonl`(모델이 실제로 받은 우리 문장) 전수에서 도구 이름 언급을 뽑고 타입별로 가른다.
**결함 = 언급했는데 그 타입의 호출 형식이 같은 문장에 없는 것.** 판정 기준은 타입마다 다르다:
  T2  같은 문장에 디스패처 이름이 있어야 한다
  T3  같은 문장에 넘김 도구(give) 이름이 있어야 한다
  T4  같은 문장이 **손님이 실행한다**고 말해야 한다(우리 A2 문구의 축자 표지로만 판정)
T1 은 검사 대상이 아니다(그대로 부르면 된다).

⚠판정은 **문자열 기준**이라 등급 [M] 이다(x247 ⒠ 와 같다). 수는 자리를 가리키는 데 쓰고,
  고칠지는 사람이 읽고 정한다.

실행(리모트): python x254_call_type_audit.py [TAG ...]
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOGDIR = "/home/woori/scratch/logs"

# **다음 한 수를 요구하는** 문장만 감사한다(우리 요구 채널의 축자 표지). 거부 사유·이력 인용은
# 이름을 대는 자리가 아니다 — 거기까지 세면 신호가 죽는다(첫 판 3031/3869).
DEMAND = re.compile(r"\[ORDER\]|do it with|Steps that are possible right now|"
                    r"next step has not been taken|Do this now|has to hold first")


def env_types():
    """네 집합을 env 에서 그대로 가져온다(우리가 목록을 적지 않는다)."""
    from tau2.domains.banking_knowledge.environment import get_environment
    env = get_environment()
    tk, ut = getattr(env, "tools", None), getattr(env, "user_tools", None)
    t_all = set(getattr(tk, "tools", {}) or {})
    t2 = set(tk.get_discoverable_tools()) if tk is not None else set()
    u_all = set(getattr(ut, "tools", {}) or {})
    t3 = set(ut.get_discoverable_tools()) if ut is not None else set()
    return {"T1": t_all - t2, "T2": t2, "T3": t3, "T4": u_all - t3}


def dispatchers(types):
    """디스패처·넘김 도구도 **구조로** 찾는다: 이름에 다른 도구를 실어 나르는 T1 도구들."""
    out = {"unlock": None, "call": None, "give": None, "ucall": None}
    for n in sorted(types["T1"]):
        if "agent_tool_name" in n or "discoverable_agent" in n:
            out["call" if "call" in n else "unlock"] = n
        elif "discoverable_user" in n:
            out["give" if "give" in n else "ucall"] = n
    return out


def main():
    tags = sys.argv[1:] or [os.path.basename(p)[3:-6]
                            for p in sorted(glob.glob(os.path.join(LOGDIR, "fb_*.jsonl")))]
    types = env_types()
    d = dispatchers(types)
    print("타입 크기: " + " · ".join("%s %d" % (k, len(v)) for k, v in sorted(types.items())))
    print("디스패처(구조 도출): %s\n" % d)

    of_type = {}
    for k, names in types.items():
        for n in names:
            of_type[n] = k
    # 접미사 없는 가족명도 같은 타입으로 본다(우리 문구가 실제로 쓰는 형태다)
    fam = {}
    for n, k in of_type.items():
        base = re.sub(r"_\d{3,}$", "", n)
        fam.setdefault(base, (k, n))

    rows, bad = collections.Counter(), collections.defaultdict(list)
    for tag in tags:
        p = os.path.join(LOGDIR, "fb_%s.jsonl" % tag)
        if not os.path.exists(p):
            continue
        for ln in open(p, encoding="utf-8", errors="replace"):
            try:
                o = json.loads(ln)
            except Exception:
                continue
            txt = o.get("text") or ""
            if not txt.strip():
                continue
            # ★첫 판 무효(2026-08-11 자기교정): **모든 언급**을 셌더니 T2 3031/3869 가 나왔고
            #   대부분이 *요구*가 아니라 거부 사유·이력 인용이었다(그 자리는 호출 형식을 댈 자리가
            #   아니다). [[64]] 가 겨누는 것은 **다음 한 수를 요구하는 문장**뿐이다. x247 ⒟ 가 두 번
            #   밟은 함정과 같다 — 계기가 신호를 만든다. 요구 채널만 남긴다.
            if not DEMAND.search(txt):
                continue
            for base, (k, real) in fam.items():
                if k == "T1" or base not in txt:
                    continue
                rows[k] += 1
                ok = True
                if k == "T2":
                    ok = bool(d["call"]) and d["call"] in txt
                elif k == "T3":
                    ok = bool(d["give"]) and d["give"] in txt
                elif k == "T4":
                    ok = bool(re.search(r"tell the customer to run|the customer .{0,20}run|"
                                        r"customer executes|ask the customer to run", txt))
                if not ok:
                    bad[k].append((tag, o.get("turn"), base, txt[:160]))
    print("언급 수(T1 제외): " + " · ".join("%s %d" % (k, rows[k]) for k in ("T2", "T3", "T4")))
    for k in ("T2", "T3", "T4"):
        print("\n== %s 결함 %d/%d (그 타입의 호출 형식이 같은 문장에 없다)"
              % (k, len(bad[k]), rows[k]))
        seen = set()
        for tag, turn, base, txt in bad[k]:
            key = (base, txt[:60])
            if key in seen:
                continue
            seen.add(key)
            print("   [%s t%s] %s :: %s" % (tag, turn, base, " ".join(txt.split())[:150]))
            if len(seen) >= 6:
                print("   … (%d건 더)" % (len(bad[k]) - len(seen)))
                break
    print("\n※ 문자열 기준 = [M]. 수는 자리를 가리킬 뿐, 고칠지는 축자를 읽고 정한다.")


if __name__ == "__main__":
    main()
