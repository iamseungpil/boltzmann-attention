# -*- coding: utf-8 -*-
r"""x582 — P-A 술어를 바꾸면 **어디가 갈리나**를 배선 전에 센다 (모델 0 · 무료 · 계수만).

## 왜 (2026-08-28 밤)

t7376 에서 P-A(`T2_ACTIONREQ_GROUNDED`)가 016 의 `submit_referral` 지목을 15회 침묵시켰고,
그와 함께 통과 프레임(`750` 발화)이 23·12 -> 0·1 로 사라졌다. 핸드오프 §4 는 술어를
*"이 이름이 대화에 나왔나"* -> *"미이행 손님-액션인가"* 로 바꾸자고 했는데, 구현을 읽어 보면

    _upending = sorted(_uacts - _effall)     # 이미 "아직 안 불린 손님-액션"
    if _utgt in _upending:                   # 이미 바깥 조건이다
        if GROUNDED and str(_utgt) not in 대화축자: 침묵

**미이행 조건은 이미 걸려 있다** — 그대로 바꾸면 P-A 를 끄는 것과 같다. 그러므로 실제 구분자를
다른 데서 찾아야 하고, 후보는 하나다:

    016 손님은 "referral bonus" 를 계속 말하지만 **도구 이름 `submit_referral` 은 안 말한다**.
    072 손님은 ATM 수수료 크레딧 얘기만 하고 `submit_transaction` 은 주제 자체가 아니다.

=> 술어를 **이름 축자**에서 **주제어 축자**로 넓히면 016 은 발화하고 072 는 침묵할까?
   그것만 센다. **고치기 전에 반경을 재는 것이 목적이고, 여기서 아무것도 안 고친다.**

## 무엇을 세나 (닫힌 술어뿐 · gold 무참조)

각 `[T2_ACTIONREQ] window=open ... formalized_target=X` 결정점마다:

    name_hit   X 가 대화 축자에 있나                (현행 술어 · 있으면 발화)
    subj_all   X 의 **고유 토큰**이 대화 어디든 있나  (변형 A)
    subj_user  X 의 고유 토큰이 **손님 발화**에 있나  (변형 B · 더 엄격)

고유 토큰 = 도구 이름에서 범용어(submit/call/get/apply/... )를 뺀 나머지. 낱말 경계로 맞춘다
(`get_bank_account_transactions_9173` 안의 transactions 는 밑줄이 낱말 문자라 안 걸린다).

⛔한계: 현행 술어는 **그 턴까지의 messages** 를 보는데 여기서는 **sim 전체**를 본다 =
   groundedness 의 **상한**이다. 따라서 여기서 "현행이 침묵했을 것"으로 세어지는 수는
   실제보다 **적다**. 로그의 실제 침묵 계수를 나란히 찍어 그 차이를 보인다.
"""
import collections
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIMS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results"))

GENERIC = {"submit", "call", "get", "apply", "change", "give", "unlock", "for", "by", "to",
           "the", "a", "an", "user", "users", "agent", "agents", "tool", "tools",
           "discoverable", "all", "information", "info", "id", "ids", "request", "create"}

RE_WIN = re.compile(r"\[sim=(task_\d+#s\d+)\].*?\[T2_ACTIONREQ\] window=open .*?formalized_target=([a-z_]+)")
RE_SIL = re.compile(r"\[sim=(task_\d+#s\d+)\].*?\[T2_ACTIONREQ\] 침묵: formalized_target=([a-z_]+)")


def subj_tokens(name):
    return [t for t in name.split("_") if t and t not in GENERIC and not t.isdigit()]


def has_word(text, word):
    return re.search(r"\b%s(s|es)?\b" % re.escape(word), text, re.I) is not None


def conv_text(sim, user_only=False):
    out = []
    for m in (sim.get("messages") or []):
        if user_only and m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        if not user_only:
            for tc in (m.get("tool_calls") or []):
                out.append(str(tc.get("name") or ""))
                out.append(json.dumps(tc.get("arguments") or {}, ensure_ascii=False)
                           if not isinstance(tc.get("arguments"), str) else tc["arguments"])
    return "\n".join(out)


def load(tag):
    lg = os.path.join(SIMS, tag + ".log.gz")
    rs = os.path.join(SIMS, tag + ".results.json.gz")
    if not (os.path.exists(lg) and os.path.exists(rs)):
        return None, None
    with gzip.open(lg, "rt", encoding="utf-8", errors="replace") as f:
        log = f.read()
    with gzip.open(rs, "rt", encoding="utf-8", errors="replace") as f:
        res = json.load(f)
    return log, res.get("simulations") or []


def main(argv=None):
    tags = (argv or sys.argv[1:]) or [
        "bank_t7376_treat_20260828", "bank_t7372_control_20260828",
        "bank_t7375_072_20260828", "bank_t7371_treat_20260828",
        "bank_t7370_radius_20260828", "bank_t7369_072_20260828",
        "bank_t7368_hard0_20260827"]
    rows = []
    for tag in tags:
        log, sims = load(tag)
        if log is None:
            print("(건너뜀 · 재료 없음) %s" % tag)
            continue
        by = {}
        for s in sims:
            by["%s#s%s" % (s.get("task_id"), s.get("seed"))] = s
        pts = collections.Counter(RE_WIN.findall(log))
        sil = collections.Counter(RE_SIL.findall(log))
        for (simkey, tgt), n in sorted(pts.items()):
            s = by.get(simkey)
            if s is None:
                continue
            ta = conv_text(s)
            tu = conv_text(s, user_only=True)
            toks = subj_tokens(tgt)
            rows.append({
                "tag": tag, "sim": simkey, "task": simkey.split("#")[0], "target": tgt,
                "n_points": n, "n_silenced_log": sil.get((simkey, tgt), 0),
                "name_hit": tgt in ta,
                "subj_toks": toks,
                "subj_all": bool(toks) and any(has_word(ta, t) for t in toks),
                "subj_user": bool(toks) and any(has_word(tu, t) for t in toks),
                "reward": (s.get("reward_info") or {}).get("reward"),
            })
    if not rows:
        print("행 0 — 재료를 못 읽었다"); return 1

    print("=" * 100)
    print("결정점 census — %d 런 · %d (sim,표적) 짝" % (len(set(r["tag"] for r in rows)), len(rows)))
    print("=" * 100)
    print("%-14s %-24s %-22s %-6s %-6s %-5s %-6s %-7s %s"
          % ("런", "sim", "표적", "결정점", "실침묵", "이름", "주제全", "주제손님", "rew"))
    for r in sorted(rows, key=lambda x: (x["task"], x["sim"], x["target"], x["tag"])):
        print("%-14s %-24s %-22s %-6d %-6d %-5s %-6s %-7s %s"
              % (r["tag"].split("_")[1], r["sim"], r["target"], r["n_points"],
                 r["n_silenced_log"],
                 "O" if r["name_hit"] else "-", "O" if r["subj_all"] else "-",
                 "O" if r["subj_user"] else "-", r["reward"]))

    print("")
    print("=" * 100)
    print("★반경 — 술어를 바꾸면 갈리는 자리 (현행 침묵 <-> 변형이 발화)")
    print("=" * 100)
    for nm, key in (("변형A 주제어(대화 전체)", "subj_all"), ("변형B 주제어(손님 발화만)", "subj_user")):
        flip = [r for r in rows if (not r["name_hit"]) and r[key]]
        keep = [r for r in rows if (not r["name_hit"]) and not r[key]]
        print("  %s" % nm)
        print("     현행이 침묵하던 것 중 **발화로 바뀌는** 짝 %d (결정점 %d · 실침묵 %d)"
              % (len(flip), sum(r["n_points"] for r in flip), sum(r["n_silenced_log"] for r in flip)))
        tb = collections.Counter("%s/%s" % (r["task"], r["target"]) for r in flip)
        print("       %s" % dict(tb))
        print("     그대로 침묵 유지 짝 %d · %s"
              % (len(keep), dict(collections.Counter("%s/%s" % (r["task"], r["target"]) for r in keep))))
        print("")

    print("=" * 100)
    print("판독 (이 프로브는 아무것도 고치지 않는다)")
    print("=" * 100)
    print("  · 016 이 '발화로 바뀌는' 쪽에 있고 072 가 '침묵 유지' 쪽에 있으면 그 변형은 후보다.")
    print("  · 둘 다 발화로 바뀌면 그 변형은 **072 를 되돌린다** = 지금 산 pass 를 판다.")
    print("  · 한계: 대화 전체를 봤으므로 name_hit 은 상한이다(실제 침묵은 로그의 '실침묵' 열).")

    dst = os.path.join(SIMS, "..", "x582_actionreq_subject_census.json")
    with io.open(os.path.normpath(dst), "w", encoding="utf-8") as f:
        json.dump({"probe": "x582_actionreq_subject_census", "date": "2026-08-28",
                   "tags": tags, "rows": rows,
                   "limits": ["대화 전체를 봐서 name_hit 은 groundedness 의 상한이다.",
                              "gold 무참조 — reward 는 참고 열일 뿐 술어에 안 들어간다([[23]]).",
                              "주제어는 도구 이름의 낱말 분해일 뿐 의미 판단이 아니다([[22]])."]},
                  f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
