# -*- coding: utf-8 -*-
r"""x516 — G3 격리: **후보에서 `submit_referral` 을 빼면 서브가 gold 를 고르는가** (x509 S3 선행).

사용자 승인 2026-08-24 — *"G3 먼저 하라"*.

## 무엇을 재는가 (x515 가 남긴 유일한 미지수)

x515 재생이 G1·G2 를 채웠다: `_exec_side` 의 `UNKNOWN→user` 폴백을 제거하면 손님-실행 안내가
걸린 발화 307 중 **129 가 사라지고**(016 은 59/59 전부), 그 22 sim 은 **전부 모델 앞에 섰다**.
그리고 gold 의 `requestor=user` 액션은 10 태스크 **전부** 그 태스크 `user_tools` 안에 있으므로
**떨어뜨리는 이름 중 gold 는 없다**.

남은 것은 재생이 원리상 답할 수 없는 하나다 — 후보집합이 좁아지면 격리 서브
(`t2_resolve.formalize_intent_tool`)가 **무엇을 대신 고르는가**. gold `submit_transaction` 이
같은 집합에 남아 있으므로 그것을 고르면 수리가 반사실까지 채운다.

## 계기 — 사본 0 ([[67]])

프롬프트·파싱·집합소속 검사를 베끼지 않는다. **정본 `t2_resolve.formalize_intent_tool` 을 그대로
호출**하고 `action_tools` 인자만 팔마다 바꾼다. LLM 은 `t2_subcall.sub_generate` 를 지나가므로,
`agent`/`la`/`UserMessage` 자리에 프로브 엔드포인트로 나가는 최소 어댑터만 끼운다.

## 재료 — 궤적 축자 ([[62]] 2b · 공정한 격리)

그 함수가 실제로 보는 것은 **마지막 손님 발화 6개**뿐이다. 그래서 016 sim 들의 메시지에서 그
창을 그대로 뜯는다. 후보집합도 지어내지 않고 로그의 `[T2_ACTIONREQ]` 줄이 인쇄한
`pending_user ∪ pending_agent` 를 쓴다. 도메인 리터럴 0 · gold 0([[23]]).

## 팔

    A_asis    로그가 인쇄한 후보집합 그대로              → 현재 산출(`submit_referral`)이 재현되나
    B_repair  거기서 `submit_referral` 제외              → UNKNOWN 폴백 제거 후의 집합
    N_neg     같은 크기로 **무관한 이름 하나**를 대신 제외 → 크기 축소 자체의 효과 통제([[57]])

N_neg 이 B_repair 만큼 움직이면 산 것은 수리가 아니라 **집합이 작아진 것**이다.

## 실행 (리모트 · GPU1 · 무료)

    PYTHONIOENCODING=utf-8 python x516_induction_target_iso.py --port 8141
"""
import argparse
import collections
import glob
import gzip
import io
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"

SIMRE = re.compile(r"\[sim=(task_\d+#s\d+)\]")
AREQ = re.compile(r"\[T2_ACTIONREQ\] window=(\w+) pending_user=(\[[^\]]*\]) "
                  r"pending_agent=(\[[^\]]*\]) formalized_target=(\S+)")
TASK = "task_016"
DROP = "submit_referral"          # x515 가 이 태스크에서 이름밖으로 판정한 유일한 표적


def parse_list(s):
    return [x.strip().strip("'\"") for x in s[1:-1].split(",") if x.strip()]


# ── 프로브 어댑터: 정본 서브콜이 지나가는 자리에만 끼운다 (판단 0) ────────────────
class _UM(object):
    def __init__(self, role="user", content=""):
        self.role = role
        self.content = content


class _Msg(object):
    __slots__ = ("role", "content")

    def __init__(self, role, content):
        self.role = role
        self.content = content


class _Agent(object):
    llm = MODEL
    llm_args = {"temperature": 0.0}


class _LA(object):
    def __init__(self, port, maxtok=64):
        self.port = port
        self.maxtok = maxtok
        self.calls = 0

    def generate(self, model=None, tools=None, messages=None, call_name=None, **kw):
        body = str(getattr(messages[0], "content", "") or "")
        payload = {"model": MODEL, "temperature": 0.0, "max_tokens": self.maxtok,
                   "messages": [{"role": "user", "content": body}]}
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % self.port,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as r:
            txt = json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]
        self.calls += 1
        return type("_R", (), {"content": txt})()


# ★큐 세대만 본다 — 코퍼스 전량을 긁으면 옛 런의 후보집합이 섞인다(§74 세대-뭉개기).
RUNS = ("bank_t7348_halfA_20260824", "bank_t7348_halfB_20260824",
        "bank_t7346_halfA_20260822", "bank_t7346_halfB_20260822")


def windows():
    """(simtag, 후보집합, 런타임 표적 분포, 손님 발화 창) — 전부 궤적·로그 축자.

    ★창은 **대화 진행에 따라** 만든다. `formalize_intent_tool` 이 보는 것은 *그 시점까지의*
      마지막 손님 발화 6개인데, 초판은 sim 전체 메시지에서 마지막 6개만 떠서 **모든 창이
      대화 끝(`###TRANSFER###`)** 이 됐다 — 그 자리에서는 gold 가 정답이 아니므로 거짓 음성이
      나온다. 손님 발화 인덱스마다 창을 하나씩 만든다.
    후보집합은 지어내지 않고 그 sim 의 `[T2_ACTIONREQ]` 줄이 인쇄한 것 중 **가장 흔한 것**을 쓴다.
    """
    cases = []
    for tag in RUNS:
        rp = os.path.join(SIMS, tag + ".results.json.gz")
        lp = os.path.join(SIMS, tag + ".log.gz")
        if not (os.path.exists(rp) and os.path.exists(lp)):
            continue
        d = json.load(gzip.open(rp, "rt", encoding="utf-8", errors="replace"))
        sims = {}
        for s in (d.get("simulations") or []):
            if s.get("task_id") != TASK:
                continue
            key = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            sims[key] = [_Msg(m.get("role"), str(m.get("content") or ""))
                         for m in (s.get("messages") or [])]
        if not sims:
            continue
        cand_of = collections.defaultdict(collections.Counter)
        tgt_of = collections.defaultdict(collections.Counter)
        with gzip.open(lp, "rt", encoding="utf-8", errors="replace") as f:
            for ln in f:
                if "[T2_ACTIONREQ]" not in ln:
                    continue
                m0, m1 = SIMRE.search(ln), AREQ.search(ln)
                if not (m0 and m1) or m0.group(1) not in sims:
                    continue
                cand_of[m0.group(1)][tuple(sorted(set(parse_list(m1.group(2))
                                                      + parse_list(m1.group(3)))))] += 1
                tgt_of[m0.group(1)][m1.group(4)] += 1
        for st, msgs in sims.items():
            if not cand_of[st]:
                continue
            cands = list(cand_of[st].most_common(1)[0][0])
            uidx = [i for i, m in enumerate(msgs) if m.role == "user"]
            seen = set()
            for k in range(len(uidx)):
                win = [msgs[j] for j in uidx[max(0, k - 5):k + 1]]
                sig = tuple(m.content[:120] for m in win)
                if sig in seen:
                    continue
                seen.add(sig)
                cases.append({"run": tag, "simtag": st, "cands": cands,
                              "live_target": tgt_of[st].most_common(1)[0][0],
                              "turn_k": k, "msgs": win})
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    import t2_resolve as RZ          # 정본. 프롬프트·파싱·집합소속 전부 이쪽 것.

    cases = windows()
    if a.limit:
        cases = cases[:a.limit]
    if not cases:
        print("재료 없음 — 016 의 ACTIONREQ 줄을 못 찾았다. 돌리지 않는다([[25]]).")
        return 1

    gold_user = None
    for rp in sorted(glob.glob(os.path.join(SIMS, "bank_t73*_half*.results.json.gz"))):
        d = json.load(gzip.open(rp, "rt", encoding="utf-8", errors="replace"))
        for t in (d.get("tasks") or []):
            if t.get("id") == TASK:
                gold_user = sorted({x.get("name") for x in
                                    ((t.get("evaluation_criteria") or {}).get("actions") or [])
                                    if x.get("requestor") == "user"})
        if gold_user:
            break
    print("표적 %s · gold(requestor=user) = %s · 창 %d개"
          % (TASK, gold_user, len(cases)))

    la = _LA(a.port)
    ag = _Agent()
    res = collections.defaultdict(collections.Counter)
    rows = []
    for i, c in enumerate(cases):
        cands = list(c["cands"])
        # N_neg: `DROP` 이 아닌 이름 중 **gold 도 아닌** 하나를 뺀다(같은 크기·무관)
        neg_pool = [x for x in cands if x != DROP and x not in (gold_user or [])]
        arms = {"A_asis": cands,
                "B_repair": [x for x in cands if x != DROP],
                "N_neg": [x for x in cands if x != (neg_pool[0] if neg_pool else None)]}
        row = {"run": c["run"], "simtag": c["simtag"], "live_target": c["live_target"],
               "cands": cands, "neg_dropped": neg_pool[0] if neg_pool else None}
        for arm, ct in arms.items():
            got = RZ.formalize_intent_tool(ag, la, _UM, c["msgs"], ct)
            row[arm] = got
            res[arm][str(got)] += 1
            if gold_user and got in gold_user:
                res[arm]["__GOLD__"] += 1
        rows.append(row)
        print("  [%2d] %s live=%-18s A=%-22s B=%-22s N=%s"
              % (i, c["simtag"], c["live_target"], row["A_asis"], row["B_repair"], row["N_neg"]))

    print("")
    print("=" * 96)
    print("결과 — 팔별 산출 분포 (n=%d 창 · 서브콜 %d회)" % (len(cases), la.calls))
    print("=" * 96)
    for arm in ("A_asis", "B_repair", "N_neg"):
        g = res[arm].pop("__GOLD__", 0)
        dist = " · ".join("%s×%d" % kv for kv in res[arm].most_common())
        print("  %-9s gold %d/%d   %s" % (arm, g, len(cases), dist))
        res[arm]["__GOLD__"] = g
    print("")
    print("판독:")
    print("  A 에서 `%s` 가 나오면 라이브 산출이 격리에서 재현된 것이다." % DROP)
    print("  B 의 gold 가 A 보다 크고 **N_neg 은 안 그러면** 산 것은 수리다.")
    print("  B 와 N_neg 이 같이 오르면 산 것은 **집합이 작아진 것**이다([[57]]).")
    print("  B 도 gold 를 못 고르면 결손은 후보집합이 아니라 **그 위**에 있다 — 수리의 반사실이")
    print("  안 채워지고, x509 S3 는 이 자리로 다시 내려와야 한다.")

    out = {"probe": "x516_induction_target_iso", "date": "2026-08-24",
           "task": TASK, "drop": DROP, "gold_user": gold_user,
           "n_windows": len(cases), "subcalls": la.calls,
           "arms": {k: dict(v) for k, v in res.items()},
           "rows": [{k: v for k, v in r.items() if k != "msgs"} for r in rows],
           "limits": ["창은 `formalize_intent_tool` 이 실제로 보는 것(마지막 손님 발화 6개)뿐이다.",
                      "temperature 0 — 창마다 1회. n 은 창 수이지 재시행 수가 아니다.",
                      "gold 는 **채점에만** 썼다. 후보집합·프롬프트 어디에도 안 들어간다([[23]])."]}
    dst = os.path.join(OUT, "x516_induction_target_iso_2026_08_24.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
