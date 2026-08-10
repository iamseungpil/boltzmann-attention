# -*- coding: utf-8 -*-
r"""x217 — **무엇이 읽기를 방해하는가**: 문맥 성분을 하나씩 빼며 추적 (유료 0 · 엔진 0).

## 가설 (사용자 2026-08-10)

> *"이전 프롬프트에서 읽기를 방해하는 steer 가 있을 것 같다. 지시나 이런 걸 격리하면 읽기도 될
>  것 같다. 어떤 게 방해하는 원인인지 추적해 보라."*

## 이미 나온 것 (x216·C398)

  ISO      깨끗한 문맥 + 요구        **7/8** 호출
  ISO_NO   깨끗한 문맥 + 요구 없음   **0/8** (대신 `KB_search_bm25` 6/8 — **기본 prior 는 검색**)
  라이브   요구 12회 + 이름 변환 12회 → **0회**

⇒ prior 는 실재하되 **깨끗한 문맥에서는 요구가 그것을 뒤집는다**. 궤적에서 뒤집기가 죽는다.

## 설계 — 요구·도구 스키마는 **완전히 고정**하고 문맥 성분만 뺀다

  T_FULL       실제 실패 궤적 + 요구            ← 궤적 기준선
  T_noKB       − KB 검색 결과 메시지
  T_noOURS     − **우리가 주입한 문장**(`[T2_`·통과표·상태별세기·창산수·출처요구 서명)
  T_noASSIST   − 이전 **어시스턴트 발화**(자기-정박)
  T_noUSER     − 손님의 **이관 요청·거절** 발화
  T_TAILONLY   마지막 2턴만 (혼잡 거의 0 · 정보도 거의 0 — 상한 아님·참고)
  ISO          깨끗 + 요구                      ← 천장
  ISO_LONG     깨끗 + **무관한 텍스트로 T_FULL 과 같은 길이까지 패딩** + 요구  ← **길이 통제**
  ISO_NO       깨끗 + 요구 없음                 ← prior 기본값(부정 통제)

판정 — `ISO` 높고 `T_FULL` 낮은데 **`ISO_LONG` 이 높으면 길이가 아니다**. 그다음 어느 성분을
뺐을 때 회복되는가가 **범인의 이름**이다. 아무것도 회복 못 시키면 성분 하나가 아니라 누적이다.

⚠팔마다 **무엇이 남았는지 인쇄**하고 재는 것이 이 파일의 규율이다(오늘 프로브 구성 실수 3회의
  공통 원인이 그것을 안 한 것).

실행: python x217_read_steer_trace.py [N]
"""
import collections
import glob
import gzip
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import TOOLS, TARGET, DEMAND, SAID, chat, called_target  # noqa: E402

PATS = ["/home/woori/scratch/tau2-bench/data/simulations/*/results.json",
        "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/"
        "sim_results/*.json.gz"]
OURS = ("[T2_", "Policy constants on record", "grouped by the status each record carries",
        "Date arithmetic on the records above", "[COMPUTED FACTS]",
        "was NOT checked against any allowance", "no room left this year")
PAD = ("The bank's lobby hours are unchanged this quarter. Parking validation is offered at the "
       "garage next door. The quarterly newsletter is mailed to registered addresses. ")


def pick_case():
    """**실제 거부 사례**를 고른다 — 계좌를 안 읽었고 **혼잡이 실재**하는 궤적.

    ⚠**자기적발 (1차 실행)**: 조건을 *"안 읽은 099 실패 sim"* 으로만 걸었더니 KB 덤프도 우리
      주입도 **0** 인 14메시지짜리 짧은 궤적이 뽑혔고, 거기에 요구를 붙이니 `T_FULL` 이 **10/10**
      이었다. 방해 steer 가 없는 문맥에서는 성분을 빼도 가릴 것이 없다 — **재는 대상이 없는
      사례를 고른 것**이다.

    ⇒ 이제 **혼잡 실재**를 조건으로 넣고(KB 결과 또는 우리 주입이 있어야 한다), 후보 중
      **가장 혼잡한 것**을 고른다. 후보 목록도 인쇄해 감사 가능하게 둔다.
    """
    cands = []
    for pat in PATS:
        for p in sorted(glob.glob(pat)):
            try:
                f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
                d = json.load(f)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for s in d.get("simulations") or []:
                if not isinstance(s, dict) or s.get("task_id") not in ("task_099", "task_100"):
                    continue
                if (s.get("reward_info") or {}).get("reward") == 1:
                    continue
                msgs = s.get("messages") or []
                blob = "\n".join(str(m.get("content") or "") for m in msgs)
                calls = " ".join(str((tc.get("function") or tc).get("name") or "")
                                 + str((tc.get("function") or tc).get("arguments") or "")
                                 for m in msgs for tc in (m.get("tool_calls") or []))
                if "get_all_user_accounts" in calls:
                    continue                       # 읽은 sim 은 재현 대상이 아니다
                kb = blob.count("Score:")
                ours = sum(blob.count(sig) for sig in OURS)
                if len(msgs) < 8 or (kb == 0 and ours == 0):
                    continue                       # 혼잡이 없으면 잴 것이 없다
                cands.append((kb + ours, kb, ours, len(msgs), len(blob),
                              os.path.basename(os.path.dirname(p)), s.get("trial"), msgs))
    if not cands:
        return None, None, None
    cands.sort(key=lambda x: -x[0])
    print("후보 %d개 (혼잡 순):" % len(cands))
    for c in cands[:6]:
        print("   KB %-3d 우리 %-3d 메시지 %-3d %7d자  %s trial=%s"
              % (c[1], c[2], c[3], c[4], c[5], c[6]))
    top = cands[0]
    return top[5], top[6], top[7]


def render(msgs, drop=()):
    parts = []
    for m in msgs:
        role = m.get("role")
        raw = str(m.get("content") or "")
        c = " ".join(raw.split())
        if "kb" in drop and role == "tool" and "Score:" in raw:
            continue
        if "ours" in drop and any(sig in raw for sig in OURS):
            continue
        if "assist" in drop and role == "assistant":
            continue
        if "user" in drop and role == "user" and re.search(
                r"transfer|human agent|i (can'?t|don'?t|won'?t)|not comfortable", c, re.I):
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or tc
            parts.append("[%s calls %s]" % (role, fn.get("name")))
        if c:
            parts.append("[%s] %s" % (role, c))
    return "\n".join(parts)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    tag, trial, msgs = pick_case()
    if not msgs:
        print("⚠계좌를 안 읽은 099 실패 sim 을 못 찾았다.")
        return 1
    full = render(msgs)
    arms = [
        ("T_FULL", full),
        ("T_noKB", render(msgs, ("kb",))),
        ("T_noOURS", render(msgs, ("ours",))),
        ("T_noASSIST", render(msgs, ("assist",))),
        ("T_noUSER", render(msgs, ("user",))),
        ("T_TAILONLY", "\n".join(full.split("\n")[-4:])),
        ("ISO", SAID["task_099"]),
        ("ISO_LONG", SAID["task_099"] + "\n\n" + PAD * max(1, len(full) // len(PAD))),
        ("ISO_NO", SAID["task_099"]),          # 요구를 안 붙인다 (아래에서 분기)
    ]
    print("사례 %s trial=%s · 메시지 %d · 궤적 %d자 · n=%d" % (tag, trial, len(msgs), len(full), n))
    print("%-11s %8s %6s %6s %6s %6s" % ("팔", "자수", "KB", "우리", "어시", "손님"))
    for name, body in arms:
        print("%-11s %8d %6d %6d %6d %6d"
              % (name, len(body), body.count("Score:"),
                 sum(body.count(s) for s in OURS),
                 body.count("[assistant]"), body.count("[user]")))
    print()
    out = {}
    for name, body in arms:
        p = body + ("" if name == "ISO_NO" else "\n\n" + DEMAND) + "\n\nWhat do you do next?"
        c = collections.Counter()
        for i in range(n):
            try:
                m = chat(p, TOOLS, 0.0 if i == 0 else 0.7)
            except Exception:
                c["ERR"] += 1
                continue
            c["호출O" if called_target(m) else "호출X"] += 1
            for tc in (m.get("tool_calls") or []):
                c["도구:" + str((tc.get("function") or {}).get("name"))] += 1
        out[name] = [c["호출O"], n]
        print("  %-11s 읽기 %d/%d   %s"
              % (name, c["호출O"], n,
                 [x for x in c.most_common(5) if str(x[0]).startswith("도구")]))
    # ── §2 블록 성장 (사용자 지시) — 뒤에서부터 블록을 **하나씩 얹으며** 어디서 죽는가 ──
    #    성분 절제가 *무엇이* 방해하는지를 묻는다면, 이쪽은 *얼마나 얹으면* 죽는지를 묻는다.
    #    두 답이 만나는 지점이 우리가 격리 서브에 담아도 되는 문맥의 경계다.
    print("\n§2 블록 성장 — **깨끗한 격리에서 시작해** 궤적 블록을 하나씩 얹는다 (요구·도구 고정)")
    print("   (사용자 지시: *'깨끗한 격리부터 여러 추가 블록까지 세밀하게 엄격하게 — 읽기 거부의 기전 확보'*)")
    blocks = []
    for m in msgs:
        role = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        piece = []
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or tc
            piece.append("[%s calls %s]" % (role, fn.get("name")))
        if c:
            piece.append("[%s] %s" % (role, c))
        if piece:
            blocks.append("\n".join(piece))
    def kind_of(b):
        """그 블록이 **무엇인가** — 꺾이는 자리에서 범인의 이름이 된다."""
        if any(s in b for s in OURS):
            return "우리주입"
        if "Score:" in b:
            return "KB결과"
        if b.startswith("[tool]"):
            return "도구출력"
        if b.startswith("[assistant"):
            return "어시스턴트"
        if b.startswith("[user"):
            return "손님"
        return "기타"

    print("  블록 %d개 · 기저 = 깨끗한 격리(손님의 말)" % len(blocks))
    base = SAID["task_099"]
    prev = None
    for k in range(0, len(blocks) + 1):
        body = base if k == 0 else ("\n".join(blocks[-k:]) + "\n\n" + base)
        c = collections.Counter()
        for i in range(n):
            p = body + "\n\n" + DEMAND + "\n\nWhat do you do next?"
            try:
                m2 = chat(p, TOOLS, 0.0 if i == 0 else 0.7)
            except Exception:
                c["ERR"] += 1
                continue
            c["호출O" if called_target(m2) else "호출X"] += 1
            for tc in (m2.get("tool_calls") or []):
                c["도구:" + str((tc.get("function") or {}).get("name"))] += 1
        out["grow-%02d" % k] = [c["호출O"], n]
        added = ("(깨끗)" if k == 0 else "%s: %s" % (kind_of(blocks[-k]),
                                                    blocks[-k].split("]")[0][:22] + "]"))
        drop = ("  ← **여기서 꺾인다**"
                if prev is not None and c["호출O"] <= prev - max(2, n // 3) else "")
        prev = c["호출O"]
        print("  +%2d블록 (%6d자) 읽기 %2d/%d  방금 얹은 것 = %-34s%s  %s"
              % (k, len(body), c["호출O"], n, added[:34], drop,
                 [x for x in c.most_common(2) if str(x[0]).startswith("도구")]))
    json.dump(out, open(os.environ.get("T2_X217_OUT", "x217_out.json"), "w"), indent=1)
    print("\n※ ISO 높고 T_FULL 낮은데 ISO_LONG 이 높으면 **길이가 아니다**."
          "\n  어느 성분을 뺐을 때 회복되는가 = 범인. §2 에서 읽기가 꺾이는 블록이 그 범인을 지목한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
