# -*- coding: utf-8 -*-
r"""x585 - 배달된-미호출 도구 이름을 **되짚기만** 하면 부르나 (격리 · 라이브 프롬프트 위).

## 왜 이 프로브인가

`x583`: 표적 쓰기 도구를 끝내 못 부른 20 sim **전부(20/20)** 에서 그 이름은 이미
`role=tool` 본문으로 배달돼 있었고, 그 20 개의 reward 는 전부 0 이다. 072 계열은 배달 시점이
**msg 3** - 대화 첫머리다. 검색을 더 한 쪽이 오히려 못 찾았다(7.2 vs 5.1).
=> 결손은 능력이 아니라 **부하**이고 레버의 자리는 **전달**이다([[62]] §1.4).

`x584`/`x584b`: 무조건 되짚으면 반경이 검색 턴의 74%·후보 중앙값 8 이라 상시 메뉴가 된다.
좁히기 (6)(*방금 읽은 문서가 이름 댄 것만*)이 반경 24% 에 적중 13/20 으로 가장 싸다.

남은 물음은 둘이고 오프라인으로는 못 잰다:

    (사는 것)  되짚으면 그 도구를 **부르나**
    (파는 것)  **다른** 도구를 더 부르나 = Delta-spurious ([[70]] 반대편 계측 의무)

## 팔 (새 사실 0 · 지시 0)

    A_asis     회수된 라이브 프롬프트 그대로        <- 재현 게이트(또 검색해야 한다)
    B_restate  + 재진술 한 줄. **이 프롬프트에 이미 있는 이름만** 되읊는다.
               [금지] "unlock and call one of them" 같은 **행동 지시를 넣지 않는다** - 현행
                 `T2_SEARCH_EXHAUST` 문면이 그렇게 말하는데 그건 떠먹이기이고 over-action 을
                 산다. 여기서는 **전달만** 한다([[62]] (2)가 허락하는 것은 부하 축소뿐).
    N_len      길이만 맞춘 무관 문장([[57]] 부정통제)

## 채점 - 닫힌 술어 · gold 미접촉

    표적호출   출력에 표적 이름이 나오나
    타도구     출력에 나온 **다른** 발견형 도구(`이름_숫자4`) 수   <- Delta-spurious 대리
    또검색     출력이 `KB_search` 를 부르나

## [[62]] 4문

  (1) 재봤나 - `x583` 이 6런 34 sim 으로 쟀다. 못 닿으면 reward 0 이 20/20.
  (2) 격리에서 되나 - **미측정. 그것이 이 프로브의 물음이다.**
  (3) 사라지는 모델 판단 - 없다. 엔진은 이름을 **고르지 않는다**; 이미 배달됐고 아직 안 부른
      목록을 정렬해 그대로 되읊을 뿐이다. 무엇을 부를지는 전부 모델이 정한다.
  (4) 엔진이 argmax·"정답은 X" 를 내나 - 아니오. 표적을 지목하지 않는다(목록에 여럿이 있다).

## 재료 - 이 프로브는 **라이브 프롬프트 덤프**를 요구한다

영속 궤적으로 문맥을 재구성하면 안 된다. `x572` 가 그 길을 이미 판정했다: 라이브 프롬프트는
커밋 메시지 + 비커밋 주입 + 뷰 압축의 합이고 궤적과 **5,216자** 다르다([[78]]).

수집 방법(다음 유료 런에 얹는다 · 계기이지 레버가 아니다):

    export T2_PROMPT_DUMP=1
    export T2_PROMPT_DUMP_TASKS=task_072
    export T2_FB_SIDECAR_TEXT_MAX=80000     # 기본 4000 이면 **시스템 메시지만** 남는다

[주의] `len` 필드로 상한이 열렸는지 판정하지 마라 - 그 값은 **자르기 전 원본 길이**다.
저장분은 `text` 로 재라(2026-08-27 에 그 오독이 있었다).

사용: PYTHONPATH=. py -3 x585_delivered_name_restate_iso.py --dump <fb_*.jsonl.gz> [--wiring-only]
"""
import argparse
import collections
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NL = chr(10)
RE_DISC = re.compile(r"\b[a-z][a-z0-9_]*_\d{4}\b")
TARGET = "apply_checking_account_credit_5829"
CALLED = 'agent_tool_name"'


def prompts(path, simtag=None, minlen=8000):
    out = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("kind") != "prompt":
                continue
            if simtag and r.get("simtag") != simtag:
                continue
            if len(r.get("text") or "") < minlen:
                continue
            out.append(r)
    out.sort(key=lambda r: (r.get("turn") or 0, len(r.get("text") or "")))
    return out


def was_called(text, name):
    """이 프롬프트 안에서 그 도구가 **이미 불렸다는 흔적**이 있나.

    ⚠2026-08-28 수리 — 첫 판은 `agent_tool_name": "<이름>` 만 봤고, 라이브 프롬프트의 렌더가
    그 모양이 아니라 **이미 실행된 도구 둘을 '아직 안 불렀다'고 되읊을 뻔했다**
    (`get_all_user_accounts_by_user_id_3847` 는 `Executed:` 로, `get_bank_account_transactions_9173`
    은 인자 지목으로 흔적이 있었다). 우리 문면이 거짓이면 그건 우리 결함이다([[25]]).
    ★라이브 레버(`_cand9`)는 `state.messages` 로 보므로 이 결함이 **없다** - 여기서 놓치면
    프로브가 라이브와 **다른 문면**을 재게 된다([[78]] iso<->live).
    """
    if ("Executed: " + name) in text:
        return True
    if re.search(re.escape(CALLED) + r"?\s*[:=]\s*\"?" + re.escape(name), text):
        return True
    return False


def _already_committed(text, alt="initial_transfer_to_human_agent"):
    """꼬리에서 에이전트가 **이미 대안을 하겠다고 자기 입으로 선언**했나.

    왜 (2026-08-28 · x590): turn=61 을 결정점으로 잡았는데 그 프롬프트의 꼬리가
    *"I need to unlock and call the `initial_transfer_to_human_agent_0218` tool ... Let me unlock
    and call the `initial_transfer_to_human_agent_0218` tool"* 였다. 거기서는 모델이 **결정하는
    게 아니라 자기 문장을 잇는다**([[18]] 자기-정박·`x572` 동형) - 네 팔(A·B_add·C_sub·N_len)이
    전부 표적 0/5 · 이관지목 5/5 로 **완전히 동률**이었고, 그건 빼기의 실패가 아니라
    **결정점이 아니었다**는 뜻이다.
    ⇒ 결정점의 자격: 그 대안에 대한 **자기 선언이 아직 없어야** 한다. 술어는 닫혀 있다
      (꼬리 2500자에 `unlock|call ... <대안이름>` 이 있나).
    """
    tail = text[-2500:]
    return re.search(r"(unlock|call)[^.]{0,60}" + re.escape(alt), tail, re.I) is not None


def _is_subcall(text):
    """우리 층 서브콜 프롬프트인가 - 꼬리가 **구조 출력**을 요구하면 그렇다."""
    tail = text[-1800:]
    return ('"claims"' in tail) or ('"pending"' in tail) or ("Return ONLY" in tail)


def pick_turn(recs, pick="last"):
    """되짚기가 의미 있는 프롬프트 = 표적이 이미 배달됐는데 **아직 안 불린** 자리.

    ⚠기본은 **가장 늦은** 것이다. 첫 판에서 `turn=4` 를 골랐는데 그건 대화 초반이라
    *"이름을 일찍 대면 부르나"* 를 재는 것이고, 실패는 거기서 나지 않는다. 실물의 결정점은
    에이전트가 **또 검색하고 포기하는 늦은 턴**이다(x584b 의 좁히기 (6)도 그 자리에 선다).
    `--pick first` 로 초반 자리도 볼 수 있게 남겨 둔다."""
    hits = [r for r in recs
            if TARGET in str(r.get("text") or "") and not was_called(str(r.get("text") or ""), TARGET)]
    # 서브콜 프롬프트를 뺀다 (2026-08-28 수리). 첫 판이 `turn=61` 의 **형식화 서브**를 골랐고
    #   세 팔이 전부 `{"claims": [...]}` 를 냈다 - 그 서브는 애초에 도구를 안 부르므로
    #   *"되짚으면 부르나"* 를 잴 수 없는 자리였다([[78]] 격리에서 실패하면 무엇이 빠졌는지부터).
    #   판별은 **우리 자신의 서브 지시 문면**으로 한다(도메인 텍스트 0).
    hits = [r for r in hits if not _is_subcall(str(r.get("text") or ""))]
    # 자기 선언 이후는 결정점이 아니다 (위 `_already_committed`). 남는 게 없으면 필터를 풀고
    # **그 사실을 호출자가 알 수 있게** 종전대로 돌려준다(조용한 통과 금지).
    _open = [r for r in hits if not _already_committed(str(r.get("text") or ""))]
    if _open:
        hits = _open
    if not hits:
        return None
    return hits[0] if pick == "first" else hits[-1]


def restate_line(base):
    names = sorted(set(RE_DISC.findall(base)))
    uncalled = [n for n in names if not was_called(base, n)]
    if not uncalled:
        return "", []
    say = (NL + NL + "[note] Documents already returned in this conversation name these tools, "
           "which have not been called yet: " + ", ".join(uncalled[:6]) + ".")
    return say, uncalled[:6]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="")
    ap.add_argument("--simtag", default="task_072#s373753")
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--pick", default="last", choices=["first", "last"])
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    if not a.dump or not os.path.exists(a.dump):
        sys.stderr.write("[STOP] 재료가 없다 - 라이브 프롬프트 덤프가 필요하다.\n")
        sys.stderr.write("       궤적으로 재구성하지 마라([[78]]·x572: 라이브와 5,216자 다르다).\n")
        sys.stderr.write("       수집: T2_PROMPT_DUMP=1 T2_PROMPT_DUMP_TASKS=task_072 "
                         "T2_FB_SIDECAR_TEXT_MAX=80000\n")
        return 2

    recs = prompts(a.dump, a.simtag)
    r = pick_turn(recs, a.pick)
    if r is None:
        sys.stderr.write("[STOP] 표적 이름이 실린 미호출 프롬프트가 없다 - 판정하지 않는다([[25]])\n")
        return 2
    base = str(r.get("text"))
    say, names = restate_line(base)
    if not say:
        sys.stderr.write("[STOP] 되읊을 미호출 이름이 0 - 판정하지 않는다\n")
        return 2

    print("# x585 - 라이브 프롬프트 turn=%s · %d자 · sim=%s" % (r.get("turn"), len(base), a.simtag))
    print("   되읊는 이름 %d개: %s" % (len(names), ", ".join(names)))
    leak = [n for n in names if n not in base]
    print("   프롬프트 밖 이름 누출: %s" % (leak or "없음"))
    print("   표적이 목록에 있나: %s (지목은 안 한다 - 목록에 %d개가 함께 있다)"
          % (TARGET in names, len(names)))

    if a.wiring_only:
        print("--- B_restate 문면 ---")
        print("   " + " ".join(say.split()))
        print("--- 지시어 검사 (있으면 [[62]] 위반) ---")
        bad = [w for w in ("call ", "unlock", "you should", "rather than", "must ")
               if w in say.lower()]
        print("   %s" % (bad or "지시어 없음 - 재진술뿐"))
        return 0

    import x559_016_row_pick_iso as X559
    adds = {"A_asis": "", "B_restate": say,
            "N_len": NL + NL + ("[note] the information gathered so far in this conversation "
                                "remains current. " * 6)[:len(say)]}
    print("")
    print("%-11s %-5s %-8s %-7s %-7s %s" % ("팔", "temp", "표적호출", "타도구", "또검색", "답"))
    print("-" * 104)
    tally = collections.defaultdict(collections.Counter)
    for nm in ("A_asis", "B_restate", "N_len"):
        body = base + adds[nm]
        for tp, cnt in ((0.0, 1), (a.temp, a.n)):
            for _ in range(cnt):
                try:
                    rep = " ".join(str(X559.gen(a.port, body, 300, tp)).split())
                except Exception as e:
                    print("%-11s %-5s 호출 실패: %r" % (nm, tp, e))
                    continue
                hit = TARGET in rep
                others = sorted(set(RE_DISC.findall(rep)) - set([TARGET]))
                srch = "KB_search" in rep
                tally[nm]["표적"] += 1 if hit else 0
                tally[nm]["타도구"] += len(others)
                tally[nm]["또검색"] += 1 if srch else 0
                tally[nm]["n"] += 1
                print("%-11s %-5s %-8s %-7d %-7s %s"
                      % (nm, tp, "O" if hit else "-", len(others), "O" if srch else "-", rep[:44]))
    print("")
    print("## 집계")
    for nm in ("A_asis", "B_restate", "N_len"):
        c = tally[nm]
        if not c["n"]:
            continue
        print("   %-11s 표적 %d/%d · 타도구 합 %d · 또검색 %d/%d"
              % (nm, c["표적"], c["n"], c["타도구"], c["또검색"], c["n"]))
    print("")
    print("[읽기] A_asis 가 이미 표적을 부르면 결손이 아니다([[62]] 2b) - 배선 근거가 없다.")
    print("[읽기] N_len 이 B 와 같으면 산 것은 내용이 아니라 길이다([[57]]).")
    print("[읽기] B 의 타도구 가 A 보다 크면 그것이 판 것이다([[70]] Delta-spurious).")
    print("[[76]] 자격: 배선은 B 가 **100%** 일 때만 한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
