#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""`T2_ACTIONREQ_GROUNDED` 래칫 — 스모크 없이 **초 단위로**. (P-A · `TASK_072.md` §7-2)

## 왜 (2026-08-26 · N1)

`formalize_intent_tool` 이 **이 대화에 한 번도 안 나온** 손님-측 도구를 지목하면 `[ACTION]` 이
*"'X' 는 손님이 실행한다"* 고 말한다. 참이지만 **이 대화와 무관**하고, 072 t0 에서 그 한 줄이
강제-행동 경로를 통째로 죽였다. 같은 site 를 `x505_TASK_073_t7348_perstep.md` §2.1 이 독립
지목했다(단발 아님). 문서는 처방 P-A 를 적고 상태를 **미착수**로 뒀다 — 이것이 그 구현이다.

빈도 실측(최근 12런·태그별): `formalized_target` 발화 **383건 중 29건(8%)** 이 궤적 축자 0회 ·
그중 **23건이 `submit_transaction`** · 태스크 040(8)·085(6)·074(5)·057(5)·063(4)·055(1)
⇒ **hard-0 여섯**에 걸친다(문서 추정 둘보다 넓다).

## 재료는 **실제 로그·궤적**이다

라이브 로그의 `[T2_ACTIONREQ] … formalized_target=X` 행과 그 sim 의 궤적을 짝지어, X 가 축자로
있는지 없는지를 **엔진과 같은 방식**으로 다시 센다. 합성 픽스처가 아니다.

⚠이 검정은 *"침묵이 점수를 사는가"* 를 판정하지 않는다 — 그건 런이 잰다([[62]]).

실행: PYTHONIOENCODING=utf-8 py -3 test_actionreq_grounded.py
"""
import gzip
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import t2_forensic as F                                          # noqa: E402

FAIL = []


def chk(c, m, extra=""):
    if not c:
        FAIL.append(m)
    print("  %s %s%s" % ("ok  " if c else "FAIL", m, ("  " + str(extra)) if extra else ""))


def grounded(blob, name):
    """엔진과 **같은 술어** — 축자 포함 여부 하나뿐(정규화·유사도 0)."""
    return str(name) in blob


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()

    print("① 배선 — 엔진 안에 있고, 플래그 뒤에 있고, 고르지 않는다")
    chk('os.environ.get("T2_ACTIONREQ_GROUNDED") == "1"' in src, "환경 플래그로 갈린다")
    i = src.find('os.environ.get("T2_ACTIONREQ_GROUNDED") == "1"')
    blk = src[i:i + 5200]
    chk("_utgt = None" in blk, "근거가 없으면 **지목을 비운다**(발화가 사라진다)")

    # 2026-08-29 - 침묵의 **자격**. 072 에서 옳은 침묵이 016 에서는 대화를 막다른 곳으로
    #   보냈다(두 런 다 인간 상담원 이관으로 종료 · `750` 발화 23·12 -> 0). 갈리는 것은
    #   그 시점에 **에이전트가 직접 할 수 있는 일이 남아 있느냐**이고, 코퍼스 전수
    #   (533 런·침묵 116 건)에서 016 만 38 건 발화로 바뀌고 072(38)·074(13) 는 한 건도 안 바뀐다.
    chk("_delivered_unused_agent_tools" in src, "자격 술어가 정본 헬퍼로 있다")
    chk("_pa_open" in blk, "침묵이 그 자격 뒤에 걸려 있다")
    chk("침묵 안 함" in blk, "자격 미달일 때 **왜 침묵 안 하는지** 로그가 남는다")
    j = src.find("def _delivered_unused_agent_tools")
    hblk = src[j:j + 1400] if j >= 0 else ""
    chk("_agent_discoverable" in hblk and "_unlocked_names" in hblk,
        "헬퍼는 레지스트리 ∩ 배달 − 호출/해제 뿐이다 (고르는 것 0)")
    chk("gold" not in hblk.lower(), "헬퍼에 gold 참조 0 ([[23]])")
    chk("chr(10).join(_seen_txt)" in blk, "축자 대조 하나로 판정한다")
    for bad in ("sort(", "max(", "argmax", "[0]"):
        chk(bad not in blk, "엔진이 고르지 않는다 — %r 없음" % bad)
    gs = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    chk("export T2_ACTIONREQ_GROUNDED=0" in gs, "정본 선언이 기본 OFF(효과는 런이 잰다)")

    print()
    print("② 술어 — 실제 라이브 발화에 걸어 본다")
    fs = [p for p in F.all_result_files() if p.endswith(".results.json.gz")]
    fs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    tot = ung = 0
    examples, tools = [], {}
    for p in fs[:12]:
        tag = F.tag_of_file(p)
        try:
            log = F.log_text(tag)
        except Exception:
            continue
        if not log:
            continue
        # 2026-08-29 - 손상/빈 gz 하나가 **배터리 전체를 붉게** 만들어 발사를 막았다.
        #   정지시킨 런의 회수 단계가 0 바이트 results 를 gz 로 남긴다(t7387 실물).
        #   그 파일은 이 검정의 대상이 아니므로 건너뛴다 - 조용히 넘기지 않고 찍는다.
        try:
            d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        except Exception as _e:
            print("  (건너뜀) 읽을 수 없는 결과 파일: %s %r" % (os.path.basename(p), _e))
            continue
        bysim = {}
        for s in (d.get("simulations") or d.get("results") or []):
            bysim["%s#s%s" % (s.get("task_id"), s.get("seed"))] = (
                str(s.get("task_id")),
                json.dumps(s.get("messages") or [], ensure_ascii=False))
        for m in re.finditer(r"\[sim=([^\]]+)\] \[T2_ACTIONREQ\].*?formalized_target=(\S+)", log):
            sim, tgt = m.group(1), m.group(2)
            if tgt in ("None", "-") or sim not in bysim:
                continue
            tot += 1
            task, blob = bysim[sim]
            if not grounded(blob, tgt):
                ung += 1
                tools[tgt] = tools.get(tgt, 0) + 1
                if len(examples) < 4:
                    examples.append((tag, sim, task, tgt))
    print("     발화 %d건 · 그중 궤적 축자 0회 **%d건**" % (tot, ung))
    for tag, sim, task, tgt in examples:
        print("        %-30s %-22s %s" % (tag[:30], sim[:22], tgt))
    chk(tot > 0, "라이브 발화를 실제로 찾았다(표본이 있다)", tot)
    chk(ung > 0, "근거 없는 지목이 **실재한다** — 이 레버가 겨냥하는 것", ung)
    chk("submit_transaction" in tools,
        "문서가 지목한 `submit_transaction` 이 표본에 있다", sorted(tools.items()))

    print()
    print("③ 침묵은 **근거 없는 것에만** — 있는 이름은 그대로")
    ok_grounded = 0
    for p in fs[:12]:
        tag = F.tag_of_file(p)
        try:
            log = F.log_text(tag)
        except Exception:
            continue
        if not log:
            continue
        # 2026-08-29 - 손상/빈 gz 하나가 **배터리 전체를 붉게** 만들어 발사를 막았다.
        #   정지시킨 런의 회수 단계가 0 바이트 results 를 gz 로 남긴다(t7387 실물).
        #   그 파일은 이 검정의 대상이 아니므로 건너뛴다 - 조용히 넘기지 않고 찍는다.
        try:
            d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
        except Exception as _e:
            print("  (건너뜀) 읽을 수 없는 결과 파일: %s %r" % (os.path.basename(p), _e))
            continue
        for s in (d.get("simulations") or d.get("results") or []):
            blob = json.dumps(s.get("messages") or [], ensure_ascii=False)
            key = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            for m in re.finditer(
                    r"\[sim=%s\] \[T2_ACTIONREQ\].*?formalized_target=(\S+)" % re.escape(key), log):
                t = m.group(1)
                if t not in ("None", "-") and grounded(blob, t):
                    ok_grounded += 1
        if ok_grounded:
            break
    chk(ok_grounded > 0,
        "궤적에 **있는** 지목도 표본에 많다 — 이 레버는 그것들을 건드리지 않는다", ok_grounded)

    print()
    print("RESULT: %s%s" % ("PASS" if not FAIL else "FAIL",
                            "" if not FAIL else "  " + str(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
