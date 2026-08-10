# -*- coding: utf-8 -*-
r"""x222 — **098 의 정박원은 무엇인가** (격리 A/B · 유료 0 · 엔진 0).

## 왜 (C399·C400)

098 은 답이 문맥에 들어간 채로 진다. 그리고 x219 에서 **블록을 맨 끝에 세워도(B_LAST)
KB 를 통째로 지워도(C_NOKB) 0/8** 이었다 — 위치도 경쟁 스니펫도 범인이 아니다.
C400 은 *"블록이 뒤로 밀린다"* 도 기각했다. 남은 후보는 **문맥 자체의 정박**이다.

궤적 정독(`bank_anchorslot_20260809s14` 098 t0)이 후보를 하나 준다: 그 대화에서 **에이전트
자신이 먼저 `EcoCard` 를 권고**했고(assistant 4턴), **손님이 그것을 되물으며**(user 3턴)
대화가 그 이름 위에서 굳었다. 계좌 기록에는 카드가 **없다**(`No credit card accounts found`)
— 즉 정박은 DB 행이 아니라 **자기 발화 + 그에 얽힌 손님 발화**다(C124 자기-정박과 동형).

## 팔 (블록은 F_NULL 을 뺀 **모든 팔에 그대로** 실린다 · 정보-맞춤 [[18]])

  A_CTRL    문맥 그대로 + 블록                      ← 라이브 재현(0/8 이어야 한다)
  B_NOSELF  **에이전트 자신의** 오답어 문장만 제거
  C_NOUSER  **손님의** 오답어 문장만 제거
  D_BOTH    B + C
  E_NODOC   KB 스니펫 중 **제목에 오답어가 든 항목**만 제거
  H_ALL3    B + C + E (오답 정박 전부 제거)
  I_NOERR   **오류·거부 메시지 전부 제거** + 블록    ← 사용자 질문 ⑴
  J_CALLVAL **호출 표기 + 성공 반환값 + 대화**만 (오류·내부 잡음 제거) ← 사용자 질문 ⑵
  K_USERSUB **손님 메시지 + 서브에이전트 블록**만    ← 사용자 질문 ⑶ (사용자가 제안한 설계)
  F_NULL    문맥 그대로, **블록 없음**              ← 부정 통제
  G_ONLY    **블록만**                              ← 천장·계기 검사

## 사용자 질문 셋 (2026-08-10) — *"값과 질문 사이의 로그·에러를 걷어내면 되지 않나"*

⑴ *"쓸데없는 로그·에러가 많아서 채택이 안 된 것 아닌가"* ⑵ *"함수 호출과 반환값, 질문과
답변만 남기면 되지 내부 함수나 에러를 문맥에 둘 필요가 있나"* ⑶ *"메인이 답할 때 손님
메시지와 서브에이전트 메시지만 추려 격리해서 답하면 되지 않나"*.

부분적으로 **이미 답이 있다**: 이 프로브의 문맥은 영속된 **대화 전사**라 우리가 라이브에서
끼워 넣는 리마인더·거부는 **애초에 들어 있지 않고**, `A_CTRL`·x219 `B_LAST` 는 블록과 질문
사이에 **아무것도 없는** 배치다 — 그런데도 098 은 0/8 이었다. x220 도 블록과 한 메시지에
동거하던 `Error:` 지시를 떼어 봤지만 098 은 1/8(잡음)에 그쳤다. ⇒ *"사이의 잡음"* 만으로는
098 이 설명되지 않는다. 그래도 **전사에 남은 오류·잡음까지 싹 걷어낸 형태**(`I`·`J`)와
**사용자가 제안한 격리 형태**(`K`)는 안 쟀다. 여기서 잰다.
⚠`K` 는 천장(`G_ONLY`)과 달리 **손님 발화를 그대로 담는다** — 098 은 손님이 `EcoCard` 를
세 번 되묻는 대화라, 정박이 손님 쪽에 있으면 `K` 도 진다. 그게 이 팔의 값어치다.
⚠전면 제거는 게이트 거부를 모델이 못 보게 만들어 **게이트를 죽인다**(HANDOFF §8-2)
— 여기서는 **잰다**, 켜지 않는다.

⚠**이것은 진단이지 레버가 아니다.** 절제 기준인 '오답어'는 그 시행이 **실제로 낸 답**을 보고
정한다 — 라이브에서는 그것을 모른다. 여기서 정박이 범인으로 확정되면, 그때 *"오답을 모르고도
쓸 수 있는 형태"* 를 따로 설계해야 한다. gold 는 채점에만 쓴다([[23]]).

⚠팔마다 무엇이 담겼는지(블록·KB 수·오답어 수·원장 행 수·자수) **먼저 인쇄한다** — 정보를
잃은 팔이 있으면 그 사례는 재지 않는다(C395′ 교훈).

채점 = 정확 일치(`Blue` 가 `Light Blue` 에 걸린다).

실행: python x222_anchor_source.py [N]
"""
import collections
import copy
import glob
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                            # noqa: E402
from x219_adoption import ASK, BLOCK_SIG, render                  # noqa: E402
from x219_adoption import live_blocks as live_blocks_remote       # noqa: E402

FB_GZ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../../../reports/facet_rft_2026/sim_results/"
                     "fb_bank_alllevers_20260810.jsonl.gz")


def live_blocks():
    """라이브에서 실제로 나간 결정 블록. 리모트 원본이 없으면 repo 영속본에서 읽는다."""
    out = live_blocks_remote()
    if out or not os.path.exists(FB_GZ):
        return out
    for ln in gzip.open(FB_GZ, "rt", encoding="utf-8", errors="replace"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        t = o.get("text") or ""
        if BLOCK_SIG not in t:
            continue
        m = re.search(r"It answers: ([^.\n]+)", t)
        if m:
            out.setdefault(m.group(1).strip(), t.strip())
    return out

PATS = ["/home/woori/scratch/tau2-bench/data/simulations/*/results.json",
        "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/"
        "sim_results/*.json.gz",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../../../reports/facet_rft_2026/sim_results/*.json.gz")]

# 사례 = (결과파일 태그, 태스크, 시행, gold, **그 시행이 실제로 낸 오답**)
# 오답어는 관측이다 — 아래 main() 이 그 시행의 마지막 에이전트 발화를 함께 인쇄하므로
# 읽는 사람이 직접 대조할 수 있다.
CASES = [
    ("bank_anchorslot_20260809s14", "task_098", 0, "Blue", "EcoCard"),
    ("bank_alllevers_20260810", "task_098", 2, "Blue", "Light Blue"),
]

SENT = re.compile(r"(?<=[.!?])\s+")


def load_case(tag, task, trial):
    for pat in PATS:
        for p in sorted(glob.glob(pat)):
            if tag not in os.path.basename(p):
                continue
            try:
                f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") \
                    else open(p, encoding="utf-8")
                d = json.load(f)
            except Exception:
                continue
            for s in (d.get("simulations") or []):
                if s.get("task_id") == task and s.get("trial") == trial:
                    return s
    return None


def drop_sentences(msgs, roles, word):
    """지정한 역할의 메시지에서 오답어가 든 **문장만** 뺀다 (메시지 자체는 남긴다)."""
    out = copy.deepcopy(msgs)
    for m in out:
        if m.get("role") not in roles:
            continue
        c = str(m.get("content") or "")
        if word.lower() not in c.lower():
            continue
        keep = [s for s in SENT.split(c) if word.lower() not in s.lower()]
        m["content"] = " ".join(keep)
    return out


def drop_snippets(msgs, word):
    """KB 결과에서 **제목에 오답어가 든 번호 항목**만 뺀다."""
    out = copy.deepcopy(msgs)
    for m in out:
        if m.get("role") != "tool":
            continue
        c = str(m.get("content") or "")
        if "Score:" not in c or word.lower() not in c.lower():
            continue
        items = re.split(r"(?m)^(?=\s*\d+\.\s)", c)
        keep = [it for it in items
                if not (re.match(r"\s*\d+\.\s", it) and word.lower() in it.split("\n")[0].lower())]
        m["content"] = "".join(keep)
    return out


# 오류·거부로 관측되는 반환값의 서명 (전사 정독으로 확인한 것만 — 지어내지 않는다)
ERR_SIG = ("NOT_VERIFIED", "blocked by policy gate", "Error:", "cannot be carried out",
           "is run by the CUSTOMER", "not permitted", "failed")


def is_err(m):
    if m.get("role") != "tool":
        return False
    c = str(m.get("content") or "")
    return any(sig.lower() in c[:200].lower() for sig in ERR_SIG)


def drop_errors(msgs):
    """오류·거부 반환값 메시지를 통째로 뺀다."""
    return [copy.deepcopy(m) for m in msgs if not is_err(m)]


# 대화의 내용이 아니라 **절차 부기**로 오간 반환값 (전사 정독으로 확인한 것만)
INTERNAL_SIG = ("Tool unlocked:", "VERIFIED —", "Verification logged", "The current time is")


def drop_internal(msgs):
    """호출 표기와 **데이터 반환값**은 남기고 절차 부기만 뺀다."""
    out = []
    for m in msgs:
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and any(c.strip().startswith(s) for s in INTERNAL_SIG):
            continue
        out.append(copy.deepcopy(m))
    return out


def live_injections(gold):
    """그 답을 낸 사이드카 sim 들의 **주입 전부**를 turn 별로 모은다 (라이브 근사).

    ⚠사이드카 `sim` 은 해시라 전사와 시행별로 대응되지 않는다(HANDOFF §10). 그래서 이것은
    *복원*이 아니라 **같은 태스크에서 실제로 나갔던 주입을 같은 turn 위치에 얹은 근사**다.
    모델이 메시지로 보는 것만 담는다 — `reminder-user`·`tool-deny` (버려진 초안 `reminder-
    assistant` 는 재생성으로 대체되므로 제외).
    """
    path = FB_GZ if os.path.exists(FB_GZ) else None
    if not path:
        return {}
    rows = [json.loads(l) for l in gzip.open(path, "rt", encoding="utf-8") if l.strip()]
    mine = set()
    for r in rows:
        m = re.search(r"It answers: ([^.\n]+)", r.get("text") or "")
        if m and m.group(1).strip().rstrip(".") == gold:
            mine.add(r["sim"])
    by = collections.defaultdict(list)
    for r in rows:
        if r["sim"] in mine and r["kind"] in ("reminder-user", "tool-deny"):
            if BLOCK_SIG in r["text"]:      # 블록 자체는 팔에서 따로 붙인다
                continue
            by[r["turn"]].append(r["text"])
    return by


def render_live(msgs, by_turn):
    """전사에 주입을 turn 위치대로 끼워 넣는다."""
    parts = []
    for i, m in enumerate(msgs):
        t = m.get("turn_idx", i)
        for txt in by_turn.get(t, []):
            parts.append("[system] %s" % " ".join(txt.split()))
        parts.append(render([m]))
    return "\n".join(p for p in parts if p)


def only_user_and_block(msgs):
    """손님 발화만 남긴다 (서브에이전트 블록은 호출부에서 붙인다)."""
    out = []
    for m in msgs:
        if m.get("role") == "user" and str(m.get("content") or "").strip():
            out.append({"role": "user", "content": m.get("content")})
    return out


def audit(name, body, word):
    return ("   %-9s 블록 %s · KB %2d · 오답어 %2d · 원장행 %2d · %6d자"
            % (name, "O" if BLOCK_SIG in body else "X", body.count("Score:"),
               len(re.findall(re.escape(word), body, re.I)),
               body.count("Record ID:"), len(body)))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    blocks = live_blocks()
    print("라이브 결정 블록 %d종: %s" % (len(blocks), sorted(blocks)))
    out = {}
    for tag, task, trial, gold, wrong in CASES:
        s = load_case(tag, task, trial)
        if not s:
            print("\n%s/%s t%s — 결과 파일을 못 찾았다. 건너뛴다." % (tag, task, trial))
            continue
        blk = blocks.get(gold)
        if not blk:
            print("\n%s — 라이브 블록 문구(%r)를 못 찾았다. 건너뛴다." % (task, gold))
            continue
        msgs = s.get("messages") or []
        last = [m for m in msgs
                if m.get("role") == "assistant" and str(m.get("content") or "").strip()][-1]
        print("\n" + "=" * 96)
        print("%s  %s t%s · reward=%s · gold=%r · 오답어=%r · n=%d"
              % (task, tag, trial, (s.get("reward_info") or {}).get("reward"), gold, wrong, n))
        print("  [그 시행의 마지막 에이전트 발화] %s"
              % " ".join(str(last.get("content")).split())[:220])
        print("  [블록에 오답어가 들었나] %s"
              % ("들었다 — 차순위 목록" if wrong.lower() in blk.lower() else "아니다"))

        full = render(msgs)
        arms = [
            ("A_CTRL", full + "\n\n" + blk),
            ("B_NOSELF", render(drop_sentences(msgs, ("assistant",), wrong)) + "\n\n" + blk),
            ("C_NOUSER", render(drop_sentences(msgs, ("user",), wrong)) + "\n\n" + blk),
            ("D_BOTH", render(drop_sentences(msgs, ("assistant", "user"), wrong)) + "\n\n" + blk),
            ("E_NODOC", render(drop_snippets(msgs, wrong)) + "\n\n" + blk),
            ("H_ALL3", render(drop_snippets(drop_sentences(msgs, ("assistant", "user"), wrong),
                                            wrong)) + "\n\n" + blk),
            ("I_NOERR", render(drop_errors(msgs)) + "\n\n" + blk),
            ("J_CALLVAL", render(drop_internal(drop_errors(msgs))) + "\n\n" + blk),
            ("K_USERSUB", render(only_user_and_block(msgs)) + "\n\n" + blk),
            ("L_LIVEISH", render_live(msgs, live_injections(gold)) + "\n\n" + blk),
            ("F_NULL", full),
            ("G_ONLY", blk),
        ]
        base_rows = full.count("Record ID:")
        # 원장을 **설계상** 안 담는 팔은 면제한다 (담아야 하는데 잃은 팔만 잡는다)
        exempt = ("G_ONLY", "K_USERSUB")
        skip = False
        for name, body in arms:
            print(audit(name, body, wrong))
            if name not in exempt and body.count("Record ID:") < base_rows:
                print("      ⚠원장 행이 줄었다(%d < %d) — 정보를 잃은 팔이다."
                      % (body.count("Record ID:"), base_rows))
                skip = True
        if skip:
            print("  ⇒ 이 사례는 재지 않는다(C395′ 규칙).")
            continue

        for name, body in arms:
            c = collections.Counter()
            for i in range(n):
                p = body + "\n\n" + ASK
                try:
                    t = chat(p, None, 0.0 if i == 0 else 0.7, 24).get("content", "")
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                c[" ".join(str(t).split())[:40]] += 1
            hit = sum(v for k, v in c.items()
                      if re.fullmatch(r"\**%s( Account)?\**\.?" % re.escape(gold),
                                      str(k).strip(), re.I))
            out["%s/%s/t%s/%s" % (tag, task, trial, name)] = [hit, n]
            print("  %-9s gold %d/%d   %s" % (name, hit, n, c.most_common(2)))

    json.dump(out, open(os.environ.get("T2_X222_OUT", "x222_out.json"), "w"), indent=1)
    print("\n※ 읽는 법 — A_CTRL 은 0/8 이어야 하고(재현), F_NULL 도 0/8(부정 통제),"
          "\n  G_ONLY 는 8/8 이어야 한다(계기 검사)."
          "\n  ⒜ 정박원: B 나 C 가 살리면 **대화의 자기 발화**, E 가 살리면 **문서**,"
          "\n     H 조차 못 살리면 정박 제거로는 안 되는 것이다."
          "\n  ⒝ 사용자 질문: I 가 살리면 **오류 잡음**, J 가 살리면 **내부 부기까지**,"
          "\n     K 가 살리면 *'손님 말 + 서브 답만 추려 격리'* 라는 설계가 성립한다."
          "\n     K 가 지고 G_ONLY 만 살면 정박은 **손님 발화 자체**에 있다(설계 수정 필요).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
