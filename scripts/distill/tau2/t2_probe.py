# -*- coding: utf-8 -*-
r"""t2_probe — **격리 프로브 정본**(2026-08-16·사용자 지시 *"기준을 기계적으로 만들라"*).

## 왜 (하루에 사본 3개·무효 통제 2개)

2026-08-16 하루에 같은 4셀 프로브를 **세 번** 새로 짰다(`x335`·`x335b`·`x336`). 그리고 그중
**둘은 부정통제가 무효**였다 — 뒤집을 수 없는 통제를 세워 놓고 판정할 뻔했다:

  · `x335 D_NEG`   손님 잔고를 $200k 로 올림 → 그래도 `Purple` 이 자격 충족(면제잔고는 낮을수록
                   유리) ⇒ 정답이 바뀌지 않는 통제
  · `x336 D_FLIP`  070 에서 고지 날짜를 맞바꿈 → `Lime Green` 은 **다른 축**($15,000 면제잔고)에서
                   이미 탈락 ⇒ 역시 바뀌지 않는 통제

둘 다 **돌리고 나서** 알았다. 이 모듈은 그 절차를 코드로 고정해 같은 실수를 반복하지 않게 한다.

## 무엇을 고정하나

  ① 사이트     (tag, task, cut) → 궤적 축자 본문(`x313.transcript` 정본 재사용)
  ② 셀         (label, 본문) 목록 — **A_REF 는 필수**(개입 없음 기준선)
  ③ 지표       이름 → 축자 부분문자열(대문자 비교). 판단 0·엔진 해석 0([[59]])
  ④ 표집       n = k×nb 블록(기본 8×3=24) · 첫 표본만 temp 0 · `max_tokens` **고정**
               (런마다 바꾸면 팔 간 비교가 깨진다 — 2026-08-15 x330~332 사고)
  ⑤ 판정       **사전 고정 문구를 결과보다 먼저 인쇄**한다. 잡음 바닥 ±4(C483) ⇒ **차 ≥5 만 인용**
  ⑥ 통제 검사  `control=(참조팔, 뒤집혀야 할 지표)` 를 선언한 팔은 실제로 갈렸는지 **기계가 본다**.
               안 갈렸으면 판정을 내지 않고 **`통제 무효 가능`** 으로 멈춘다 ← 오늘의 두 사고 자리

## 무엇을 안 하나 ([[62]] ③)

어느 답이 옳은지 모른다. 통제가 *의미상* 타당한지도 못 정한다(문서 전체를 읽는 판단이다).
할 수 있는 것은 **반증**뿐이다: 뒤집으라고 만든 팔이 한 번도 안 갈리면 그 통제는 못 쓴다.

## 쓰는 법

    import t2_probe as P
    site = P.site("bank_t7295_b_20260815n", "task_055", 14)
    P.run(name="x335c", site=site,
          arms=[("A_REF", ""), ("B_DOCS", material), ("D_FLIP", material_flipped)],
          marks={"PURPLE": "PURPLE", "BLUEST": "BLUEST"},
          verdict="B PURPLE ≥18 → 전달로 닫힘 · D 가 안 갈리면 통제 무효",
          control=("D_FLIP", "B_DOCS", "BLUEST"))
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from x216_read_and_offset import chat                             # noqa: E402
import t2_forensic as F                                           # noqa: E402
import x313_bailout_iso as B                                      # noqa: E402

__all__ = ["site", "run", "NOISE", "cite"]

NOISE = 4                 # 잡음 바닥 ±4 (C483 · `x320_baseline_variance`)
MAXTOK = 60               # ★고정. 바꾸려면 **모든 팔에서** 바꾸고 그 사실을 기록에 남겨라


def cite(a, b):
    """두 팔의 차이를 **인용해도 되는가**(차 ≥ NOISE+1). 규율을 눈이 아니라 코드가 본다."""
    return abs(a - b) > NOISE


def site(tag, task, cut, head=None):
    """궤적 축자 본문. 컷 안에 정보가 있는지는 **호출부가** 검산한다(정보-맞춤·§1.4)."""
    sim = next(s for s in F.scored(tag) if F.task_id(s) == task)
    body = "\n".join([head or B.HEAD, "", B.transcript(sim, cut)])
    return {"tag": tag, "task": task, "cut": cut, "sim": sim, "base": body}


def _count(msg_or_text, marks):
    """지표 적중. ★도구를 묶은 팔에서는 **방출된 tool_calls 까지** 본다.

    2026-08-16 자기 결함: `t2_gap` 의 `I4_TOOL` 단이 도구를 묶어 놓고 **본문의 이름만** 셌다.
    끝맺음(C489)은 *말했나* 가 아니라 *불렀나* 이므로 그 단은 무효였다. 여기서 닫는다.
    """
    if isinstance(msg_or_text, dict):
        blob = str(msg_or_text.get("content") or "")
        for tc in (msg_or_text.get("tool_calls") or []):
            f = tc.get("function") or {}
            blob += " " + str(f.get("name") or "") + " " + str(f.get("arguments") or "")
    else:
        blob = msg_or_text or ""
    t = blob.upper()
    return {k: (1 if v.upper() in t else 0) for k, v in marks.items()}


def emitted(msg):
    """도구 호출을 **실제로 방출했는가**(끝맺음 축의 지표)."""
    return 1 if (isinstance(msg, dict) and (msg.get("tool_calls") or [])) else 0


def run(name, site, arms, marks, verdict, ask, control=None, k=8, nb=3, maxtok=MAXTOK,
        tools=None, det=False):
    """4셀류 격리 프로브 한 번. `arms` = [(label, 덧붙일 재료(문자열))] · `ask` = 지시문.

    `control=(팔, 참조팔, 지표)` — 그 팔에서 지표가 참조팔과 **갈려야** 한다(부정통제).
    반환: {label: {지표: (합, [블록…])}} · 통제가 무너지면 `verdict` 를 인쇄하지 않는다.

    ★`det=True` — **온도 0 결정론 모드**(사용자 지시 2026-08-17: *"온도 0 이면 n=1 로 확정할 수
      있으면 그렇게 하라"*). 온도 0 에서 답이 고정이면 24 회 표집은 **같은 계산의 반복**이다.
      구현: 팔마다 **2 회**만 뽑되 둘 다 온도 0 — 두 답이 같으면 `n=1(결정론 확인)` 로 보고하고,
      다르면 결정론이 깨졌다고 인쇄하고 표집 모드로 판단을 미룬다(서버 배칭 등 비결정 요인).
      ⚠[[55]] 한계 병기: 온도 0 은 **최빈답**이고 적중이 표집 꼬리에서 오는 자리가 실제로 있었다
        (2026-08-14: 온도 0 이 8/8 오답·표집 40 회 최빈값도 그 오답·적중은 꼬리). 그러니 det 는
        *같은 입력이 같은 답을 준다*는 확인이지 *그 답이 옳다*는 보증이 아니다.
    """
    if det:
        k, nb = 2, 1
    if det:
        print("%s · %s/%s · cut=%d · **결정론 모드**(온도 0 ×2·같으면 n=1) · max_tokens=%d"
              % (name, site["tag"], site["task"], site["cut"], maxtok))
    else:
        print("%s · %s/%s · cut=%d · n=%d(%d×%d) · max_tokens=%d · 잡음바닥 ±%d"
              % (name, site["tag"], site["task"], site["cut"], k * nb, k, nb, maxtok, NOISE))
    print("판정(사전 고정·결과보다 먼저 인쇄): %s\n" % verdict)
    if not any(a[0] == "A_REF" for a in arms):
        print("⚠A_REF(개입 없음) 팔이 없다 — 기준선 없이는 귀속 못 한다. 중단.")
        return None

    res = {}
    for label, extra in arms:
        body = site["base"] + (("\n\n" + extra) if extra else "") + ask
        blocks = {m: [] for m in marks}
        for b in range(nb):
            acc = {m: 0 for m in marks}
            for i in range(k):
                try:
                    r = chat(body, tools, 0.0 if (det or i == 0) else 0.7, maxtok)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                out = " ".join(str(r.get("content") or "").split())
                hit = _count(r, marks)          # 본문 + 방출된 tool_calls 둘 다 본다
                for m in marks:
                    acc[m] += hit[m]
                print("    [%s b%d %02d] %s%s %s"
                      % (label, b + 1, i, ",".join(m for m in marks if hit[m]) or "-",
                         "|CALL" if emitted(r) else "", out[:60]),
                      flush=True)
            for m in marks:
                blocks[m].append(acc[m])
        res[label] = {m: (sum(blocks[m]), blocks[m]) for m in marks}
        if det:
            # 두 답이 같은지 = 결정론 확인. 지표 적중이 {0,2} 면 두 답이 일치한 것이다.
            _split = [m for m in marks if res[label][m][0] == 1]
            if _split:
                print("    ⚠결정론 깨짐(온도 0 두 답이 갈림·지표 %s) — n=1 확정 불가"
                      % ",".join(_split))
            else:
                print("    ✓결정론 확인(온도 0 ×2 동일) ⇒ **n=1**")
        print("%-11s %s\n" % (label, " · ".join("%s %d/%d %s" % (m, res[label][m][0], k * nb,
                                                                res[label][m][1]) for m in marks)))

    ok = True
    if control:
        arm, ref, mark = control
        a, rlab = res.get(arm, {}).get(mark, (0, []))[0], res.get(ref, {}).get(mark, (0, []))[0]
        # ★도달 가능성 먼저 본다 (2026-08-16 · 오늘의 세 판정을 기계적으로 가른 규칙).
        #   뒤집으라고 만든 지표가 **어느 팔에서도 한 번도 안 열렸다면** 그것은 이 자리에서
        #   *낼 수 없는 답*이라는 뜻이고 ⇒ 통제 자체가 설계 결함이다(뒤집힐 수가 없다).
        #   어디선가 열렸다면 그 답은 이 자리에서 **낼 수 있는 답**이므로, 통제 팔에서
        #   안 나온 것은 설계가 아니라 **모델이 그 축을 안 썼다**는 뜻이다.
        #   실측 대조: x335 BLUEST 전 팔 0 → 무효(맞다) · x336-070 LIME 전 팔 0 → 무효(맞다) ·
        #             x336-071 LIME 은 A_REF 24 → 도달 가능 ⇒ 유효, 차 0 = **날짜 미사용**.
        reach = max(v.get(mark, (0, []))[0] for v in res.values())
        if reach <= NOISE:
            ok = False
            print("⛔**통제 무효** — 지표 %s 가 **어느 팔에서도** 안 열렸다(최대 %d ≤ %d).\n"
                  "   이 자리에서 낼 수 없는 답을 뒤집으라고 세운 통제다 = 설계 결함.\n"
                  "   ⇒ **판정 보류.** 다른 축에서 탈락하지 않는 대안으로 다시 세워라." % (mark, reach, NOISE))
        elif not cite(a, rlab):
            print("✓통제 유효(도달 가능·최대 %d) — 그런데 `%s` %s=%d 가 참조 `%s` %d 와 안 갈렸다.\n"
                  "   낼 수 있는 답인데 안 냈다 ⇒ **모델이 그 축을 쓰지 않는다**(그 한 칸은 결정론 후보)."
                  % (reach, arm, mark, a, ref, rlab))
        else:
            print("✓통제 유효·갈렸다: `%s` %s=%d ↔ 참조 `%s` %d (차 %d > %d) ⇒ 모델이 그 축을 쓴다"
                  % (arm, mark, a, ref, rlab, abs(a - rlab), NOISE))
    if ok:
        print("\n측정치: " + " · ".join(
            "%s[%s]" % (lab, " ".join("%s=%d%s" % (m, v[m][0], v[m][1]) for m in marks))
            for lab, v in res.items()))
    return res
