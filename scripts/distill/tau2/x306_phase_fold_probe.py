# -*- coding: utf-8 -*-
r"""x306 — 국면(접힘-누적) 격리: 같은 문면이 국면에 따라 죽는가, 죽는다면 무엇이 쌓여서인가.

동기(사용자 질문 2026-08-14: *"태스크마다 계속 고칠 게 나올까? 일반화가 안 되는 건가"*):
기전은 수렴하는데(C470: NOTCALLED 105·DENIED 0) **같은 처방이 국면에 따라 무효**가 된다 —
  x304 B_STEP2(087 컷34)          **6/8**   ← 이른/중간 국면
  x299 (075·접힘 2회 누적 후)     **0/8**   ← 같은 문면·늦은 국면 ([[64]] C413)
  x302b(073 컷59·계기 수리판)     **0/8**   ← 옳은 이름 3종을 줘도 (B_KIND·mx=1500 재확인)
  관측: 실패 sim 은 deny·regen 누적이 크다(073 137/105 ↔ 통과 075 10/11·C470).
국면이 **태스크 수만큼 늘지 않는 축 하나**라면, 처방은 태스크별 수리가 아니라 국면-회피
배치로 수렴한다. 그 갈림을 이 프로브가 잰다.

사이트 = 073 credit 착수(x302 와 동일 궤적·동일 컷·동일 문면 — 변수는 **문맥 국면뿐**·[[03b]]):
  P_LATE   (재사용·발사 없음) x302b B_KIND = 라이브 전체 문맥(컷59·우리 주입 포함) + note → **0/8**
  P_EARLY  정보-맞춘 최소 문맥(신원 확인·fee 도구 산출 3건·손님의 정정 요구 — 전부 라이브
           축자) + **같은 note** → 이른 국면의 같은 결정
  C_MASS   라이브 전체 문맥(컷59)에서 **우리 주입 줄만 제거**(ours={}) + 같은 note →
           누적의 정체 가름: 우리 텍스트 질량인가, 모델 자신 궤적인가 ([[57]] 인자-변화 통제)

계기(오늘 5결함 반영): mx=1500 · function.name 판독 · 전건 라벨 · 컷 앞뒤 실물 인쇄.

판정(사전 고정·n=8):
  P_EARLY ≥6 ∧ P_LATE ≤2 → **국면 확정**(누적이 인자) → C_MASS 가 처방 축을 가른다:
      C_MASS ≥6 → 쌓인 것 = **우리 주입 텍스트** → 처방 = 전달-후-제거/비가시 채널(측정 후 설계)
      C_MASS ≤2 → 쌓인 것 = 모델 자신 궤적(자기 접힘 산문·길이) → 처방 = 배치([[65]] 서브·FIX-10 동형)
  P_EARLY ≤2 → 국면 아님(이 사이트 문면 자체 무효) → 사이트 재선정·U1 잔여 재수사
  중간(3~5) → n=16 재측정 1회 후 판정.

실행(리모트·8141·체인 뒤): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x306_phase_fold_probe.py [N]
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x283_discovery_reach_probe as P                            # noqa: E402
import x291_checking_pick_iso as B                                # noqa: E402
import x302_kindmap_ownership_probe as K                          # noqa: E402

TARGET = K.TARGET                        # apply_checking_account_credit_5829


def classify(r):
    blob = " ".join(str(t) for t in (r.get("tool_calls") or []))
    if TARGET in blob:
        return "target"
    for t in (r.get("tool_calls") or []):
        n = str((t.get("function") or {}).get("name") or t.get("name") or "")
        if n:
            return n[:36]
    return "(text)" if r.get("content") else "(empty)"


def cut_of(sim):
    """x302 와 동일 규칙 축자 — 마지막 fee 도구 결과(net 산출) 직후."""
    cut = None
    for i, m in enumerate(sim["messages"]):
        if m.get("role") == "tool" and "net correction" in str(m.get("content") or ""):
            cut = i + 1
    return cut


def early_ctx(sim, cut):
    """정보-맞춘 최소 문맥 — 전 줄 라이브 축자([[03b]])·요약 0.

    포함: ①첫 손님 요구 ②신원 확인 tool 줄 ③fee 도구 산출(계좌별·net correction 포함 전부)
    ④컷 직전 마지막 손님 발화. 이 결정에 필요한 사실은 전부 있고, 접힘 이력만 없다.
    """
    msgs = sim["messages"][:cut]
    rows = ["[user] " + " ".join(str(msgs[1].get("content") or "").split())[:600]]
    for m in msgs:
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and "Verification logged successfully" in c:
            rows.append("[tool] " + " ".join(c.split())[:300])
            break
    for m in msgs:
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and ("net correction" in c or "does NOT match" in c):
            rows.append("[tool] " + " ".join(c.split())[:1200])
    last_user = next(" ".join(str(m.get("content")).split()) for m in reversed(msgs)
                     if m.get("role") == "user")
    rows.append("[user] " + last_user[:600])
    return "\n".join(rows)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = next(s for s in X.load(K.TAG) if s["task_id"] == K.TASK
               and s.get("reward_info") is not None)
    cut = cut_of(sim)
    if cut is None:
        print("컷 없음")
        return
    names = K.kind_matches()
    note = K.NOTE_T % ", ".join(names)          # x302 B_KIND 문면 축자 — 변수는 문맥뿐
    tools = U.tools_of(sim)
    P.TAG = K.TAG
    ours = P.our_lines(sim)
    full = B.render(sim["messages"][:cut], ours)
    full = full[:full.rfind("\n[user] ")] if "\n[user] " in full else full
    mass = B.render(sim["messages"][:cut], {})
    mass = mass[:mass.rfind("\n[user] ")] if "\n[user] " in mass else mass
    early = early_ctx(sim, cut)
    print("x306 cut=%d · target=%s · note=%d자 · P_EARLY %d자 · C_MASS %d자 · FULL %d자 · n=%d" % (
        cut, TARGET, len(note), len(early), len(mass), len(full), n))
    print("P_LATE 는 x302b B_KIND(같은 note·FULL 문맥) = 0/8 재사용 — 발사 없음\n")
    for label, body in (("P_EARLY", early), ("C_MASS", mass)):
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body + "\n[system] " + note, tools, 0.0 if i == 0 else 0.7, 1500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            k = classify(r)
            hit += k == "target"
            cnt[k] += 1
            print("  [%s %02d] %s" % (label, i, k), flush=True)
        print("%-8s target %d/%d · %s\n" % (label, hit, n, dict(cnt)))
    print("※ 판정(사전 고정): P_EARLY ≥6 ∧ P_LATE ≤2 → 국면 확정 → C_MASS ≥6 = 우리 텍스트 질량"
          " / C_MASS ≤2 = 모델 자신 궤적. P_EARLY ≤2 → 국면 아님(사이트 재선정). 중간 → n=16.")


if __name__ == "__main__":
    main()
