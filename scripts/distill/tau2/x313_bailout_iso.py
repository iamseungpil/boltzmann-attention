# -*- coding: utf-8 -*-
r"""x313 — 인간-이관 **이탈**의 결손 가름: 남은 일을 모르는가 · 알면서 넘기는가.

관측(2026-08-14 야간·독립 표본 둘·`bank_bailout_audit`):
  t7286 G대표 12  gold 에 없는 이관으로 종료 **5/12** — 버린 gold 액션 합 52(중앙 6·최대 24)
  t7290 fee 4     통과 2건 이관 0 ↔ 실패 2건(072·074) **둘 다 이탈** — 버린 gold 14
  이관 자체는 결함이 아니다(081 은 이관이 gold) — **gold 가 요구하지 않은** 이관만 결함이다.

⚠[[62]] — 여기서 레버로 직행하지 않는다. 처방 후보(예: 이탈 억제 문구)는 [[63]]/x301 B_WARN 0/8
전례상 **맞히던 것까지 파괴**할 수 있다. 먼저 **결손의 종류**를 격리로 재고, 격리에서 되면
레버는 전달(부하 축소)뿐이다.

⚠[[55]] 우리 배관 먼저 — `TRANSFER NOTICE` 는 우리 A2 문구지만 게이트 `ask` 라 **모델이 이관을
시도한 뒤에만** 발화한다(개시자 아님). 다만 그 문구는 "지금 이 메시지를 보내고 그 다음
`transfer_to_human_agents` 를 호출하라" 로 **완주를 몰아붙인다** — 증폭 후보라서 D 팔로 잰다.

컷 = **이탈 직전**(그 sim 이 처음 이관을 부른 호출 바로 앞). 셀 5 (n=8·사이트 고정·단일변수):
  A_REF     라이브 문맥 축자 · 평소 계약        ← 이탈 재현 기준선(이관/기권이면 재현)
  B_KNOW    같은 컷 · **도구 금지·텍스트로** "이 손님 요청 중 아직 안 끝난 일은 무엇인가"
                                                ← knowing 측정(x307 C_ASK 동형)
  C_LIST    같은 컷 · B 가 낼 만한 **남은 일 목록을 우리가 동봉** · 평소 계약
                                                ← 전달만 했을 때 착수하는가([[62]] ② 전달 레버)
  D_NONOTE  같은 컷 · 문맥에서 **우리 TRANSFER NOTICE 문구를 제거**
                                                ← 우리 문구가 이탈을 증폭하는가([[55]])
  E_NOBASIS 같은 컷 · **도구 결과 근거 제거**    ← 날조 부정통제([[57]])

판정(사전 고정):
  B_KNOW ≥6 ∧ A_REF 이탈    → **knowing-doing** = 남은 일을 알면서 넘긴다 ⇒ 레버는 배치/전달
  B_KNOW ≤2                 → 남은 일을 **모른다** ⇒ 지식·재료 축(coverage/discovery 상류)
  C_LIST ≥6 ∧ A_REF ≤2      → **전달만으로 열린다**(부하 축소 레버 확정·새 판단 0)
  D_NONOTE ≥6 ∧ A_REF ≤2    → **우리 문구가 인자** ⇒ 우리 층 수리가 먼저([[55]])
  E_NOBASIS ≥2              → 날조 = 이 판 무효

계기: 이탈=이관 도구 호출 ∨ 이관 예고 산문 · 착수=**gold 아닌** 실행 도구 호출(이름은 판정에
쓰지 않고 **이관인가 아닌가**만 본다 — 무엇이 정답인지는 여기서 말하지 않는다).

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x313_bailout_iso.py [N]
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
import t2_forensic as F                                           # noqa: E402

# 이탈 사이트(관측에서 나온 둘 — 서로 다른 런·다른 태스크로 사이트 다양성 확보)
SITES = (("bank_t7290_a_20260814m", "task_074"),
         ("bank_t7286_a_20260814h", "task_048"))

HEAD = ("You are the banking agent in an ongoing conversation. Below is the transcript so far. "
        "Decide what to do next.")
ASK_TEXT = ("\n[instruction] Do NOT call any tool now. In text only, list every part of this "
            "customer's request that is still not finished. If everything is finished, say so.")
NOTE_MARK = "TRANSFER NOTICE"
TRANSFER_WORDS = ("transfer_to_human_agents", "request_human_agent_transfer",
                  "human agent", "transferred to a human")


def transfer_cut(sim):
    """이탈 직전 = 이관을 처음 부른 호출을 담은 메시지의 **인덱스**."""
    for i, m in enumerate(sim.get("messages") or []):
        for tc in (m.get("tool_calls") or []):
            nm = F.label(F.nameof(tc), F.argsof(tc)).split(":")[-1]
            if nm in F.TRANSFER or "transfer_to_human" in nm or "human_agent_transfer" in nm:
                return i
    return None


def transcript(sim, cut, drop_note=False, drop_tools=False):
    out = []
    for m in (sim.get("messages") or [])[:cut]:
        role = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        if role == "tool":
            if drop_tools or m.get("error") or not c:
                continue
            out.append("[tool] " + c[:1200])
        elif role in ("user", "assistant") and c:
            if drop_note and NOTE_MARK in c:
                continue
            out.append("[%s] %s" % (role, c[:900]))
    return "\n".join(out[-60:])


def bailed(r):
    t = str(r.get("content") or "").lower()
    return any(w.lower() in t for w in TRANSFER_WORDS)


def acted(r):
    """이관이 아닌 **실행 의사**: 도구 호출 형태가 있고 그것이 이관이 아니다."""
    t = str(r.get("content") or "")
    if bailed(r):
        return False
    return bool(re.search(r'"?(tool|name|function)"?\s*[:=]\s*"?[a-z_]{6,}', t)
                or re.search(r"\b[a-z_]+_\d{4}\b", t))


def named_unfinished(r):
    """남은 일을 **하나라도 이름으로** 댔는가(내용의 옳고 그름은 판정하지 않는다)."""
    t = str(r.get("content") or "")
    if re.search(r"(?i)everything (is )?(finished|done|complete)", t):
        return False
    return len(re.findall(r"(?m)^\s*[-*\d].{12,}", t)) >= 1 or len(t.split()) >= 25


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    for tag, task in SITES:
        try:
            sim = next(s for s in F.scored(tag) if F.task_id(s) == task)
        except StopIteration:
            print("%s / %s 없음 — 건너뜀" % (tag, task))
            continue
        cut = transfer_cut(sim)
        if cut is None:
            print("%s / %s 이탈 컷 없음 — 건너뜀" % (tag, task))
            continue
        base = transcript(sim, cut)
        arms = (
            ("A_REF", "\n".join([HEAD, "", base])),
            ("B_KNOW", "\n".join([HEAD, "", base, ASK_TEXT])),
            ("C_LIST", None),               # B 결과를 받아 아래에서 조립
            ("D_NONOTE", "\n".join([HEAD, "", transcript(sim, cut, drop_note=True)])),
            ("E_NOBASIS", "\n".join([HEAD, "", transcript(sim, cut, drop_tools=True)])),
        )
        print("\n%s\n# %s / %s · cut=%d · 본문 %d자 · n=%d\n%s"
              % ("=" * 88, tag, task, cut, len(base), n, "=" * 88))
        knows = []
        res = {}
        for label, body in arms:
            if label == "C_LIST":
                if not knows:
                    print("%-10s 건너뜀(B_KNOW 산출 없음)" % label)
                    continue
                body = "\n".join([HEAD, "", base, "",
                                  "[note] These parts of the request are still not finished:",
                                  knows[0][:1200]])
            k = 0
            cnt = collections.Counter()
            for i in range(n):
                try:
                    r = chat(body, None, 0.0 if i == 0 else 0.7, 1200)
                except Exception as e:
                    r = {"content": "ERR %s" % type(e).__name__}
                if label == "B_KNOW":
                    ok = named_unfinished(r)
                    if ok and len(knows) < 1:
                        knows.append(" ".join(str(r.get("content") or "").split()))
                    cnt["named" if ok else "none"] += 1
                else:
                    ok = acted(r)
                    cnt["act" if ok else ("bail" if bailed(r) else "other")] += 1
                k += ok
                print("    [%s %02d] %s" % (label, i, "HIT" if ok else "-"), flush=True)
            res[label] = k
            print("%-10s %d/%d · %s\n" % (label, k, n, dict(cnt)))
        print("판정(사전 고정): B_KNOW≥6∧A_REF 이탈 → knowing-doing · B_KNOW≤2 → 지식/재료 축 · "
              "C_LIST≥6∧A_REF≤2 → 전달로 열림 · D_NONOTE≥6∧A_REF≤2 → 우리 문구가 인자 · "
              "E_NOBASIS≥2 → 무효")
        print("측정치: " + " · ".join("%s=%d" % (k, v) for k, v in res.items()))


if __name__ == "__main__":
    main()
