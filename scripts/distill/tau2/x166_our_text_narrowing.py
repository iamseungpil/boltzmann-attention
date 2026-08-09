# -*- coding: utf-8 -*-
r"""x166 — **우리 개입 문구가 그 자체로 정의역을 좁히는가** (§6.3 를 우리 층에 적용·유료 0).

## 왜 (2026-08-09·C348⒢ ↔ §6.3 충돌 해소)

§6.3 은 방아쇠가 **계좌 이름 지목**이고 그것이 $Y$ 를 가족으로 좁힌다는 것을 보였다(지목을
바꾸면 그 가족 최대가 나온다). 그리고 방아쇠 메시지 #26 은 **에이전트 자신의 문장**이다.
한편 레지스트리 대조 결과 우리 `claimprov` 의 *"실행되지 않았다"* 는 **참**이었다
(`submit_referral` = `@is_tool(ToolType.WRITE)` = 에이전트 도구) ⇒ *"거짓 문구 제거"* 는
표적이 없다. 남는 가설은 **명령**이다 — `feedback_pending` 은 *"Do the promised work NOW by
calling the real tools"* 라고 밀고, 그 행동 프레임이 에이전트로 하여금 **계좌를 지목하게** 한다.

그러면 우리 문구는 두 경로 중 하나로 작동한다:
  ⒜ **직접**: 우리 문장이 문맥에 들어가는 것만으로 $Y$ 가 좁아진다 → 여기서 잡힌다
  ⒝ **간접**: 우리 문장은 무해하고, 에이전트의 **다음 문장**이 좁힌다 → 여기서는 안 잡히고
     라이브 루프가 필요하다
이 프로브는 ⒜ 를 **먼저** 배제/확인한다. 값싸고, 결과가 어느 쪽이든 다음 실험을 정한다.

## arm (전부 msgs[:26] = 아직 안 무너진 문맥 위에 한 문장만 얹는다)

  C_none      아무것도 안 얹음                     ← 기준선(무너지기 전·p(gold)≈0.995)
  A_imper     현행 `feedback_pending` (명령형)
  B_owner     신규 `feedback_ownership` (표면화만)  ← 어제 배포한 수리
  D_named     #26 원문(계좌 지목 포함)              ← 양성 통제(§6.3 이 이미 무너뜨린 문장)

**사전 예측**(먼저 적는다·[[08]]): A·B 는 계좌 이름을 담지 않으므로 §6.3 대로면 **둘 다
p(gold) 유지**. D 만 무너진다. 만약 A 가 무너지면 우리 문구가 **직접** 정의역을 좁히는 것이고,
그때는 §6.3 이 "이름 지목"만으로는 부족하다는 뜻이라 기전 서술을 넓혀야 한다.
만약 A 와 B 가 **똑같이** 무너지면 어제의 수리는 이 축에서 **효과 0**이다 — 그렇게 적는다.

실행: T2_PROBE_URL=... T2_PROBE_MODEL=... py -3 x166_our_text_narrowing.py [N]
"""
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402
import x157_entrainment_lambda as P                            # noqa: E402
import t2_factdag as FD                                        # noqa: E402
import t2_ledger as LG                                         # noqa: E402
from gate_interpreter import load_domain_a2                     # noqa: E402

DOMAIN = "banking_knowledge"
# 실제 라이브 발화에서 관측된 미이행 약속 문구(런 m 사이드카 축자). 도메인 어휘는 여기서
# 만들지 않는다 — 사이드카가 낸 그대로를 `{claims}` 에 넣는다.
CLAIMS = "give: guide customer to use submit_referral tool"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    a2 = load_domain_a2(DOMAIN) or {}
    cpv = a2.get("claim_prov") or {}
    spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
    rows = a2["policy_ontology"]["rows"]
    maps = {ax: FD._a3_map(rows, {"axis": ax}) for ax in spec["eligible"]["show_axes"]}
    table = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}).strip()
    base = table + "\n\n" + X.FACTS[P.TASK] + "\n\n" + X.QUESTION
    choices = [l.strip().split(":")[0].strip() for l in table.splitlines() if l.startswith("  ")]
    msgs = Y.msgs_of(os.environ.get("T2_PROBE_TAG", "bank_elig_20260809i"), P.TASK)

    cut = len(msgs) - 1                       # 마지막(방아쇠) 직전까지 = 아직 안 무너진 문맥
    ctx = ("Here is a customer-service conversation so far.\n\n" + Y.render(msgs[:cut]))
    last = msgs[-1]
    last_txt = " ".join(str(getattr(last, "content", None)
                            or (last.get("content") if isinstance(last, dict) else "")).split())

    def fb(key):
        t = cpv.get(key)
        return t.replace("{claims}", CLAIMS) if t else None

    arms = collections.OrderedDict()
    arms["C_none   기준선"] = ""
    if fb("feedback_pending"):
        arms["A_imper  현행(명령형)"] = "System: " + fb("feedback_pending")
    if fb("feedback_ownership"):
        arms["B_owner  신규(표면화)"] = "System: " + fb("feedback_ownership")
    # ★길이 교란 통제 (2026-08-09·1차 실행이 A 0.586 vs B 0.943 을 냈는데 662자 대 213자였다).
    #   **같은 원문을 길이만 맞춰 쪼갠다** — 새 문장을 지어내지 않으므로 어휘 교란 0.
    #     A_head = 앞부분(사실 진술·명령 없음) · A_tail = 뒷부분(명령문)
    #   길이가 원인이면 둘이 같이 떨어지고, 명령이 원인이면 tail 만 떨어진다.
    _ap = fb("feedback_pending")
    _bn = len(fb("feedback_ownership") or "") or 213
    if _ap and len(_ap) > _bn:
        arms["A_head   앞부분(사실·짧게)"] = "System: " + _ap[:_bn]
        arms["A_tail   뒷부분(명령·짧게)"] = "System: " + _ap[-_bn:]
    # ★완결-문장 통제 (2026-08-09 2차): 위 두 arm 은 662자를 글자수로 자른 것이라 **문장이
    #   중간에서 끊긴다** — 미완성 문장 자체가 교란일 수 있다. 여기서는 **문장 경계로만** 자른다.
    #     E_nofault = 첫 문장에서 **비난 절만 제거**(약속 사실은 남긴다)
    #     F_faultonly = 첫 문장 **그대로**(비난 포함·완결)
    #   E 와 F 의 차이는 *"but the ledger shows it was never executed"* 한 절뿐이다.
    if _ap:
        _first = _ap.split(". ")[0] + "."
        _cut = _first.find(", but the conversation ledger shows")
        if _cut > 0:
            arms["F_fault   비난 문장(완결)"] = "System: " + _first
            arms["E_nofault 같은 문장−비난절"] = "System: " + _first[:_cut] + ": " + CLAIMS + "."
    arms["D_named  #26원문(양성통제)"] = "Assistant: " + last_txt

    print("model=%s · 궤적 %d(사용 %d) · 후보 %d · gold=%s"
          % (P.MODEL, len(msgs), cut, len(choices), P.GOLD_HEAD))
    for k in ("feedback_pending", "feedback_ownership"):
        v = cpv.get(k)
        print("  A2 %-20s %s" % (k, "있음(%d자)" % len(v) if v else "**없음**"))
    print("\n%-26s %10s %10s  %s" % ("arm", "p(gold)", "H(nats)", "argmax (자유생성 %d회)" % n))

    for label, tail in arms.items():
        prompt = ctx + (("\n\n" + tail) if tail else "") + "\n\n" + base
        d = P.first_token_dist(prompt, choices)
        heads = sorted({c.split()[0] for c in choices if c.split()}, key=len, reverse=True)
        bk = {}
        for h in heads:
            p = math.exp(P.lp_of(d, h))
            if p > 1e-12:
                bk[h] = p
        tot = sum(bk.values()) or 1.0
        ent = -sum((p / tot) * math.log(p / tot) for p in bk.values() if p > 0)
        pg = math.exp(P.lp_of(d, P.GOLD_HEAD))
        free = [" ".join(X.ask(prompt, 0.0 if i == 0 else 0.7).split())[:28] for i in range(n)]
        top = max(bk, key=bk.get) if bk else "?"
        print("%-26s %10.4f %10.3f  guided=%-10s free=%s"
              % (label, pg, ent, top, collections.Counter(free).most_common(2)))
    print("\n  예측: A·B 는 계좌 이름을 안 담으므로 유지 · D 만 붕괴.")
    print("  A 가 무너지면 우리 문구가 **직접** 좁히는 것 · A≈B 면 어제 수리는 이 축에서 효과 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
