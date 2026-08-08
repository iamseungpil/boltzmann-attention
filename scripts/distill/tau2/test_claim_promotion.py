# -*- coding: utf-8 -*-
"""claim_audit 승격 등가 게이트 (2026-07-31).

설계 = `CLAIM_AUDIT_ENGINE_PROMOTION_DESIGN_2026_07_31.md`. 근거 = C254([[23]] 감사에서
`claim_prov`·`completion_guard`만 **정책 근거 없음**으로 확정 — gold 경유도 아니고, 담긴 것이
banking 사실이 아니라 도메인-일반 원리였다).

이 승격의 가치는 **회계**이지 성능이 아니다. 그러므로 **행동이 바뀌면 버그**다:
  ① 로더가 합성한 `claim_prov`/`completion_guard`가 승격 전과 **바이트 동일**
  ② retail·airline은 결합 미선언 ⇒ **여전히 비활성**(새 도메인에 조용히 켜지면 회귀)
  ③ 산문에 업종 명사·kind enum이 남아 있지 않다(L1이 진짜 도메인-불변인가)
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import gate_interpreter as G  # noqa: E402

OK = True


def chk(c, m):
    global OK
    OK = OK and bool(c)
    print("  %s %s" % ("✓" if c else "✗", m))


def canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


# ★의도된 델타 (2026-08-08·C341). 스냅샷을 조용히 갱신하면 이 게이트가 죽는다 — 그러면
#   다음 사고를 못 잡는다. 그래서 **바꾼 필드만 이름으로 면제**하고 나머지는 그대로 동결하며,
#   면제한 필드는 **실제로 달라야** 한다(안 다르면 면제가 낡은 것이므로 그것도 실패다).
DELTA = {
    ("claim_prov", "question"):
        "주장마다 `tool`(그것을 수행한 호출 이름)을 함께 내게 한다. 구판은 모델의 `kind` 라벨을 "
        "event_map 색인으로 썼는데 라벨이 우리 패턴과 어긋나 **거짓 발화**가 났다 — run f 실측: "
        "`log_verification` 이 양쪽 sim 에서 실행됐는데 kind=record_update 로 선언돼 "
        "'ledger shows NO such event' 를 결정 턴마다 내보냈다([[25]]). 권위는 실행 원장이고 "
        "kind 는 해석이다([[52]]). ⚠라이브 효과 미측정.",
}

print("[①] 합성 결과 == 승격 전 (의도된 델타 제외 바이트 동일)")
orig_path = os.path.join(HERE, "_orig_claim.json")
if os.path.exists(orig_path):
    orig = json.load(io.open(orig_path, encoding="utf-8"))
    m = G.load_domain_a2("banking_knowledge")
    for k in ("claim_prov", "completion_guard"):
        want = {kk: vv for kk, vv in orig[k].items() if not kk.startswith("_")}
        got = m.get(k) or {}
        frozen = [kk for kk in want if (k, kk) not in DELTA]
        chk(all(canon(got.get(kk)) == canon(want[kk]) for kk in frozen),
            "%s 동일 (동결 %d필드)" % (k, len(frozen)))
        for kk in [x for x in want if (k, x) in DELTA]:
            chk(canon(got.get(kk)) != canon(want[kk]),
                "%s.%s 는 의도된 델타이고 실제로 달라졌다 (면제가 낡지 않았다)" % (k, kk))
else:
    chk(False, "_orig_claim.json 없음 — 승격 전 스냅샷이 있어야 등가를 잴 수 있다")

print("\n[②] 미선언 도메인은 비활성 유지 (조용히 켜지면 회귀)")
for dom in ("retail", "airline"):
    m = G.load_domain_a2(dom) or {}
    chk("claim_prov" not in m and "completion_guard" not in m,
        "%s 비활성" % dom)

print("\n[③] L1 산문이 진짜 도메인-불변인가")
base = json.load(io.open(os.path.join(HERE, "a2", "base", "shared.json"), encoding="utf-8"))
ca = base.get("claim_audit") or {}
chk(bool(ca), "base에 claim_audit 존재")
blob = json.dumps(ca, ensure_ascii=False)
chk("{kinds}" in blob, "kind enum이 {kinds} 자리표시자로 빠져 있다")
for w in ("rho", "dispute_file", "credit_card", "banking"):
    chk(w not in blob, "업종어 '%s' 없음" % w)

print("\nRESULT: %s" % ("ALL PASS" if OK else "FAIL"))
sys.exit(0 if OK else 1)
