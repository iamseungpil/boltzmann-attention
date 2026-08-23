# -*- coding: utf-8 -*-
"""R6 — 전제-딸린 갱신 명령문 검정 (2026-08-24 · refute_3 ⒠ · 20런 실측).

결함(수리 전): `get_reward_discrepancies`(ratefix) 의 `return_template` 축자
  *"… — after its dispute is resolved, update that transaction's rewards to EXACTLY the correct
    value shown: {details}"*
이 gold 밖 `update_transaction_rewards_3847` 를 만든다. 원인은 모델이 아니라 **발화 위치**다.

  ⌜불변식⌝ **전제를 관측할 수 없는 site 는 전제-딸린 명령문을 낼 수 없다.**
  `scaffold_get_tools[*].return_template` 은 도구 반환 시점에 렌더되고 원장 술어
  (`after`/`requires`)가 없다 ⇒ *"after its dispute is resolved"* 의 성립을 관측할 방법이
  구조적으로 없다. 관측 없이 전제를 말하면 전제는 무시되고 **명령만 남는다**.
  명령문의 정당한 자리는 원장 술어를 **가진** site — `follow_up_chains[*]` 뿐이다.

  ⚠이 불변식은 **태스크 id·제품명·문구로 조건화하지 않는다**. 판정 재료는 *선언의 구조*
  (그 site 가 원장 술어를 갖는가)와 *우리가 저작한 문면* 둘뿐이고, 도메인 어휘 0·gold 0 이다
  (사용자 일반화 시험 2026-08-24).

검정 축:
  ① 축자 부재 — 구 명령문이 3사본 어디에도 없다
  ② 회귀 보존 — {details}(기대값 노출·2026-07-19 026 회귀 방지) · return_template_empty ·
     op/isolate/params/grounded_params 불변
  ③ 3사본 바이트 동일([[24]]) — return_template · chain feedback
  ④ 코퍼스 불변식 lint — 세 도메인 A2 의 **모든** 반환 문면에 '시간 전제 + 명령절' 동거 0
  ⑤ 양성 대조 — 구 문면을 되돌린 사본에서 ④가 실제로 잡힌다(검정이 죽어 있지 않다)
  ⑥ 부정통제(능력 삭제 아님) — 원장 술어를 **가진** site(follow_up_chains)는 여전히 갱신을
     지시하고, 발화 조건(after/requires/resign_th)과 양방향 축자가 불변이다
  ⑦ [[25]] 자기-인용 정직성 — chain 문면이 우리 반환문이 *말하지 않는 것*을 인용하지 않는다
  ⑧ [[64]] — 금지 문면이 '무엇을 하면 풀리나'를 함께 담는다

오프라인 전용(유료 X·[[09]]). 실행: py -3 test_return_template_imperative.py
"""
import copy
import io
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

PATHS = ["a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json",
         "a2/split/banking_knowledge.core.json"]
TOOL = "get_reward_discrepancies"

# 수리 전 축자(양성 대조용 상수 — 되살아나면 ①·④가 잡는다)
OLD = ("Transactions whose recorded reward does NOT match the expected reward under the "
       "reward-rate policy (each needs a cash back dispute). The CORRECT total reward per policy "
       "is shown for each — after its dispute is resolved, update that transaction's rewards "
       "to EXACTLY the correct value shown: {details}")
OLD_CLAUSE = "update that transaction's rewards to EXACTLY the correct value shown"
OLD_CITE = "The reward-check tool's own output states"

PASS, FAIL = [], []


def check(label, cond):
    (PASS if cond else FAIL).append(label)
    print(("  PASS " if cond else "  FAIL ") + label)


def load(rel):
    return json.load(io.open(os.path.join(HERE, rel), encoding="utf-8"))


def ratefix(j):
    t = next((x for x in j.get("scaffold_get_tools") or [] if x.get("name") == TOOL), None)
    return t, ((t or {}).get("variants") or {}).get("ratefix")


def update_chain(j):
    return next((c for c in j.get("follow_up_chains") or []
                 if any("update_transaction_rewards" in str(r) for r in (c.get("requires") or []))),
                None)


# ── ④ lint 술어 ──────────────────────────────────────────────────────────────
# 시간 전제 = 선행 사건의 성사를 가리키는 접속사. 명령절 = 절 머리에 오는 상태-변경 동사 원형.
# 두 개가 **한 문장 안에** 오면, 그 문장은 관측 불가능한 전제에 명령을 매단 것이다.
TEMPORAL = re.compile(r"(?i)\b(after|once|as soon as|when)\b")
IMPERATIVE_CLAUSE = re.compile(
    r"(?i)(?:^|[.;:—\-]\s*|,\s*(?:then\s+)?|\band then\s+)"
    r"(update|submit|apply|credit|approve|deny|close|create|issue|adjust|"
    r"set|change|log|pay|refund|reverse|unlock|call)\b")


def offending_sentences(tpl):
    """그 문면이 '시간 전제 + 명령절' 을 한 문장에 담은 곳을 돌려준다(없으면 [])."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+", tpl or ""):
        if s.strip() and TEMPORAL.search(s) and IMPERATIVE_CLAUSE.search(s):
            out.append(re.sub(r"\s+", " ", s.strip()))
    return out


def _walk_strings(o, in_note=False):
    """(문자열, 주석-내부인가) 를 전부 편다. `_` 로 시작하는 키 아래는 전부 주석으로 본다."""
    if isinstance(o, dict):
        for k, v in o.items():
            for pair in _walk_strings(v, in_note or str(k).startswith("_")):
                yield pair
    elif isinstance(o, list):
        for v in o:
            for pair in _walk_strings(v, in_note):
                yield pair
    elif isinstance(o, str):
        yield o, in_note


def live_strings(j):
    return [s for s, note in _walk_strings(j) if not note]


def note_strings(j):
    return [s for s, note in _walk_strings(j) if note]


def all_return_templates(j):
    """이 A2 의 모든 반환 문면 — 원장 술어가 **없는** site 전량."""
    for t in j.get("scaffold_get_tools") or []:
        nm = t.get("name")
        for key in ("return_template", "return_template_empty"):
            if isinstance(t.get(key), str):
                yield "%s.%s" % (nm, key), t[key]
        for vn, v in (t.get("variants") or {}).items():
            for key in ("return_template", "return_template_empty"):
                if isinstance(v.get(key), str):
                    yield "%s/%s.%s" % (nm, vn, key), v[key]


def main():
    js = {rel: load(rel) for rel in PATHS}
    rfs = {rel: ratefix(js[rel])[1] for rel in PATHS}
    tls = {rel: ratefix(js[rel])[0] for rel in PATHS}
    chs = {rel: update_chain(js[rel]) for rel in PATHS}
    r0, t0, c0 = rfs[PATHS[0]], tls[PATHS[0]], chs[PATHS[0]]

    # ── ① 축자 부재 ───────────────────────────────────────────────────────────
    # ⚠술어는 **비-주석 값 전량**이다. `_` 로 시작하는 키는 주석이라 렌더되지 않고, 수리 이력은
    #   그 안에서 구 축자를 **인용해야** 한다(provenance) — 그것까지 잡으면 검정이 기록을 죽인다.
    print("[①] 구 명령문 축자가 3사본의 **살아 있는 문면**(비-주석 값) 어디에도 없다")
    check("①: ratefix/tool/chain 전부 실재", all(r is not None for r in rfs.values())
          and all(c is not None for c in chs.values()))
    for rel in PATHS:
        live = live_strings(js[rel])
        check("①: %s 살아 있는 문면 %d개 중 구 명령절 0"
              % (os.path.basename(rel), len(live)),
              not any(OLD_CLAUSE in s for s in live))
        check("①: %s 주석은 출처로 구 축자를 보존한다(기록 死 아님)"
              % os.path.basename(rel),
              any(OLD_CLAUSE in s for s in note_strings(js[rel])))

    # ── ② 회귀 보존 ───────────────────────────────────────────────────────────
    print("[②] 회귀 보존 — 값 노출·빈-결과 문면·op/isolate 불변")
    check("②: {details} 유지(기대값 노출 = 2026-07-19 task_026 회귀 방지)",
          "{details}" in (r0.get("return_template") or ""))
    check("②: detail_item_template 이 기록값·기대값 둘 다 낸다",
          all(k in (r0.get("detail_item_template") or "")
              for k in ("{actual_int}", "{expected_floor}")))
    check("②: return_template_empty 불변(갱신 지시 없음·기존 D4 수리 보존)",
          "rewards update from this result" in (r0.get("return_template_empty") or "")
          and not offending_sentences(r0.get("return_template_empty")))
    for k in ("op", "isolate", "params", "grounded_params", "byref_join"):
        check("②: ratefix['%s'] 3사본 동일(거동 불변)" % k,
              rfs[PATHS[0]].get(k) == rfs[PATHS[1]].get(k))
    check("②: base(비-ratefix) return_template 불변(명령문 없었음)",
          t0.get("return_template", "").endswith("{ids}")
          and not offending_sentences(t0.get("return_template")))

    # ── ③ 3사본 바이트 동일([[24]]) ───────────────────────────────────────────
    print("[③] 3사본 바이트 동일 — return_template · chain feedback")
    check("③: ratefix.return_template 3사본 동일",
          rfs[PATHS[0]]["return_template"] == rfs[PATHS[1]]["return_template"]
          == rfs[PATHS[2]]["return_template"])
    check("③: chain.feedback 3사본 동일",
          chs[PATHS[0]]["feedback"] == chs[PATHS[1]]["feedback"] == chs[PATHS[2]]["feedback"])

    # ── ④ 코퍼스 불변식 lint ──────────────────────────────────────────────────
    print("[④] 원장 술어 없는 site(모든 반환 문면) = '시간 전제 + 명령절' 동거 0")
    trips = []
    scanned = 0
    a2dir = os.path.join(HERE, "a2")
    files = [os.path.join(a2dir, f) for f in sorted(os.listdir(a2dir)) if f.endswith(".json")]
    files += [os.path.join(HERE, PATHS[2])]
    for p in files:
        try:
            j = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(j, dict):
            continue
        for nm, tpl in all_return_templates(j):
            scanned += 1
            for s in offending_sentences(tpl):
                trips.append("%s :: %s :: %s" % (os.path.basename(p), nm, s[:150]))
    check("④: 반환 문면 %d개 스캔 · 위반 %d — %s"
          % (scanned, len(trips), "없음" if not trips else trips[0]), not trips)

    # ── ⑤ 양성 대조 ───────────────────────────────────────────────────────────
    print("[⑤] 양성 대조 — 구 문면을 되돌린 사본에서 ①④가 실제로 잡힌다")
    probe = copy.deepcopy(r0)
    probe["return_template"] = OLD
    check("⑤: 구 문면에 구 명령절 실재(대조가 진짜 구 문면이다)", OLD_CLAUSE in OLD)
    hits = offending_sentences(probe["return_template"])
    check("⑤: 구 문면이 lint 에 걸린다 — %s"
          % (hits[0][:120] if hits else "안 걸림(검정 死)"), bool(hits))
    check("⑤: 수리된 문면은 안 걸린다", not offending_sentences(r0["return_template"]))
    # 부정통제(무내용 재시도·[[57]]): 명령문 없이 '시간 전제'만 있는 문장은 걸리지 않아야 한다
    check("⑤: 부정통제 — 전제만 있고 명령이 없으면 미검출",
          not offending_sentences("Rewards are corrected after the dispute is resolved."))
    # 부정통제 2: 명령만 있고 전제가 없으면 걸리지 않는다(우리 문면의 정상 지시를 안 죽인다)
    check("⑤: 부정통제 — 명령만 있고 전제가 없으면 미검출",
          not offending_sentences("Submit a cash back dispute for each id above."))
    # ⑤b 끝단 양성대조 — **선언 전체를 되돌린 사본**에 ①④를 그대로 돌린다.
    #    (실파일을 되돌리지 않는다: 동시 실행 중인 다른 세션이 구 문면을 읽어 되쓸 수 있다.)
    reverted = copy.deepcopy(js[PATHS[0]])
    _, rrf = ratefix(reverted)
    rrf["return_template"] = OLD
    for k in [x for x in list(rrf) if x.startswith("_")]:
        del rrf[k]                       # 주석의 출처 인용이 ① 을 대신 만족시키지 못하게
    update_chain(reverted)["feedback"] = ("Error: [FOLLOW-UP] " + OLD_CITE
                                          + " that AFTER a dispute is resolved …")
    check("⑤b: 되돌린 사본에서 ①(살아 있는 문면의 구 명령절)이 잡힌다",
          any(OLD_CLAUSE in s for s in live_strings(reverted)))
    check("⑤b: 되돌린 사본에서 ④(lint)가 잡힌다",
          any(offending_sentences(t) for _, t in all_return_templates(reverted)))
    check("⑤b: 되돌린 사본에서 ⑦(자기-인용)이 잡힌다",
          OLD_CITE in (update_chain(reverted).get("feedback") or ""))
    check("⑤b: 되돌린 사본에서 ⑧([[64]] fix-naming 부재)이 잡힌다",
          "The step this result calls for is the dispute" not in rrf["return_template"])

    # ── ⑥ 부정통제: 능력 삭제가 아니라 자리 이동 ───────────────────────────────
    print("[⑥] 원장 술어를 **가진** site 는 여전히 갱신을 지시한다(028 경로 보존)")
    fb = c0.get("feedback") or ""
    check("⑥: chain 이 원장 술어를 가진다(after ∧ requires)",
          bool(c0.get("after")) and bool(c0.get("requires")))
    check("⑥: chain 이 갱신 도구를 requires 로 지목",
          any("update_transaction_rewards" in str(r) for r in c0["requires"]))
    check("⑥: chain 문면이 갱신 실행을 지시한다(능력 삭제 아님)",
          "unlock and call the update tool now" in fb)
    check("⑥: chain 양방향 축자 보존(test_c204_nextrun D8 검사축)",
          "If resolution has NOT been confirmed yet, do not update" in fb)
    check("⑥: chain 발화 조건 불변 — after=[submit_cash_back_dispute*] · resign_th=1",
          len(c0["after"]) == 1 and "submit_cash_back_dispute" in str(c0["after"][0])
          and c0.get("resign_th") == 1)
    check("⑥: chain 은 lint 대상이 아니다 — 전제-딸린 지시가 **허용되는** 자리",
          bool(TEMPORAL.search(fb)))

    # ── ⑦ [[25]] 자기-인용 정직성 ─────────────────────────────────────────────
    print("[⑦] chain 이 우리 반환문에 없는 말을 인용하지 않는다([[25]])")
    check("⑦: 구 자기-인용('The reward-check tool's own output states') 제거", OLD_CITE not in fb)
    check("⑦: 인용처가 정책 문서로 옮겨졌다",
          "internal procedure for applying resolved cash back dispute corrections" in fb)

    # ── ⑧ [[64]] 거부는 고치는 법까지 ─────────────────────────────────────────
    print("[⑧] 금지 문면이 '무엇을 하면 풀리나'를 함께 담는다([[64]])")
    ret = r0["return_template"]
    check("⑧: 무엇이 틀렸나 — 이 결과로는 갱신하지 말 것",
          "Do NOT change any transaction's rewards on the strength of this result" in ret)
    check("⑧: 무엇을 하면 풀리나 — 다음 단계가 분쟁임을 이름으로 지목",
          "The step this result calls for is the dispute" in ret)
    check("⑧: 갱신의 피연산자 출처를 이름으로 지목(정책 축자 유래)",
          "read from the resolved disputes" in ret)
    check("⑧: 검사 범위 정직 — 분쟁 레코드를 안 읽었음을 자인([[25]])",
          "It read no dispute record" in ret)

    # ── ⑨ 라이브 병합 렌더([[24]] 편집 후 load_domain_a2 확인 의무) ─────────────
    print("[⑨] 병합 A2 로 실제 렌더 — 라이브가 보는 문면이 수리본이다")
    import gate_interpreter as GI
    merged = GI.load_domain_a2("banking_knowledge")
    _, mrf = ratefix(merged)
    check("⑨: 병합 A2 의 return_template == 정본 3사본", mrf["return_template"] == ret)
    dets = "; ".join(mrf["detail_item_template"].format(id=i, actual_int=a, expected_floor=e)
                     for i, a, e in [("txn_a8f1c2d3e403", 3150, 6300),
                                     ("txn_a8f1c2d3e404", 160, 1599)])
    # 엔진 호출부(t2_scaffold_get `_txt = d.get(_tpl_key).format(...)`)와 같은 kwargs
    txt = mrf["return_template"].format(ids="txn_a8f1c2d3e403, txn_a8f1c2d3e404",
                                        details=dets, delta_total=0.0)
    check("⑨: 렌더 성공 + 기대값이 문면에 실린다(026 회귀 방지의 실물 확인)",
          "6300" in txt and "1599" in txt)
    check("⑨: 렌더된 문면에 구 명령절 0", OLD_CLAUSE not in txt)
    check("⑨: 렌더된 문면이 lint 통과", not offending_sentences(txt))
    check("⑨: empty 경로도 렌더 가능·명령문 0",
          "{" not in mrf["return_template_empty"]
          and not offending_sentences(mrf["return_template_empty"]))

    print("\n== 결과: %d PASS / %d FAIL ==" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED:")
        for x in FAIL:
            print("  - " + x)
        sys.exit(1)
    print("ALL PASS — 전제-딸린 명령문은 원장 술어를 가진 site 로만 남았다.")


if __name__ == "__main__":
    main()
