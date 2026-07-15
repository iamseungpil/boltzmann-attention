# -*- coding: utf-8 -*-
"""synth_schema_classify.py — Track B 학습데이터 v0: 도메인일반 스키마-분류(+prior-conflict) 생성 (2026-07-16).

스킬 = "제공된 enum 스키마 정의를 읽고 NL 분류·salient prior로 안 덮기". banking 아님([[11]]).
각 taxonomy = 카테고리+정의 + salient(프로토타입=prior-default) 지정. 케이스 2종:
  clear         : NL이 한 카테고리 정의에 명확 매칭.
  prior-conflict: NL에 *salient 단서* 有 but *정의상 판별자*가 다른 카테고리 지목(=banking fraud↔not_as_described 동형).
산출: SFT(schema+NL→gold) JSONL + DPO pairs(gold ≻ salient-default·prior-conflict서만).
다양성([[12]]): 5 도메인·표현 변형·판별자=정의적(어휘아님). banking 스키마 절대 미포함(전이=held-out).

★결정론(Date/random 금지 환경 대응): 인덱스 기반 변형(무작위 대신 순열). --n 케이스수.
사용: py synth_schema_classify.py --n 600 --out synth_f3.jsonl
"""
import json, sys, io, argparse
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 5 도메인일반 taxonomy (정의 + salient prior-default + 카테고리별 시나리오/판별자) ──
# 각 카테고리: (정의, [clear 시나리오 표현들], [정의적 판별자 구문들])
TAXONOMIES = {
  "support_ticket": {
    "salient": "bug_report",
    "cats": {
      "bug_report": ("Existing feature does not work as designed (error, crash, wrong output).",
                     ["the export button throws an error", "the app crashes when I open settings", "totals are calculated wrong"],
                     ["it used to work and now errors", "returns an error message", "produces incorrect output for a working feature"]),
      "feature_request": ("A capability that does NOT yet exist is being asked for.",
                          ["could you add a dark mode", "I wish it could export to PDF", "please support bulk edits"],
                          ["the capability does not exist yet", "asking to add something new", "no such option exists today"]),
      "billing_question": ("A question or dispute about charges, invoices, or payment.",
                           ["why was I charged twice", "my invoice seems too high", "I need a refund on my subscription"],
                           ["concerns an amount charged", "about an invoice or payment", "wants a refund"]),
      "account_access": ("Unable to log in or access the account (password, lockout, 2FA).",
                         ["I can't log in", "locked out after too many attempts", "my 2FA code never arrives"],
                         ["cannot authenticate or sign in", "account is locked", "credential/login problem"]),
      "how_to_question": ("User asks how to accomplish a task with existing features.",
                          ["how do I share a folder", "where is the setting for notifications", "how can I invite a teammate"],
                          ["asks how to use an existing feature", "a usage/instruction question", "feature exists, user needs guidance"]),
    }},
  "product_defect": {
    "salient": "manufacturing_defect",
    "cats": {
      "manufacturing_defect": ("Fault present from production (materials, assembly, workmanship).",
                               ["the seam was already torn out of the box", "button fell off first use", "it was assembled crooked"],
                               ["fault existed before use", "flaw in materials or assembly", "defective as produced"]),
      "shipping_damage": ("Damage that occurred in transit / during delivery.",
                          ["the box was crushed on arrival", "screen cracked during shipping", "arrived dented from the courier"],
                          ["damaged in transit", "packaging was crushed in delivery", "harm occurred during shipping"]),
      "wrong_item_sent": ("A different item than ordered was delivered.",
                          ["I ordered blue but got red", "this is a different model than I bought", "wrong size arrived"],
                          ["item differs from what was ordered", "incorrect product delivered", "mismatch to the order"]),
      "user_misuse": ("Damage caused by the customer's own handling.",
                      ["I dropped it in water", "it broke after I sat on it", "damaged when I forced it open"],
                      ["harm caused by customer handling", "misuse by the owner", "self-inflicted damage"]),
      "missing_parts": ("Components absent from the package.",
                        ["the charger wasn't included", "two screws are missing", "no manual in the box"],
                        ["a component is absent", "parts missing from package", "incomplete contents"]),
    }},
  "insurance_claim": {
    "salient": "accident",
    "cats": {
      "accident": ("Unintentional collision or mishap.",
                   ["I rear-ended a car at a light", "slipped and dropped it", "backed into a pole"],
                   ["unintentional collision", "an accidental mishap", "no intent, mishap"]),
      "theft": ("Property was stolen by another party.",
                ["my bike was stolen from the rack", "someone broke in and took the TV", "wallet lifted on the train"],
                ["property taken by another", "stolen item", "a thief removed it"]),
      "natural_disaster": ("Damage from weather or natural events.",
                           ["flood water ruined the basement", "hail dented the roof", "a tree fell in the storm"],
                           ["caused by weather/nature", "storm or flood damage", "natural event"]),
      "vandalism": ("Intentional damage by another person.",
                    ["someone keyed my car overnight", "windows smashed on purpose", "graffiti sprayed on the wall"],
                    ["deliberate damage by another", "intentional destruction", "malicious act by a person"]),
      "mechanical_failure": ("Part failed on its own without external cause.",
                             ["the engine seized while idling", "transmission gave out on the highway", "the compressor died"],
                             ["component failed internally", "no external cause, part broke", "mechanical breakdown"]),
    }},
  "hr_request": {
    "salient": "time_off",
    "cats": {
      "time_off": ("Request for vacation, sick, or personal leave.",
                   ["I'd like to take next Friday off", "requesting a week of vacation", "need a sick day tomorrow"],
                   ["asking for leave/vacation", "time away from work", "PTO request"]),
      "payroll_issue": ("A problem with pay, hours, or paycheck.",
                        ["my paycheck was short this month", "overtime wasn't paid", "wrong tax withheld"],
                        ["concerns the paycheck/pay", "hours or wages problem", "pay amount is wrong"]),
      "benefits_enrollment": ("Enrolling in or changing benefits (health, retirement).",
                              ["I want to add my spouse to health insurance", "change my 401k contribution", "enroll in dental"],
                              ["about benefits enrollment", "health/retirement plan change", "coverage election"]),
      "workplace_complaint": ("Reporting a workplace conduct or environment issue.",
                              ["my manager yells at the team", "unsafe conditions on the floor", "a colleague is harassing me"],
                              ["reports conduct/environment issue", "complaint about treatment", "workplace grievance"]),
      "equipment_request": ("Requesting tools or hardware to do the job.",
                            ["I need a second monitor", "my laptop is too slow, need a new one", "requesting a headset"],
                            ["asking for work equipment", "hardware/tools request", "needs a device to work"]),
    }},
  "content_moderation": {
    "salient": "spam",
    "cats": {
      "spam": ("Unsolicited bulk or promotional junk content.",
               ["same ad posted 50 times", "bulk promo links everywhere", "repetitive marketing junk"],
               ["unsolicited bulk promotion", "repetitive junk/ads", "mass promotional content"]),
      "harassment": ("Targeted abuse or threats toward a specific person.",
                     ["they keep insulting one user by name", "threatening messages to a member", "coordinated pile-on against her"],
                     ["targets a specific person", "abuse/threats at an individual", "personal attack"]),
      "misinformation": ("False factual claims presented as true.",
                         ["claims the vaccine contains microchips", "fake statistics about the election", "a debunked health cure"],
                         ["false factual claim", "spreads debunked info", "presents falsehood as fact"]),
      "copyright": ("Unauthorized use of protected material.",
                    ["reposted a full movie", "someone stole my article word for word", "uploaded the paid course"],
                    ["unauthorized use of protected work", "copies owned material", "IP infringement"]),
      "off_topic": ("Content unrelated to the forum's subject.",
                    ["posting recipes in a coding forum", "car talk in the gardening group", "unrelated chit-chat"],
                    ["unrelated to the topic", "wrong forum/section", "off-subject content"]),
    }},
}

CONNECTORS = ["Note that", "For context,", "Importantly,", "To be clear,", "Also,", "Specifically,"]
OPENERS = ["Hi, ", "Hello — ", "", "Quick one: ", "I need help. ", "Reaching out because "]

# per-taxonomy generic 문제진술 = salient 디폴트를 유발하나 gold-사실과 *일관* 가능(banking "dispute this charge"→fraud 디폴트 동형).
# prior-conflict = generic(salient 유발) + gold clear 사실 = 일관되게 gold·표면만 salient.
GENERIC = {
  "support_ticket": ["Something seems wrong with the app", "The product isn't working how I expected", "I'm having trouble with the software"],
  "product_defect": ["My item is damaged", "The product has a problem", "Something's wrong with what I received"],
  "insurance_claim": ["I need to file a claim for damage to my property", "My property got damaged and I want to claim", "There's damage I need covered"],
  "hr_request": ["I have a request for HR", "I need something sorted out at work", "There's a work matter I need help with"],
  "content_moderation": ["This post should be removed", "Please review this content", "I'm reporting this post"],
}


def _pick(lst, i):
    return lst[i % len(lst)]


def build_cases(n):
    """clear + prior-conflict 케이스 생성(결정론·인덱스 순열). 반환 [{schema, nl, gold, salient, type}]."""
    cases = []
    tax_names = list(TAXONOMIES)
    idx = 0
    while len(cases) < n:
        tname = tax_names[idx % len(tax_names)]
        T = TAXONOMIES[tname]; cats = T["cats"]; salient = T["salient"]
        catlist = list(cats)
        # 번갈아 clear / prior-conflict
        make_conflict = (idx % 2 == 1)
        gold_cat = catlist[(idx // 2) % len(catlist)]
        if make_conflict and gold_cat == salient:
            gold_cat = catlist[((idx // 2) + 1) % len(catlist)]  # 충돌은 non-salient gold
        gdef, gscen, gdisc = cats[gold_cat]
        opener = _pick(OPENERS, idx)
        if make_conflict:
            # generic 문제진술(salient 디폴트 유발) + gold clear 시나리오 = *일관되게 gold*·표면만 salient.
            # 모델은 salient 점프 말고 gold 사실로 분류해야(banking: "dispute charge"→fraud 디폴트지만 정의는 different).
            gen = _pick(GENERIC[tname], idx)
            scen = _pick(gscen, idx)
            nl = ("%s%s. %s %s." % (opener, gen, _pick(CONNECTORS, idx), scen))
            typ = "prior-conflict"
        else:
            scen = _pick(gscen, idx)
            disc = _pick(gdisc, idx + 2)
            nl = ("%s%s. %s %s." % (opener, scen[0].upper() + scen[1:], _pick(CONNECTORS, idx + 1), disc))
            typ = "clear"
        schema_txt = "\n".join("- '%s': %s" % (c, cats[c][0]) for c in catlist)
        cases.append({"taxonomy": tname, "field": tname + "_category", "options": catlist,
                      "schema": schema_txt, "nl": nl, "gold": gold_cat,
                      "salient_default": salient, "type": typ})
        idx += 1
    return cases


def to_sft(c):
    prompt = ("Classify the situation into exactly ONE allowed category, using ONLY the definitions.\n"
              "Allowed values for %s:\n%s\n\nSituation:\n%s\n\nOutput ONLY the exact category value." %
              (c["field"], c["schema"], c["nl"]))
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": c["gold"]}]}


def to_dpo(c):
    """prior-conflict서만: gold ≻ salient-default (prior 억제 신호)."""
    if c["type"] != "prior-conflict" or c["gold"] == c["salient_default"]:
        return None
    prompt = ("Classify the situation into exactly ONE allowed category, using ONLY the definitions.\n"
              "Allowed values for %s:\n%s\n\nSituation:\n%s\n\nOutput ONLY the exact category value." %
              (c["field"], c["schema"], c["nl"]))
    return {"prompt": prompt, "chosen": c["gold"], "rejected": c["salient_default"]}


def _prompt(c):
    return ("Classify the situation into exactly ONE allowed category, using ONLY the definitions.\n"
            "Allowed values for %s:\n%s\n\nSituation:\n%s\n\nOutput ONLY the exact category value." %
            (c["field"], c["schema"], c["nl"]))


def eval_base(cases, base, model):
    """base 모델이 synth서 salient로 mode-collapse하나(§6.2·banking 실패모드 재현 확인)."""
    from openai import OpenAI
    from collections import Counter
    cl = OpenAI(base_url=base, api_key="x")
    by = defaultdict(lambda: [0, 0]); salient_hit = defaultdict(lambda: [0, 0])
    for i, c in enumerate(cases):
        try:
            r = cl.chat.completions.create(model=model, messages=[{"role": "user", "content": _prompt(c)}], temperature=0, max_tokens=20)
            pred = r.choices[0].message.content.strip().strip("'\"").lower()
        except Exception as e:
            print("ERR", e); break
        ok = c["gold"].lower() in pred or pred in c["gold"].lower()
        by[c["type"]][0] += int(ok); by[c["type"]][1] += 1
        if c["type"] == "prior-conflict":
            hit = c["salient_default"].lower() in pred
            salient_hit["conflict"][0] += int(hit); salient_hit["conflict"][1] += 1
        if i < 4:
            print("[%s] gold=%s pred=%s %s" % (c["type"][:5], c["gold"], pred[:24], "OK" if ok else "X"), flush=True)
    print("\n=== base synth eval (mode-collapse 재현?·%s·n=%d) ===" % (model, len(cases)))
    for t, (ok, tot) in by.items():
        print("  %-16s acc %d/%d = %.1f%%" % (t, ok, tot, 100 * ok / max(tot, 1)))
    sh, st = salient_hit["conflict"]
    print("  prior-conflict서 salient-default 예측율 = %d/%d = %.1f%% (높으면=banking mode-collapse 재현=학습표적 유효)" % (sh, st, 100 * sh / max(st, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default="synth_f3.jsonl")
    ap.add_argument("--holdout", default="content_moderation", help="held-out taxonomy(전이검증용·학습제외)")
    ap.add_argument("--evalbase", action="store_true", help="base 모델 synth eval(mode-collapse 재현 확인)")
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()
    if a.evalbase:
        cases = build_cases(a.n)
        eval_base(cases, a.base, a.model)
        return
    cases = build_cases(a.n)
    train = [c for c in cases if c["taxonomy"] != a.holdout]
    held = [c for c in cases if c["taxonomy"] == a.holdout]
    with open(a.out, "w", encoding="utf-8") as fh:
        for c in train:
            fh.write(json.dumps(to_sft(c), ensure_ascii=False) + "\n")
    dpo = [to_dpo(c) for c in train]; dpo = [x for x in dpo if x]
    with open(a.out.replace(".jsonl", "_dpo.jsonl"), "w", encoding="utf-8") as fh:
        for x in dpo:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(a.out.replace(".jsonl", "_heldout_cases.jsonl"), "w", encoding="utf-8") as fh:
        for c in held:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    from collections import Counter
    print("=== synth 스키마-분류 v0 (도메인일반·banking 미포함·[[11]]) ===")
    print("  전 %d · train %d(SFT) · DPO pairs %d · held-out(%s) %d" % (len(cases), len(train), len(dpo), a.holdout, len(held)))
    print("  train taxonomy:", dict(Counter(c["taxonomy"] for c in train)))
    print("  type 분포:", dict(Counter(c["type"] for c in cases)))
    print("  예시 prior-conflict:")
    for c in cases:
        if c["type"] == "prior-conflict":
            print("    [%s] gold=%s salient=%s" % (c["taxonomy"], c["gold"], c["salient_default"]))
            print("      NL:", c["nl"][:140]); break
    print("  → %s (SFT) · *_dpo.jsonl (prior억제) · *_heldout_cases.jsonl (전이eval)" % a.out)
    print("  [[12]] 다양성 v0=5도메인·표현변형. 확장(더많은 taxonomy·NL패러프레이즈)이 전이 강건성 좌우.")


if __name__ == "__main__":
    main()
