# -*- coding: utf-8 -*-
"""X19 — A2 출처 감사: **정책·환경만인가, gold를 봤나** (2026-07-31·무료).

규칙([[23]]·사용자 지시 2026-07-31 축자): "A2 는 도메인 특화라도 정책만으로 해야지.
gold 보고 하는게 들어가면 안된다."

판정 방법 — A2 값 안의 **도메인 리터럴**(도구명·필드명·enum 같은 식별자)을 뽑아 출처를 대조한다:
  · **ENV**    = env 표면(도구 이름·인자 이름·enum)에 있다        → 기계 도출 가능 = opex 0 후보
  · **POLICY** = 도메인 정책 산문에 축자로 있다                   → 정당한 도메인 저작(opex +)
  · **DOC**    = KB/도메인 문서·DB에 있다(gold 제외)               → 정당(배포 시점에 가진다)
  · **NEITHER**= 셋 다 아니다                                     → ★검토 필요(gold 경유 의심)

★감사 무효화 방지: 근거 코퍼스에서 **경로에 `task`가 들어가는 파일은 전부 배제**한다.
  gold(task set)가 코퍼스에 섞이면 "gold를 봤나"라는 질문 자체가 무의미해진다([[03b]]).

★한계(정직): NEITHER는 **유죄 판정이 아니다**. 리터럴이 아니라 *규칙의 내용*이 gold에서 왔을
  수도 있고(리터럴은 다 정책에 있는데 임계값만 gold), 반대로 표기 차이로 NEITHER가 될 수도 있다.
  이 도구는 **검토 큐**를 만든다. 최종 판정은 `_note_<key>`의 축자 출처를 사람이 읽고 한다.

용법:
  py -3 x19_a2_provenance_audit.py --emit-literals out.json      # (로컬) 리터럴 추출
  py -3 x19_a2_provenance_audit.py --membership member.json      # (로컬) 원격 대조 결과로 판정
"""
import argparse
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_A2 = os.path.join(_HERE, "a2")
DOMAINS = ["banking_knowledge", "retail", "airline"]
# 식별자형 토큰만 본다(영어 산문 단어는 리터럴이 아니다): snake_case 2어절 이상 또는 접미사형
_IDENT = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b")
# ★스펙 어휘 vs 도메인 리터럴 — **기계로 가른다**(손 목록 금지).
#   A2에는 두 종류의 식별자가 섞여 있다: ① 엔진이 읽는 **스키마 키워드**(`applies_when`·
#   `bool_expr`·`dispatch_name_key` …)와 ② 그 도메인의 **실제 이름**(도구·필드·enum).
#   ①은 비용 회계에서 도메인 리터럴이 아니다. 가르는 기계 기준 = **엔진 소스가 그 토큰을
#   문자열로 언급하는가** — 엔진이 아는 이름이면 스펙 어휘다. (도메인 이름이 엔진 소스에
#   있으면 그건 엔진 리터럴 위반이고 x6h가 따로 잡는다 — 실측 0이라 이 기준은 안전하다.)
#   ⚠초판은 손으로 20개를 적었다가 banking NEITHER를 234/350으로 부풀렸다 — 대부분이
#   스펙 어휘였다. 집계에서 결론으로 직행하지 않고 원문을 읽어 잡았다([[08]]).
def _spec_vocab():
    """엔진 소스가 **따옴표 안에서** 언급하는 식별자 = 스펙 어휘.

    ★스코프는 `x6h_engine_literal_audit.discover_engine_files()`의 **import 폐포**를 쓴다.
      디렉터리의 `*.py`를 다 긁으면 분석 프로브(bank_*.py·x*.py)가 도메인 이름을 인용하므로
      스펙 어휘가 오염되고, 그러면 도메인 리터럴이 스펙으로 **거짓 면죄**된다([[03b]]).
    """
    try:
        from x6h_engine_literal_audit import discover_engine_files
        files = [os.path.join(_HERE, f) for f in discover_engine_files()]
    except Exception:
        files = [os.path.join(_HERE, "t2_gate_patch.py"), os.path.join(_HERE, "gate_interpreter.py")]
    vocab = set()
    for p in files:
        try:
            src = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for q in re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", src):
            vocab.add(q)
    return vocab


_ENGINE_VOCAB = _spec_vocab()


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def literals_of(value):
    s = json.dumps(value, ensure_ascii=False)
    return sorted({t for t in _IDENT.findall(s) if t not in _ENGINE_VOCAB})


def domain_keys(dom):
    """L3(specific) + L2(settings) 실키 → {key: value}. 분리 전이면 단일 파일."""
    out = {}
    for suffix in (".settings.json", ".specific.json"):
        p = os.path.join(_A2, dom + suffix)
        if os.path.exists(p):
            out.update({k: v for k, v in load(p).items() if not k.startswith("_")})
    if not out:
        p = os.path.join(_A2, dom + ".gate.json")
        if os.path.exists(p):
            out = {k: v for k, v in load(p).items() if not k.startswith("_")}
    return out


def notes_of(dom):
    out = {}
    for suffix in (".settings.json", ".specific.json", ".gate.json"):
        p = os.path.join(_A2, dom + suffix)
        if os.path.exists(p):
            out.update({k: v for k, v in load(p).items() if k.startswith("_note_")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-literals")
    ap.add_argument("--membership")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    per_key = {d: {k: literals_of(v) for k, v in domain_keys(d).items()} for d in DOMAINS}

    if args.emit_literals:
        allv = {d: sorted({t for ts in per_key[d].values() for t in ts}) for d in DOMAINS}
        with io.open(args.emit_literals, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(allv, ensure_ascii=False, indent=0))
        print("리터럴 → %s" % args.emit_literals)
        for d in DOMAINS:
            n = len({t for ts in per_key[d].values() for t in ts})
            print("   %-18s 키 %2d · 고유 리터럴 %d" % (d, len(per_key[d]), n))
        return

    if not args.membership:
        ap.error("--emit-literals 또는 --membership 중 하나가 필요하다")

    mem = load(args.membership)   # {domain: {literal: {"env":…,"policy":…,"doc":…}}}
    print("=" * 96)
    print("A2 출처 감사 — 리터럴 출처 대조 ([[23]]: 정책·환경만·gold 금지)")
    print("  ENV=환경에서 도출가능(opex 0 후보) · POLICY/DOC=정당한 저작 · **NEITHER=검토 필요**")
    print("=" * 96)
    for dom in DOMAINS:
        keys = per_key[dom]
        if not keys:
            continue
        notes = notes_of(dom)
        review = []
        print("\n[%s] 키 %d" % (dom, len(keys)))
        print("  %-28s %5s %5s %5s %5s  %s" % ("key", "lit", "env", "pol", "doc", "NEITHER(앞 4개)"))
        for k in sorted(keys):
            lits = keys[k]
            e = p = d_ = 0
            nei = []
            for t in lits:
                m = (mem.get(dom) or {}).get(t) or {}
                if t.startswith("task_"):
                    nei.append("★GOLD:" + t)   # task id = gold 참조. 절대 정당화되지 않는다.
                elif m.get("env"):
                    e += 1
                elif m.get("policy"):
                    p += 1
                elif m.get("doc") or t.startswith("doc_"):
                    # `doc_*` = KB_search가 **질의 시점에 붙이는 문서 id**라 원본 파일 본문·이름에는
                    # 없다. 배포 시점에 검색으로 얻는 것이므로 정당한 DOC 출처다(초판은 이걸
                    # NEITHER로 세어 4키를 거짓 고발할 뻔했다 — 원문 확인으로 잡음·[[08]]).
                    d_ += 1
                else:
                    nei.append(t)
            flag = "★" if nei else " "
            print("%s %-28s %5d %5d %5d %5d  %s" %
                  (flag, k, len(lits), e, p, d_, ", ".join(nei[:4]) + (" …" if len(nei) > 4 else "")))
            if nei:
                review.append((k, nei, bool(notes.get("_note_" + k))))
        print("\n  ★검토 큐 %d키 (NEITHER 리터럴 보유)" % len(review))
        for k, nei, has_note in review:
            print("    %-28s NEITHER %2d개 · 출처주석 %s" % (k, len(nei), "있음" if has_note else "**없음**"))


if __name__ == "__main__":
    main()
