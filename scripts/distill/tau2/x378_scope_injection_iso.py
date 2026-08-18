# -*- coding: utf-8 -*-
r"""x378 — **범위 밖 주입이 다음 행동을 틀어놓는가**(격리 ①″ · 원장 C535ⓖ · [[62]] ① 의무).

## 왜 이 모양인가 (원안 폐기 경위)

핸드오프 §5⑴ 의 격리 ① 원안은 *"무력한 컷(072·085·093)에서 요구를 배제 술어로 재진술하면
판정이 갈리는가"* 였다. **STEP 2(C535)가 그 전제를 깼다**: 그 컷들은 애초에 **제품을 고르는
자리가 아니다**(gold 가 전부 조회·정정·분쟁 접수) ⇒ 원안은 *없는 결손*을 잰다.

대신 STEP 2 가 연 자리를 잰다 — 사이드카 축자로 **제품 선택이 아닌 태스크에 제품 추천이
주입**되고 있었고(t7310 **ctl 8건 / treat 5건**), 그것은 `VERDICT_CARRY` 만의 것이 아니라
**검색-에이전트 축 자체**의 범위 문제다([[55]] 우리 층 먼저).

## 셀 (문맥은 라이브 축자 · 바뀌는 것은 주입 한 줄뿐)

    A_REF   결정 시점까지의 라이브 대화 축자 · 주입 **없음**          ← 기준선
    B_INJ   + 라이브가 실제로 넣은 줄 **축자**("It answers: X.")      ← 주입의 비용
    D_NEG   + 같은 형식 · **다른 제품 이름**(같은 군의 다른 후보)      ← 계기(이름을 읽나)

## 채점 (결정론 · gold 무참조 · [[23]])

  ⑴ `tool` — 방출된 tool_call 이름(디스패처면 **안쪽 이름까지**·접미사 일치·x370 규약)
  ⑵ `changed` — A_REF 와 다른 호출을 냈는가
  ⑶ `leaked` — **주입한 제품 이름이 그 호출의 인자에 등장**하는가(문자열 실재확인·정규식 0)

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    B 의 leak 이 과반 컷        → 주입이 **행동으로 샌다** = 비용 ⇒ 처방 = 발화 **범위 조건**
    changed 만 있고 leak 0      → 교란은 있으나 이름 유입 없음 [D] — 처방 보류
    A ≈ B (둘 다 같은 호출)     → 주입 무해 ⇒ 이 자리는 레버가 아니다(원인은 딴 데)
    D_NEG 도 같은 비율로 leak   → **주입 채널이 산다**(모델이 우리 줄을 읽는다)는 확증
    B·D 둘 다 leak 0 ∧ changed 0→ **계기 무효**(대화를 안 읽는다·결과 인용 금지)
    양성통제(055 컷)에서 B 가 제품을 말하지 못하면 → 계기 무효

⚠엔진 수정 0 · 오프라인 · 8141 전용([[30]] 포트 분리).
실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
      /home/woori/venvs/seka_env/bin/python x378_scope_injection_iso.py [태그 ...]
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                   # noqa: E402
import x370_infomatched_give_iso as G                     # noqa: E402
import x351_order_lever_iso as X                          # noqa: E402

MARK = "It answers:"
ASK = ("You are the bank's support agent. Continue this conversation and make the tool call "
       "you would make now.")
DEFAULT_TAGS = ["bank_t7310_ctl_20260818e", "bank_t7310_treat_20260818e",
                "bank_t7312_ctl_20260818g", "bank_t7312_treat_20260818g"]


def injections(tag):
    """사이드카에서 **실제로 주입된 줄**을 축자로 집는다(재구성 0 — 라이브가 쓴 문자열 그대로)."""
    out = []
    for r in F.sidecar_rows(tag):
        txt = str(r.get("text") or "")
        if MARK not in txt:
            continue
        i = txt.find(MARK)
        name = txt[i + len(MARK):].strip().rstrip(".").strip()
        if not name or len(name) > 60:
            continue
        out.append({"tag": tag, "simtag": str(r.get("simtag") or ""), "turn": r.get("turn"),
                    "text": " ".join(txt.split()), "name": name})
    return out


def candidates_of(name, po):
    """같은 군의 **다른 후보** 하나 — D_NEG 용. 출처 = A2 `doc_index`(env 파일명 유래)뿐."""
    idx = po.get("doc_index") or {}
    for g, keys in sorted(idx.items()):
        disp = [X.disp(k) for k in sorted(keys)]
        if name in disp:
            for d in disp:
                if d != name:
                    return g, d
    return "", ""


def convo_upto(sim, turn):
    """라이브 대화 축자 — 그 turn 직전까지(x370 `convo` 와 같은 규약·자르지 않는다)."""
    msgs = sim.get("messages") or []
    upto = len(msgs)
    for i, m in enumerate(msgs):
        ti = m.get("turn_idx")
        if ti is not None and int(ti) >= int(turn):
            upto = i
            break
    return G.convo(sim, upto)


def emitted(msg):
    """방출된 호출 → (이름, 인자문자열). 디스패처는 **안쪽 이름까지** 편다."""
    for tc in ((msg or {}).get("tool_calls") or ()):
        f = tc.get("function") or tc
        nm = str(f.get("name") or "")
        ar = str(f.get("arguments") or "")
        inner = ""
        try:
            if ar.strip().startswith("{"):
                inner = F.inner_name(json.loads(ar)) or ""
        except Exception:
            inner = ""
        return (("%s(%s)" % (nm, inner)) if inner else nm), ar
    return "", ""


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT_TAGS
    po = (X.a2_load().get("policy_ontology") or {})
    tools = G.agent_tool_specs()

    print("=" * 104)
    print("x378 · 범위 밖 주입 격리 · 태그 %s · 도구 스키마 %d개" % (",".join(tags), len(tools)))
    print("판정(사전 고정): B leak 과반 → 비용(처방=범위 조건) · changed만 → 보류[D] · "
          "A 와 같음 → 무해 · B·D 둘 다 무반응 → 계기 무효 · 055 양성통제 실패 → 계기 무효")
    print("=" * 104)

    cuts = []
    for tag in tags:
        sims = {F.simtag(s): s for s in F.scored(tag, ".results.json.gz")}
        for inj in injections(tag):
            sim = sims.get(inj["simtag"])
            if sim is None:
                print("  ⚠sim 없음(건너뜀): %s / %s" % (tag, inj["simtag"]))
                continue
            grp, other = candidates_of(inj["name"], po)
            if not other:
                print("  ⚠후보 군 미상(건너뜀): %r" % (inj["name"],))
                continue
            cuts.append(dict(inj, sim=sim, group=grp, other=other,
                             task=inj["simtag"].split("#")[0]))
    print("컷 %d개" % len(cuts))
    print("")

    rows = []
    for c in cuts:
        base = convo_upto(c["sim"], c["turn"])
        if not base:
            print("  ⚠대화 재구성 실패(건너뜀): %s %s" % (c["tag"], c["simtag"]))
            continue
        neg = c["text"].replace(c["name"], c["other"])
        got = {}
        for an, add in (("A_REF", ""), ("B_INJ", c["text"]), ("D_NEG", neg)):
            prompt = base + (("\n\ntool: " + add) if add else "") + "\n\n" + ASK
            msg, det = G.det(prompt, tools, 260)
            nm, ar = emitted(msg)
            look = c["other"] if an == "D_NEG" else c["name"]
            got[an] = {"tool": nm, "args": ar, "det": det,
                       "leak": int(bool(nm) and look in ar)}
        row = dict(task=c["task"], tag=c["tag"].split("_")[1],
                   arm=("treat" if "treat" in c["tag"] else "ctl"),
                   turn=c["turn"], name=c["name"], other=c["other"], got=got)
        row["changed_B"] = int(got["B_INJ"]["tool"] != got["A_REF"]["tool"])
        row["changed_D"] = int(got["D_NEG"]["tool"] != got["A_REF"]["tool"])
        rows.append(row)
        print("  %-9s %-6s %-5s turn=%-3s 주입=%-22s | A=%-30s B=%-30s D=%-30s | leak B/D=%d/%d %s"
              % (row["task"], row["tag"], row["arm"], row["turn"], c["name"],
                 got["A_REF"]["tool"] or "(없음)", got["B_INJ"]["tool"] or "(없음)",
                 got["D_NEG"]["tool"] or "(없음)", got["B_INJ"]["leak"], got["D_NEG"]["leak"],
                 "" if all(got[a]["det"] for a in got) else "⚠비결정"))

    if not rows:
        print("")
        print("⛔컷 0 — 계기 결함(결과 없음)")
        return 1
    n = len(rows)
    lb = sum(r["got"]["B_INJ"]["leak"] for r in rows)
    ld = sum(r["got"]["D_NEG"]["leak"] for r in rows)
    cb = sum(r["changed_B"] for r in rows)
    cd = sum(r["changed_D"] for r in rows)
    print("")
    print("## 집계  n=%d · leak B=%d D=%d · changed B=%d D=%d" % (n, lb, ld, cb, cd))
    if lb == 0 and cb == 0 and ld == 0 and cd == 0:
        v = "⛔**계기 무효** — 어느 팔도 반응하지 않는다(결과 인용 금지)"
    elif lb * 2 > n:
        v = "**주입이 행동으로 샌다 = 비용** ⇒ 처방 후보 = 발화 범위 조건"
    elif cb * 2 > n:
        v = "교란만 있고 이름 유입 없음 — **[D] 보류**"
    else:
        v = "A 와 같음 — 주입 무해(이 자리는 레버가 아니다)"
    print("판정: %s" % v)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "..", "reports", "facet_rft_2026",
                       "x378_scope_injection.json")
    io.open(os.path.normpath(out), "w", encoding="utf-8").write(
        json.dumps({"rows": rows, "n": n, "leak_B": lb, "leak_D": ld,
                    "changed_B": cb, "changed_D": cd, "verdict": v},
                   ensure_ascii=False, indent=1))
    print("원자료: %s" % os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
