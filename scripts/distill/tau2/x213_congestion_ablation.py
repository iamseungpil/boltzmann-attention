# -*- coding: utf-8 -*-
r"""x213 — 부하 = **혼잡**이다: 정보를 고정한 채 경쟁하는 방향만 걷어낸다 (유료 0 · 엔진 0).

## 관점 (사용자 지시 2026-08-10)

> *"부하는 단순히 길이 문제가 아니다. **모순·혼잡**의 문제다. 지시나 데이터가 정박하면서 다른
>  방향으로 steer 하는 거다. 격리는 이런 모순을 걷어내고 깨끗한 문맥으로 결론 짓는 거다.
>  그래서 관점은 **얼마나 일관된 깨끗한 데이터를 격리에 담는가** 이다."*

x212 는 길이를 부하의 대리변수로 놓아 틀렸다(85k 에서 0/6→3/6, 12k 에서 0/6→0/6 — 길이로는
일관되지 않는다). 오늘 실측은 전부 **경쟁하는 방향**을 가리킨다: 창 꼬리말이 자기 계산을 부정해
0/8(C393) · `NONE` 조항이 빈 `asked` 와 모순돼 침묵(C391) · x151 의 꼬리말 한 줄 · R8 의
*"결정 블록과 다른 지시를 섞지 말라"*.

## 혼잡의 실체 (가장 큰 실패 사례 실측)

  42 메시지 · 152,703자 · **KB 결과 8건에 문서 항목 80개**(답을 든 것은 **1개**) ·
  **이관 쪽으로 미는 메시지 12건** · 에이전트 자기-부정 0회.

## 팔 — **정보는 빼지 않고 혼잡만 뺀다**

  A_FULL     그 지점의 실제 문맥 그대로                          ← $p_{traj}$
  B_ONEDOC   KB 덤프에서 **답을 든 문서 항목만 남긴다**(79개 제거·정보 보존)
  C_NOXFER   이관 쪽으로 미는 문장을 지운다(정보 보존)
  D_BOTH     둘 다
  E_CLEAN    원장 + 정의만 (혼잡 0 = 천장)                       ← $p_{iso}$
  STRIP      정의를 지운 것 (부정 통제 · 0 이어야 한다)

⚠**정보 보존이 이 설계의 전부다.** B 는 답을 든 문서를 **남긴다**; 지우는 것은 경쟁 문서뿐이다.
  그래서 A→B 의 차이는 정보량이 아니라 **혼잡**이다([[18]] 정보-맞춘 격리).
⚠어느 팔에서도 이유를 말해 주지 않는다. 판정 지표는 **이유 진술**이고 이관은 관측이다.

실행: python x213_congestion_ablation.py [N] [--slice i/k]
"""
import collections
import glob
import gzip
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x200_disclaimer_ab import CAUSE, ESCAPE                      # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
SIG = "too many referral processes"
TARGET_DOC = "doc_credit_cards_credit_cards_(general)_001"
PATS = ["/home/woori/scratch/tau2-bench/data/simulations/*/results.json",
        "/home/woori/workspace_common/boltzmann-attention-pi/reports/facet_rft_2026/"
        "sim_results/*.json.gz"]
BUDGET = int(os.environ.get("T2_X213_BUDGET", "120000"))


def cases(limit=40):
    out, seen = [], set()
    for pat in PATS:
        for p in sorted(glob.glob(pat)):
            try:
                f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
                d = json.load(f)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for s in d.get("simulations") or []:
                if not isinstance(s, dict) or s.get("task_id") != "task_010":
                    continue
                if (s.get("reward_info") or {}).get("reward") == 1:
                    continue
                msgs = s.get("messages") or []
                blob = "\n".join(str(m.get("content") or "") for m in msgs)
                if SIG not in blob:
                    continue
                key = blob[:300]
                if key in seen:
                    continue
                seen.add(key)
                out.append((os.path.basename(os.path.dirname(p)) or os.path.basename(p),
                            s.get("trial"), msgs))
                if len(out) >= limit:
                    return out
    return out


def keep_only_target_doc(text):
    """KB 결과에서 **답을 든 문서 항목만** 남긴다 — 경쟁 문서만 지우고 정보는 보존한다.

    검색 결과는 `N. 제목 / ID: doc_… / Score: … / Content: …` 가 반복되는 형태다. 항목 경계로
    쪼개서 대상 ID 를 가진 덩어리만 남긴다. **엔진이 아니라 프로브가** 하는 절제이므로 [[59]] 와
    무관하다(측정용 절제이지 런타임 파싱이 아니다).
    """
    if "ID:" not in text:
        return text
    chunks = re.split(r"(?m)^(?=\s*\d+\.\s)", text)
    keep = [c for c in chunks if ("ID:" not in c) or (TARGET_DOC in c)]
    dropped = len(chunks) - len(keep)
    return ("\n".join(keep), dropped) if False else "\n".join(keep)


def render(msgs, upto, mode):
    parts, dropped_docs, dropped_x = [], 0, 0
    for m in msgs[:upto]:
        role = m.get("role")
        c = " ".join(str(m.get("content") or "").split())
        raw = str(m.get("content") or "")
        # ★검색 결과에만 적용한다 (x215 2차 자기점검). 구판은 `ID:` 만 보고 잘랐는데 **원장 출력도
        #   `1. Record ID:` 형태**라 원장 4행이 경쟁 항목으로 삭제됐다 — `B_ONEDOC` 이 늘 0 이던
        #   이유이고, *"정보는 빼지 않는다"* 는 이 설계의 전제를 스스로 어긴 것이다.
        #   검색 결과의 서명은 `Score:` 다.
        if mode in ("B", "D") and role == "tool" and "Score:" in raw and "ID:" in raw:
            before = len(re.findall(r"ID: doc_", raw))
            raw2 = keep_only_target_doc(raw)
            dropped_docs += before - len(re.findall(r"ID: doc_", raw2))
            c = " ".join(raw2.split())
        # ★정의나 원장을 든 메시지는 **지우지 않는다** — 지우면 정보 절제가 되어 통제가 깨진다.
        if (mode in ("C", "D") and re.search(r"transfer|human agent", c, re.I)
                and SIG not in c and "referral_status" not in c):
            dropped_x += 1
            continue
        for t in (m.get("tool_calls") or []):
            fn = t.get("function") or t
            a = fn.get("arguments")
            a = a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)
            parts.append("[%s calls %s(%s)]" % (role, fn.get("name"), a[:200]))
        if c:
            parts.append("[%s] %s" % (role, c))
    txt = "\n".join(parts)
    return (txt[-BUDGET:] if len(txt) > BUDGET else txt), dropped_docs, dropped_x


def _onto_block(led_rows):
    """**격리 서브에 실제로 실을 문맥** — 전부 A2/A3·엔진 산물이고 KB 는 0 이다.

    사용자 지시(2026-08-10): *"결정점에서의 정책 값들은 A2 A3 로 격리해서 서브에이전트에서
    정하게 하라. retrieval 사용하지 말라."* `E_CLEAN`(원장+정의)은 그 설계를 대표하지 못한다 —
    *"너무 많다"* 를 수로 뒷받침하는 **창 산수**가 빠져 있다(x210 에서 8/8 이 나온 팔에는 있었다).
    여기서는 그 셋을 다 싣는다: **상태별 세기 + 창 산수 + 상태 정의**.
    """
    import t2_ledger as _LG
    from gate_interpreter import load_domain_a2 as _L
    a2 = _L("banking_knowledge") or {}
    sp = (a2.get("ledger_metrics") or [{}])[0]
    a3 = (a2.get("policy_ontology") or {}).get("rows") or ()
    parts = [_LG.status_breakdown(led_rows, sp), _LG.window_history(led_rows, sp),
             _LG.status_meanings_text(led_rows, sp, a3)]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def _parse_led(text):
    """궤적의 원장 출력에서 **행을 복원**한다 (프로브 전용 — 엔진이 아니다).

    라이브에서는 이 전사를 **모델이** 한다(A2 `row_keys`·[[59]]). 여기서는 이미 지나간 궤적을
    재현할 뿐이라 프로브가 읽는다.
    """
    rows = []
    for chunk in re.split(r"(?=Record ID:)", text):
        st = re.search(r"referral_status: ([A-Z_]+)", chunk)
        if not st:
            continue
        ty = re.search(r"referred_account_type: (.+?) referral_status", chunk)
        dt = re.search(r"date: (\d{1,2}/\d{1,2}/\d{4})", chunk)
        rows.append({"referred_account_type": (ty.group(1).strip() if ty else ""),
                     "referral_status": st.group(1),
                     "date": (dt.group(1) if dt else "")})
    return rows


def _a3_line():
    """우리 층이 **A3 에서** 만드는 문장 — 검색 결과가 아니라 선언된 상수다(C395)."""
    import t2_ledger as _LG
    from gate_interpreter import load_domain_a2 as _L
    a2 = _L("banking_knowledge") or {}
    sp = (a2.get("ledger_metrics") or [{}])[0]
    rows = [{"referral_status": x} for x in ("COMPLETE", "IN_PROGRESS", "REJECTED")]
    return _LG.status_meanings_text(rows, sp,
                                    (a2.get("policy_ontology") or {}).get("rows") or ()).strip()


def referral_ledger(msgs):
    """**추천 원장**을 집는다 — 상태 필드가 있는 도구 출력만.

    ⚠**자기적발 (x215)**: 구판은 *"`Record ID` 가 들어간 첫 tool 메시지"* 를 집었는데, 이 도메인은
      사용자 조회도 `Found 1 record(s) in 'users': … Record ID:` 형태다. 전 사례에서 집힌 것이
      **users 조회**였고 `referral_status` 는 하나도 없었다 — 즉 `E_CLEAN` 은 판정에 필요한 4행을
      **한 번도 담지 않았다**. 그 팔을 천장이라 부른 것이 x213 1차의 근본 결함이다.
    """
    for m in msgs:
        c = str(m.get("content") or "")
        if m.get("role") == "tool" and "referral_status" in c and "Record ID" in c:
            return " ".join(c.split())
    return ""


def probe_point(msgs):
    """손님이 **이유를 되묻는** 지점. 없으면 **None** — 그 사례는 건너뛴다.

    ⚠**자기적발 (x215)**: 구판은 조건에 맞는 발화가 없으면 *마지막 user 턴*으로 떨어졌고, 그것이
      `###STOP###`·`###TRANSFER###` 같은 **제어 토큰**인 사례가 3/7 이었다. 제어 토큰을 질문으로
      쓴 셀은 아무것도 재지 않는다. 못 찾으면 **재지 않는 것**이 옳다.
    """
    for i, m in enumerate(msgs):
        if m.get("role") != "user":
            continue
        c = str(m.get("content") or "")
        if "###" in c or len(c.split()) < 6:
            continue
        lo = c.lower()
        if "transfer" in lo or "human agent" in lo:
            continue
        if ("why" in lo or "reason" in lo or "didn" in lo) and i > 2:
            return i
    return None


def ask(prompt, temp):
    body = {"model": MODEL, "temperature": temp, "max_tokens": 220,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return " ".join((json.load(r)["choices"][0]["message"]["content"] or "").split())


def main():
    n = 6
    part, parts = 0, 1
    for i, a in enumerate(sys.argv[1:]):
        if a.isdigit():
            n = int(a)
        if a.startswith("--slice"):
            v = a.split("=", 1)[-1] if "=" in a else sys.argv[i + 2]
            part, parts = (int(x) for x in v.split("/"))
    cs = cases()
    if parts > 1:
        cs = [c for i, c in enumerate(cs) if i % parts == part]
    print("사례 %d개 · n=%d · slice %d/%d" % (len(cs), n, part, parts))
    agg = collections.Counter()
    out = {}
    for tag, trial, msgs in cs:
        i = probe_point(msgs)
        if i is None:
            print("\n%s trial=%s — 이유를 되묻는 발화가 없다. 건너뛴다(제어 토큰을 질문으로 쓰지 않는다)."
                  % (tag, trial))
            continue
        led0 = referral_ledger(msgs)
        if not led0:
            print("\n%s trial=%s — 추천 원장(상태 필드 보유)을 못 찾았다. 건너뛴다." % (tag, trial))
            continue
        askmsg = " ".join(str(msgs[i].get("content") or "").split())
        built = {}
        for mode, name in (("A", "A_FULL"), ("B", "B_ONEDOC"), ("C", "C_NOXFER"), ("D", "D_BOTH")):
            t, dd, dx = render(msgs, i, mode)
            built[name] = (t, dd, dx)
        if SIG not in built["A_FULL"][0] or SIG not in built["B_ONEDOC"][0]:
            print("\n%s trial=%s — 절단/절제로 정의가 사라졌다. 건너뛴다(통제 오염)." % (tag, trial))
            continue
        clean = "[tool] " + led0 + "\n" + _a3_line()
        # ★G_ONTO — 격리 서브에 **실제로 실을** 문맥(원장 + 상태별세기 + 창 산수 + 정의).
        #   전부 A2/A3·엔진 산물이고 KB 0. `E_CLEAN` 은 창 산수가 빠져 이 설계를 대표하지 못한다.
        _lr = _parse_led(led0)
        _ob = _onto_block(_lr) if _lr else ""
        built["G_ONTO"] = (("[tool] " + led0 + "\n" + _ob) if _ob else clean, 0, 0)
        built["E_CLEAN"] = (clean, 0, 0)
        # ★F_A3 (C395) — 실제 문맥은 **그대로 두고** 우리 층이 A3 에서 만든 문장만 얹는다.
        #   E_CLEAN 은 문맥을 갈아 끼운 천장이라 라이브에서 할 수 없는 일이지만, F_A3 는
        #   라이브에서 **그대로 할 수 있는 일**이다. 인용 출처가 KB 검색이 아니라 **선언**이라
        #   대화마다 있다가 없다가 하지 않는다(x211: 질의 24개 중 12개만 그 문서를 냈다).
        built["F_A3"] = (built["A_FULL"][0] + "\n" + _a3_line(), 0, 0)
        strip = re.sub(r"[^\n]*too many referral processes[^\n]*", "[removed]", built["A_FULL"][0])
        built["STRIP"] = (strip, 0, 0)
        print("\n" + "=" * 92)
        print("%s trial=%s · 문맥 %d자 · 제거: 경쟁문서 %d · 이관문장 %d"
              % (tag, trial, len(built["A_FULL"][0]), built["B_ONEDOC"][1], built["C_NOXFER"][2]))
        print("  손님: %s" % askmsg[:120])
        # ★팔 자기점검 (x215 교훈) — **팔이 무엇을 담았는지 인쇄하지 않고는 재지 않는다.**
        #   1차 x213 은 `E_CLEAN` 이 users 조회를 담고 있었는데 그것을 천장이라 불렀다.
        bad = []
        for _nm in ("A_FULL", "B_ONEDOC", "C_NOXFER", "D_BOTH", "F_A3", "G_ONTO", "E_CLEAN", "STRIP"):
            _b = built[_nm][0]
            _nl, _nd = len(re.findall(r"referral_status", _b)), _b.count(SIG)
            print("   %-9s 원장행 %d · 정의 %d회 · %d자" % (_nm, _nl, _nd, len(_b)))
            # STRIP 만 정의가 없어야 하고, 나머지는 원장과 정의를 **둘 다** 들고 있어야 한다.
            if _nm == "STRIP":
                if _nl == 0 or _nd != 0:
                    bad.append(_nm)
            elif _nl == 0 or _nd == 0:
                bad.append(_nm)
        if bad:
            print("  ⚠팔 %s 이 정보를 잃었다 — 이 사례는 재지 않는다(절제가 정보 절제가 됐다)." % bad)
            continue
        for name in ("A_FULL", "B_ONEDOC", "C_NOXFER", "D_BOTH", "F_A3", "G_ONTO", "E_CLEAN", "STRIP"):
            body = built[name][0]
            c = collections.Counter()
            for k in range(n):
                p = (body + "\n\nThe customer now asks:\n" + askmsg
                     + "\n\nAnswer the customer in two or three sentences.")
                try:
                    t = ask(p, 0.0 if k == 0 else 0.7)
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                lo = t.lower()
                # ★채점 (x215 교훈) — **되읊기를 정답으로 세지 않는다.** 1차는 낱말만 봤는데
                #   `E_CLEAN` 에 우리가 넣은 정의 문장이 그 낱말을 담고 있어, 모델이 그것을
                #   그대로 옮기기만 해도 정답이 됐다. 이제 **거절된 상품을 짚고** 사유를 함께
                #   말할 때만 인정한다 — 정의 문장에는 상품명이 없으므로 메아리는 못 통과한다.
                ok = ("platinum" in lo) and any(x in lo for x in CAUSE)
                c["이유O" if ok else "이유X"] += 1
                c["이관O" if any(x in lo for x in ESCAPE) else "이관X"] += 1
            out["%s/%s/%s" % (tag, trial, name)] = [c["이유O"], n]
            agg[name + "/h"] += c["이유O"]
            agg[name + "/n"] += n
            print("  %-9s 이유 %d/%d · 이관 %d/%d  (%d자)"
                  % (name, c["이유O"], n, c["이관O"], n, len(body)))
    print("\n" + "=" * 92)
    print("합계 — 판정 지표 = 이유 진술")
    for name in ("A_FULL", "B_ONEDOC", "C_NOXFER", "D_BOTH", "F_A3", "G_ONTO", "E_CLEAN", "STRIP"):
        if agg[name + "/n"]:
            print("  %-9s %3d/%-3d (%.0f%%)" % (name, agg[name + "/h"], agg[name + "/n"],
                                                100.0 * agg[name + "/h"] / agg[name + "/n"]))
    json.dump(dict(out, _agg=dict(agg)),
              open(os.environ.get("T2_X213_OUT", "x213_out.json"), "w"), indent=1)
    print("\n※ B 나 C 가 A 를 크게 올리면 → 부하는 **혼잡**이고, 무엇을 빼야 하는지까지 말해 준다."
          "\n  A≈B≈C≈E 면 → 혼잡이 아니고 다른 것이다. STRIP 이 A 와 같으면 정의는 안 읽히고 있었다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
