# -*- coding: utf-8 -*-
r"""x450 — **격리에서 되던 것이 라이브에서 안 되는 이유**를 차이 하나씩 떼어 잰다 (2026-08-21·무료)

## 관측 (사용자 물음: *"왜 격리에서 되던게 스모크에서는 안되는가?"*)
격리 `x448` `B_shell` 은 task_024(트럭)에서 `operations` 를 **24/24 철회**했다(C576).
라이브 첫 발화는 축자로 `spend_category: 'operations' -> 'operations' (인용 실재)` — **유지**했다.

## [[55]] 순서대로 — 우리 층에서 찾은 차이 넷 (전부 **우리가 만든 것**이다)
    ⒜ 제목 손실   격리 재료 = 샌드박스 `cat <id>.md` = **`# 제목` + 본문**
                  라이브 재료 = `t2_search.read_docs(corpus=…)` = **본문만**(`{id: 본문}`)
                  ⇒ 판정 문서의 제목 *"…What Qualifies as Operations Spend?"* 가 통째로 사라진다
    ⒝ 소문자화    `_evidence_ctx` 가 `" ".join(users).lower()` 로 손님 발화를 **뭉갠다**
    ⒞ 메시지 구조 격리 = system + user 두 개 · 라이브 `sub_generate` = **단일 user 메시지**
    ⒟ 문구        격리 = *"which documented spending category the customer's spending belongs to"*
                  라이브 = *"Decide ONE thing: the value of `spend_category`"*
  ⚠온도는 여기서 통일한다(둘 다 0.0). 라이브가 에이전트 기본 온도를 쓰는 것은 **별건**이고
    이 프로브가 원인을 못 찾으면 그때 본다.

## 팔 (한 번에 **하나만** 라이브 쪽으로 민다)
    A_iso        격리 그대로 (제목 O · 원문 대소문자 · system+user · 격리 문구)
    B_live       라이브 그대로 (제목 X · 소문자 · 단일 user · 라이브 문구)
    C_title      B + **제목만** 복원
    D_case       B + **대소문자만** 복원
    E_sys        B + **system+user 분리와 격리 문구를 함께** 복원
    F_word       B + **격리 문구만**(구조는 라이브 = 단일 user 메시지)
    G_split      B + **구조만** 분리(문구는 라이브 그대로)
    H_front      B + **위치만** 앞으로(단일 user·라이브 문구 그대로)
  ⇒ B 가 A 와 갈리고 C/D/E 중 하나가 A 로 돌아오면 그것이 원인이다.

## 채점 (닫힌 술어만)
    correct  참조 라벨 일치 — task_024 는 **None**(gold `Business Bronze Rewards Card`)
    quote_real  인용이 그 팔이 받은 재료 안에 실재하는가(형태만 정규화)

사용: (리모트·cwd=tau2 · PYTHONPATH=src:…) py x450_iso_vs_live_bisect.py --port 8141
"""
import argparse
import io
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import x447_indexed_category_iso as IX   # noqa: E402  A2 선언 읽기(사본 금지·[[67]])
import x448_index_vs_all_iso as V        # noqa: E402  사례·정규화·샌드박스(사본 금지·[[67]])
import x430_account_facts as FT          # noqa: E402  DOCDIR

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"

ISO_SYS = V.SYS          # 격리가 쓴 그 문구 그대로(사본 금지)


def chat(port, messages, maxtok=400):
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok, "messages": messages}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"].get("content") or ""


def parse(txt):
    i, j = str(txt or "").find("{"), str(txt or "").rfind("}")
    try:
        return json.loads(txt[i:j + 1]) if i >= 0 and j > i else {}
    except Exception:
        return {}


def declared():
    """A2 선언 → [(id, 제목, 본문)]. 제목은 **문서 파일이 들고 있는 것**(샌드박스와 같은 원천)."""
    out = []
    for did, title, body in IX.index_docs():
        out.append((did, title, body))
    return out


def live_prompt(mat, said, arg, vals):
    """엔진이 실제로 만드는 문자열과 **같은 모양**(t2_scaffold_get 배달 블록 축자 복제)."""
    return ("# Documents\n%s\n\n# What the customer said\n%s\n\n"
            "Decide ONE thing: the value of `%s`. Reply with ONE JSON object "
            "only: {\"%s\": <one of: %s> or null, \"quote\": \"<one sentence "
            "copied word for word from the '# Documents' section that shows "
            "this>\"}. The quote MUST come from the documents, never from the "
            "customer. If no document sentence supports a value, set it to null."
            % (mat, said, arg, arg, ", ".join(vals)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="bis1")
    ap.add_argument("--task", default="task_024")
    ap.add_argument("--arms", default="A_iso,B_live,C_title,D_case,E_sys,F_word,G_split,H_front")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    ds = declared()
    with io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8") as f:
        decl = ((json.load(f).get("catalog_arg_docs") or {}).get("spend_category") or {})
    vals = [k for k in decl if k[:1] != "_"]

    mat_title = "\n\n".join("# %s\n\n%s" % (t, b) for _i, t, b in ds)      # 샌드박스 cat 과 같은 꼴
    mat_plain = "\n\n".join("### %s\n%s" % (i, b) for i, _t, b in ds)      # 엔진이 지금 주는 꼴
    mat_id_ti = "\n\n".join("### %s — %s\n%s" % (i, t, b) for i, t, b in ds)

    cases = [c for c in V.wide_cases() if c["task"] == a.task]
    ref = V.REF.get(a.task.split("_")[-1], "?")
    print("=" * 100)
    print("x450 · %s · 사례 %d · 참조=%r · 선언 %d편" % (a.task, len(cases), ref, len(ds)))
    print("   제목 있는 재료 %d자 / 제목 없는 재료 %d자" % (len(mat_title), len(mat_plain)))
    print("=" * 100)

    rows = []
    for c in cases:
        said_raw = c["said"]
        said_low = said_raw.lower()
        for arm in arms:
            if arm == "A_iso":
                mat, said = mat_title, said_raw
                msgs = [{"role": "system", "content": ISO_SYS},
                        {"role": "user", "content": "# Documents\n%s\n\n# What the customer said\n%s\n"
                         % (mat, said)}]
            elif arm == "B_live":
                mat, said = mat_plain, said_low
                msgs = [{"role": "user", "content": live_prompt(mat, said, "spend_category", vals)}]
            elif arm == "C_title":
                mat, said = mat_id_ti, said_low
                msgs = [{"role": "user", "content": live_prompt(mat, said, "spend_category", vals)}]
            elif arm == "D_case":
                mat, said = mat_plain, said_raw
                msgs = [{"role": "user", "content": live_prompt(mat, said, "spend_category", vals)}]
            elif arm == "E_sys":
                mat, said = mat_plain, said_low
                msgs = [{"role": "system", "content": ISO_SYS},
                        {"role": "user", "content": "# Documents\n%s\n\n# What the customer said\n%s\n"
                         % (mat, said)}]
            elif arm == "F_word":
                # 격리 **문구만** 쓰고 구조는 라이브 그대로(단일 user 메시지)
                mat, said = mat_plain, said_low
                msgs = [{"role": "user", "content": "%s\n\n# Documents\n%s\n\n"
                                                    "# What the customer said\n%s\n"
                         % (ISO_SYS, mat, said)}]
            elif arm == "G_split":
                # **구조만** 분리하고 문구는 라이브 그대로
                mat, said = mat_plain, said_low
                _lp = live_prompt(mat, said, "spend_category", vals)
                _cut = _lp.index("Decide ONE thing:")
                msgs = [{"role": "system", "content": _lp[_cut:]},
                        {"role": "user", "content": _lp[:_cut]}]
            else:  # H_front — **위치만** 앞으로(단일 user·라이브 문구 그대로)
                #   F/G 는 문구와 위치를 함께 옮긴다. 라이브 지시는 문서 15,000자 **뒤**에 있다 —
                #   그 자체가 원인일 수 있어 위치만 떼어 본다.
                mat, said = mat_plain, said_low
                _lp = live_prompt(mat, said, "spend_category", vals)
                _cut = _lp.index("Decide ONE thing:")
                msgs = [{"role": "user", "content": _lp[_cut:] + "\n\n" + _lp[:_cut]}]
            ans = parse(chat(a.port, msgs))
            cat = V.as_cat(ans.get("spend_category"))
            q = V.form_norm(ans.get("quote"))
            real = bool(q) and q in V.form_norm(mat)
            rows.append({"task": c["task"], "trial": c["trial"], "arm": arm, "cat": cat,
                         "quote": str(ans.get("quote") or "")[:200], "quote_real": real,
                         "correct": cat == ref})
        line = "  t%-4s " % c["trial"]
        for arm in arms:
            r = [x for x in rows if x["arm"] == arm and x["trial"] == c["trial"]][-1]
            line += "%s=%-13s " % (arm, r["cat"])
        print(line)

    p = os.path.abspath(os.path.join(REP, "x450_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n" + "=" * 100)
    print("%-10s %-10s %-10s" % ("팔", "참조일치", "인용실재"))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        print("%-10s %-10s %-10s"
              % (arm, "%d/%d" % (sum(1 for r in rs if r["correct"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["quote_real"]), len(rs))))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
