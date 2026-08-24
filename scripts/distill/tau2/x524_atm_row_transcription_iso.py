# -*- coding: utf-8 -*-
r"""x524 — **전사 서브가 원장 행을 빠뜨리는가**를 격리로 잰다 (2026-08-24·무료·사용자 지시)

## 왜 (관측)
t7348 `task_074` 두 trial 모두 계좌별 fee_refund 금액에서만 갈려 reward 0.0 이다. per-step
포렌식(`tasks__20260824/TASK_074.md`)이 결손을 **산수 op 가 아니라 서브가 넘긴 operand** 로
지목했다 — 로그 축자:

    [sim=task_074#s373753] [T2_SG_ISOLATE] operand-size
      get_atm_fee_discrepancies.transactions: sub=14 rows · source=30 rows  MISMATCH

그리고 원장 실측(이 프로브가 재확인한다): chk_2 원장의 `atm_withdrawal` = **16건**인데 서브는
**14행**을 넘겼다. A2 계약 축자는 `Include EVERY atm_withdrawal` 이다.

## 무엇을 가르나 ([[76]] 진단 순서 (1) — *서브가 무엇을 받았나*)
서브가 **같은 원장을 앞에 두고도** 14행을 내면 결손은 서브의 전사이고, 16행을 내면 라이브에서
서브가 **다른 것을 받은 것**(fetch 라운드·절단·중간 요약)이다. 둘은 처방이 다르다.

## 팔 (한 번에 하나만 민다)
    A_live    A2 `isolate.instructions` + `params.transactions` 축자 + 라이브 원장 원문
    B_count   A + 닫힌 사실 한 줄(`이 계좌의 atm_withdrawal 레코드는 N건이다`)
              -> 세어 주면 닫히는가 = 결손이 파싱이 아니라 **세기**인가
    N_neg     같은 지시 + **원장을 같은 길이의 무의미 텍스트로 대체** (부정통제·[[57]])

## 채점 (닫힌 술어만·[[23]] gold 미접촉)
    rows        서브가 낸 배열 길이
    ids_ok      낸 `transaction_id` 가 **원장에 실재**하는가 (substring)
    missing     원장 `atm_withdrawal` 중 서브 산출에 대응이 없는 것
    dup_of      `duplicate_of` 를 단 행 수 (A2 가 요구하는 정상 형태)
  gold 금액은 보지 않는다 — 이 프로브는 **행 집합**만 잰다.

사용: (리모트·cwd=scripts/distill/tau2) py -3 x524_atm_row_transcription_iso.py --port 8141 --n 6
"""
import argparse
import gzip
import io
import json
import os
import re
import sys
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
SIMS = os.path.join(REP, "sim_results")
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TOOL = "get_atm_fee_discrepancies"


def a2_decl():
    """A2 정본에서 이 도구의 선언을 읽는다 — 프롬프트 문면을 이 파일에 박지 않는다([[03b]])."""
    p = os.path.join(HERE, "a2", "banking_knowledge.specific.json")
    with io.open(p, encoding="utf-8") as f:
        d = json.load(f)
    for t in (d.get("scaffold_get_tools") or []):
        if t.get("name") == TOOL:
            return t
    raise SystemExit("A2 에 %s 선언이 없다 — 경로 %s" % (TOOL, p))


def ledgers(gz, task, seed):
    """라이브 궤적에서 계좌별 원장 원문을 뽑는다. 반환 = [(msg_idx, 본문)] (atm_withdrawal 보유분)."""
    d = json.load(gzip.open(gz, "rt", encoding="utf-8"))
    sims = [x for x in d.get("simulations", []) if x.get("task_id") == task]
    if seed is not None:
        sims = [x for x in sims if str(x.get("seed")) == str(seed)]
    if not sims:
        raise SystemExit("sim 없음: %s %s" % (task, seed))
    out, seen = [], set()
    for i, m in enumerate(sims[0].get("messages") or []):
        if m.get("role") != "tool":
            continue
        c = str(m.get("content") or "")
        if c.count("atm_withdrawal") < 5:
            continue
        h = hash(c)
        if h in seen:
            continue
        seen.add(h)
        out.append((i, c))
    return out


RE_ROW = re.compile(r"^\s*transaction_id:\s*(\S+)\s*$", re.M)
RE_TYPE = re.compile(r"^\s*type:\s*(\S+)\s*$", re.M)


def truth(text):
    """원장에서 닫힌 술어로 진실값을 센다: `type: atm_withdrawal` 레코드의 transaction_id 집합.

    ⚠형식은 env 가 정한다 — 번호 매긴 레코드 블록(`N. Record ID: …` + `key: value` 줄들).
      2026-08-24 실측으로 확인하고 맞췄다(JSON 가정은 틀렸다·[[67]] 계기를 믿지 마라).
    """
    ids = []
    for blob in re.split(r"\n(?=\s*\d+\.\s+Record ID:)", text):
        ty = RE_TYPE.search(blob)
        if not ty or ty.group(1) != "atm_withdrawal":
            continue
        m = RE_ROW.search(blob)
        if m:
            ids.append(m.group(1))
    return ids


def chat(port, messages, maxtok=3000):
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok, "messages": messages}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"].get("content") or ""


def parse_rows(txt):
    """서브 산출에서 transactions 배열을 뽑는다(형식 관용 — 판단 0)."""
    t = str(txt or "")
    i = t.find("[")
    j = t.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        v = json.loads(t[i:j + 1])
        return v if isinstance(v, list) else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--gz", default=os.path.join(SIMS, "bank_t7348_halfB_20260824.results.json.gz"))
    ap.add_argument("--task", default="task_074")
    ap.add_argument("--seed", default="373753")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--arms", default="A_live,B_count,N_neg")
    ap.add_argument("--out", default=os.path.join(REP, "x524_atm_row_transcription_2026_08_24.json"))
    a = ap.parse_args()

    decl = a2_decl()
    iso = decl.get("isolate") or {}
    instr = iso.get("instructions") or ""
    params = ((decl.get("params") or {}).get("transactions")) or ""
    if not instr or not params:
        raise SystemExit("A2 선언에 instructions/params.transactions 가 없다 — 프로브 중단")

    leds = ledgers(a.gz, a.task, a.seed)
    print("[x524] 원장 %d편 (msg idx %s)" % (len(leds), [i for i, _ in leds]))

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    rows = []
    for idx, text in leds:
        tru = truth(text)
        print("[x524] --- 원장 msg[%d] len=%d · atm_withdrawal %d건" % (idx, len(text), len(tru)))
        for arm in arms:
            mat = text
            extra = ""
            if arm == "B_count":
                extra = ("\n\nThis account has %d atm_withdrawal records in the history above.\n"
                         % len(tru))
            elif arm == "N_neg":
                mat = ("record " * (len(text) // 7))[:len(text)]
            prompt = (instr + "\n\n# Field contract\ntransactions: " + params +
                      extra + "\n\n# Account transaction history\n" + mat +
                      "\n\nReply with ONE JSON array only: the `transactions` value.")
            for k in range(a.n):
                try:
                    out = chat(a.port, [{"role": "user", "content": prompt}])
                except Exception as e:
                    rows.append({"msg": idx, "arm": arm, "k": k, "error": str(e)[:200]})
                    print("      %-8s k=%d ERROR %r" % (arm, k, str(e)[:80]))
                    continue
                got = parse_rows(out)
                ids = [str((r or {}).get("transaction_id", "")) for r in (got or [])]
                ok = [i2 for i2 in ids if i2 and i2 in text]
                miss = [t2 for t2 in tru if t2 not in ids]
                dup = sum(1 for r in (got or []) if isinstance(r, dict) and r.get("duplicate_of"))
                rows.append({"msg": idx, "arm": arm, "k": k, "truth": len(tru),
                             "rows": (len(got) if got is not None else None),
                             "ids_in_ledger": len(ok), "missing": miss, "dup_of": dup,
                             "raw_head": str(out)[:200]})
                print("      %-8s k=%d rows=%s/%d ids_ok=%d dup_of=%d missing=%s"
                      % (arm, k, (len(got) if got is not None else "parse_fail"),
                         len(tru), len(ok), dup, miss[:4]))

    with io.open(a.out, "w", encoding="utf-8") as f:
        json.dump({"probe": "x524", "date": "2026-08-24",
                   "question": "서브가 같은 원장을 앞에 두고도 행을 빠뜨리는가",
                   "source": {"gz": os.path.basename(a.gz), "task": a.task, "seed": a.seed},
                   "arms": arms, "n": a.n, "rows": rows}, f, ensure_ascii=False, indent=1)
    print("[x524] wrote %s (%d행)" % (a.out, len(rows)))

    print("\n[x524] 요약 (arm · 원장별 평균 행수 / 진실값)")
    for arm in arms:
        for idx, text in leds:
            rs = [r for r in rows if r["arm"] == arm and r["msg"] == idx and r.get("rows") is not None]
            if not rs:
                continue
            tru = rs[0]["truth"]
            exact = sum(1 for r in rs if r["rows"] == tru and not r["missing"])
            print("  %-8s msg[%2d] truth=%2d · rows %s · 정확 %d/%d"
                  % (arm, idx, tru, [r["rows"] for r in rs], exact, len(rs)))


if __name__ == "__main__":
    main()
