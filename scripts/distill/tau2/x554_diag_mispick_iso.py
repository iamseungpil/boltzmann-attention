# -*- coding: utf-8 -*-
r"""x554 — P1 선행 측정: `T2_DIAG` 오지목(4/4)이 **선언 한 칸**으로 갈리나

## 무엇을 재나 (사용자 지시 2026-08-26 ② · 정본 `tasks__20260824/TASK_016.md` §8 P1)
016 에서 **가장 많은 스텝을 지배한 것이 P1** 이다(문서 §8 축자). 격리 진단 서브가 4/4 로
`Platinum Rewards Card`(ERROR·2025-10-05)를 고르고, 그 답이 *"A separate check was run …"*
라는 **권위 문면**으로 배달돼 뒤따르는 assistant 발화 6개를 그 축에 묶는다. 손님이 묻는 것은
**most recent** 추천(Silver·2025-11-13·IN_PROGRESS)이다.

문서가 처방 둘을 적고 **격리 프로브 필수**로 걸어 뒀다:
  ⓐ 격리 문맥에 **선언된 필드 축**(`date_field`) 하나를 싣는다
  ⓑ 프롬프트의 **거짓 유일성**(*"One of these records did not pay out"*)을 고친다

## 재료 — 라이브 프롬프트를 **축자로 포획**해서 쓴다([[78]]·[[76]] ①)
프롬프트를 새로 쓰지 않는다. 서브콜 사이드카(`fb_<tag>.jsonl.gz` · `call_name=diagnose_formalize`)
가 **라이브가 실제로 보낸 2,190자**를 담고 있다(`t2_subcall._record_subcall`·2026-08-24 신설).
그것을 A_asis 로 쓰고, 선언 템플릿(`ledger_metrics[0].diagnose_prompt`)의 고정부를 벗겨
`{block}` 을 **정확히** 되찾은 뒤, 팔마다 **선언 한 칸**만 바꾼다. 스크립트에 재료 리터럴 0.

    A_asis   선언 그대로                      ← 라이브. **여기서 오지목이 재현돼야 판정한다**
    B_uniq   ⓑ 거짓 유일성 제거 — *"여럿일 수 있다 · 전부 대라"*
    B_field  ⓐ 선언된 `date_field` 축 한 줄 — *"가장 최근 것"*(어느 행인지는 서브가 고른다)
    N_len    부정 통제([[57]]) — B_field 와 **길이만 맞춘** 선택 무관 문장

★1차 결과(2026-08-26·8140·아래 넷): **ⓐ 기각 0/5**(부정 통제와 동일) · ⓑ 는 `picked` 0/5 인데
`named` **5/5**. 넷이 다 표적을 못 집자 블록을 다시 읽었고 **결손이 문면이 아니라 재료**임을
찾았다 — 블록은 같은 행에서 두 조각으로 나가는데 (상태→이름들)과 (이름→날짜)뿐이라
**행 하나의 날짜와 상태를 잇는 줄이 없다**. 그래서 재료 축 두 팔을 덧댔다([[78]]):

    B_join     행마다 **날짜와 상태를 한 줄에** (행 순서 그대로·정렬 0)
    N_joinlen  같은 줄에서 **상태만 뺀** 부정 통제 — 부피 같고 새 정보 0

## ⛔판정 규율
- **A_asis 가 라이브 오지목을 재현 못 하면 판정하지 않는다**([[62]] 2b · 핸드오프 §2 P1).
- 채점은 닫힌 술어뿐이고 gold 는 열지 않는다([[23]]). 표적 이름은 **우리가 만든 블록**의 날짜
  산수 줄에서 최신 날짜를 집어 온다(도메인 텍스트 파싱이 아니라 **자기 출력 파싱**·[[59]] 허용역).
- N_len 이 처치를 이기면 그 처치는 **길이 효과**다 — 그대로 인쇄한다.

## [[62]] 4문
  ① 결손은 라이브 4/4 실측(문서 §4 step 17′·22·42) + 포획된 3/3 동일 문자열.
  ② 재료가 닿으면 쓰는가를 이 프로브가 A/B 로 확정한다 — 레버는 **전달 문면**뿐.
  ③ 사라지는 모델 판단 0 — 어느 행이 최신인지·무엇을 답할지는 끝까지 서브가 고른다
     (엔진은 정렬도 argmax 도 하지 않는다).
  ④ 순위·최댓값·*"정답은 X"* 문장 0.

⚠B_field 가 이기면 **라이브 배선은 이 프로브가 아니다**: 손님 제약(*"most recent"*)을 축으로
  꺼내는 것은 LLM formalize 의 몫이고([[66]] 의도 입법 금지), 이 프로브는 그 축이 도달했을 때
  결손이 닫히는지만 잰다. 배선 설계는 측정 뒤에.

사용: (리모트·cwd=scripts/distill/tau2) py -3 x554_diag_mispick_iso.py --port 8140
      --wiring-only 로 **모델 없이** 프롬프트 복원·diff 만 돌 수 있다(무료).
"""

import argparse
import glob
import gzip
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

SIMS = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
DOMAIN = "banking_knowledge"
NL = chr(10)

# 블록의 날짜 산수 줄은 **우리가 만든 문자열**이다(`t2_ledger.window_history`).
RE_DATED = re.compile(r"([^;:.]+?) \((\d{4}-\d{2}-\d{2})\):")


def gen(port, body, maxtok=64, temp=0.0):
    payload = {"model": MODEL, "temperature": temp, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def captured(tag, task):
    """서브콜 사이드카에서 **라이브가 보낸 진단 프롬프트**를 축자로 꺼낸다.

    반환 [(simtag, prompt, out_head)…]. 없으면 빈 리스트 — 없는 것을 지어내지 않는다([[25]]).
    """
    out = []
    pats = [os.path.join(SIMS, "fb_%s.jsonl.gz" % tag)] if tag else \
        sorted(glob.glob(os.path.join(SIMS, "fb_*.jsonl.gz")), key=os.path.getmtime,
               reverse=True)
    for p in pats:
        if not os.path.exists(p):
            continue
        try:
            fh = gzip.open(p, "rt", encoding="utf-8", errors="replace")
        except Exception:
            continue
        with fh:
            for ln in fh:
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                if d.get("call_name") != "diagnose_formalize":
                    continue
                st = str(d.get("simtag") or "")
                if task and not st.startswith(task):
                    continue
                if d.get("text"):
                    out.append((os.path.basename(p)[3:-9], st,
                                str(d["text"]), str(d.get("out_head") or "")))
        if out and tag:
            break
    return out


def rows_from_traj(tag, simtag, keys):
    """영속 궤적의 도구 출력 → **선언된 키만** 담은 행들(`spec['row_keys']` 구동).

    ★왜 여기서 다시 만드나 — 라이브 엔진은 이 행들을 이미 갖고 있다(`_t2_ledger_ops[...]["rows"]`
      · `formalize_rows` 가 만든다). 블록 두 조각이 **그 행들에서** 나오는데도 날짜와 상태가
      **서로 다른 줄로 갈라져** 나간다. 이 프로브는 그 갈라짐이 결손인지 보려고 행을 되살릴 뿐이고,
      라이브 배선에는 파싱이 **필요 없다**(행이 이미 있다). 키는 전부 선언에서 온다 — 리터럴 0.
    """
    import t2_forensic as F
    files = {F.tag_of_file(q): q for q in F.all_result_files()}
    if tag not in files:
        return []
    sim = next((x for x in F.sims(files[tag]) if F.simtag(x) == simtag), None)
    if sim is None:
        return []
    for m in (sim.get("messages") or []):
        if m.get("role") != "tool":
            continue
        txt = str(m.get("content") or "")
        if not all(("%s:" % k) in txt for k in keys):
            continue
        out = []
        for blk in re.split(r"\n\s*\n", txt):
            row = {}
            for k in keys:
                mm = re.search(r"(?m)^\s*%s:\s*(.+?)\s*$" % re.escape(k), blk)
                if mm:
                    row[k] = mm.group(1)
            if len(row) == len(keys):
                out.append(row)
        if out:
            return out
    return []


def join_lines(rows, spec, with_status=True):
    """한 줄에 **날짜와 상태를 함께** 둔 목록. 정렬·순위·argmax 0 — 행 순서 그대로."""
    g, d, st = spec.get("group_field"), spec.get("date_field"), spec.get("status_field")
    part = []
    for r in rows:
        part.append("%s (%s)%s" % (r.get(g), r.get(d),
                                   (": %s" % r.get(st)) if with_status else ""))
    return "; ".join(part)


def recover_block(tpl, prompt):
    """포획된 프롬프트에서 `{block}` 을 **정확히** 되찾는다. 못 되찾으면 None."""
    pre, _, post = tpl.partition("{block}")
    if not (prompt.startswith(pre) and prompt.endswith(post.rstrip()) or
            prompt.startswith(pre)):
        return None
    body = prompt[len(pre):]
    tail = post.rstrip()
    if tail and body.endswith(tail):
        body = body[:-len(tail)]
    return body


def arms(tpl):
    """팔 = **선언 한 칸**(`diagnose_prompt`)의 교체. 재료는 손대지 않는다."""
    ask = tpl.partition("{block}")[2]
    head = tpl.partition("{block}")[0] + "{block}"
    uniq = (NL + NL + "Some of these records did not pay out - there may be more than one. "
            "Reply with the account type of every record that did not pay out, exactly as "
            "written above, and one short sentence saying what the definitions above make "
            "of them. Nothing else." + NL)
    field = (NL + NL + "One of these records did not pay out. The customer is asking about "
             "the most recent one, so read the date field above and answer for the latest "
             "of them. Reply with that record's account type exactly as written above, and "
             "one short sentence saying what the definitions above make of it. Nothing else."
             + NL)
    # 부정 통제: B_field 와 **글자 수를 맞추되** 어느 행을 고를지에 대한 정보는 0
    neg = (NL + NL + "One of these records did not pay out. The records above were compiled "
           "from the system of record earlier today, so treat the fields as final and do no "
           "further arithmetic. Reply with that record's account type exactly as written "
           "above, and one short sentence saying what the definitions above make of it. "
           "Nothing else." + NL)
    return [("A_asis", tpl), ("B_uniq", head + uniq),
            ("B_field", head + field), ("N_len", head + neg)], ask


def names_and_target(block):
    """블록의 날짜 산수 줄 → (이름 집합, 최신 날짜의 이름). **우리 출력만** 읽는다."""
    hits = [(n.strip().lstrip(".;").strip(), d) for n, d in RE_DATED.findall(block)]
    if not hits:
        return [], None
    names = []
    for n, _ in hits:
        if n and n not in names:
            names.append(n)
    return names, max(hits, key=lambda h: h[1])[0]


def picked(ans, names):
    """`diagnose_choice` 와 **같은 규칙** — 원장 이름 중 답에 든 것, 긴 것 우선."""
    hit = sorted((g for g in names if g and g.lower() in str(ans or "").lower()),
                 key=len, reverse=True)
    return hit[0] if hit else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--tag", default="bank_t7356_grpB3_20260826")
    ap.add_argument("--task", default="task_016")
    ap.add_argument("--n", type=int, default=4, help="temp>0 반복 수(견고성 팔)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--wiring-only", action="store_true")
    a = ap.parse_args(argv)

    import gate_interpreter as GI
    a2 = GI.load_domain_a2(DOMAIN) or {}
    spec = next((s for s in (a2.get("ledger_metrics") or []) if s.get("diagnose_prompt")), None)
    if not spec:
        print("선언에 `diagnose_prompt` 가 없다 — 잴 것이 없다", file=sys.stderr)
        return 2
    tpl = spec["diagnose_prompt"]

    caps = captured(a.tag, a.task)
    if not caps:
        print("포획된 진단 프롬프트가 없다 (fb_%s / %s) — 지어내지 않는다([[25]])"
              % (a.tag, a.task), file=sys.stderr)
        return 2
    live = {}
    for tg, st, txt, out in caps:
        live.setdefault(txt, []).append((tg, st, out))

    print("# x554 — P1(`T2_DIAG` 오지목) 격리 A/B")
    print("포획 %d건 · **서로 다른 프롬프트 %d종** (같으면 라이브가 한 문자열을 반복한 것)"
          % (len(caps), len(live)))
    for txt, who in live.items():
        outs = sorted({o for _, _, o in who})
        print("  · %d자 · sim %s" % (len(txt), ", ".join(sorted({s for _, s, _ in who}))))
        for o in outs:
            print("    라이브 답: %r" % o)
    prompt = max(live, key=lambda t: len(live[t]))
    block = recover_block(tpl, prompt)
    print()
    print("## [[78]] 프롬프트 diff — 선언 템플릿 ↔ 포획본")
    if block is None:
        print("  ⛔되찾기 실패 — 선언 고정부가 포획본과 다르다. 판정하지 않는다.")
        return 3
    rebuilt = tpl.format(block=block)
    print("  블록 %d자 되찾음 · 재조립 == 포획본: **%s**"
          % (len(block), "예" if rebuilt == prompt else "아니오"))
    if rebuilt != prompt:
        print("  ⛔불일치 — 계기가 라이브와 다른 것을 재고 있다. 판정하지 않는다([[78]]).")
        return 3

    names, target = names_and_target(block)
    print()
    print("## 채점 술어 (gold 미접촉·[[23]])")
    print("  원장 이름 %d: %s" % (len(names), ", ".join(names)))
    print("  표적 = 블록의 **최신 날짜** 행의 이름 → %r" % target)
    print("  ⚠이 값은 우리가 만든 날짜 산수 줄에서 왔다. gold 는 열지 않았다.")
    if not (names and target):
        print("  ⛔이름·표적을 못 세웠다 — 판정하지 않는다.")
        return 3

    plan, _ask = arms(tpl)
    plan = [(nm, t, block) for nm, t in plan]

    # ★재료 축 두 팔 (2026-08-26 · 선언 네 칸으로는 안 갈린 뒤에 붙였다·[[78]]):
    #   블록은 같은 행에서 **두 조각**으로 나간다 — `status_text` 는 (상태 → 이름들),
    #   `window_history_text` 는 (이름 → 날짜). **행 하나의 날짜와 상태를 잇는 줄이 없다.**
    #   그래서 *"가장 최근 미지급 행"* 은 이 문맥에서 **원리상 결정 불가**다(이름 하나가
    #   COMPLETE·REJECTED·IN_PROGRESS 를 동시에 이고 있다). 프롬프트를 어떻게 써도 못 낫는다.
    #   B_join = 그 조인을 **한 줄로** 얹는다(행 순서 그대로·정렬 0·argmax 0).
    #   N_joinlen = **같은 줄에서 상태만 뺀** 부정 통제 — 부피는 같고 새 정보는 0([[57]]).
    rows = rows_from_traj(a.tag, next(iter(sorted({c[1] for c in caps}))),
                          list(spec.get("row_keys") or ()))
    if rows:
        jt = NL + "Each record above, with its date and the status that record carries: %s."
        plan.append(("B_join", tpl, block + jt % join_lines(rows, spec, True)))
        plan.append(("N_joinlen", tpl, block + jt % join_lines(rows, spec, False)))
    else:
        print("⚠궤적에서 행을 못 되살렸다 — 재료 축 두 팔은 건너뛴다([[25]]).")

    print()
    print("## 팔 (선언 `diagnose_prompt` 한 칸 또는 **블록 한 줄**)")
    for nm, t, blk in plan:
        print("  %-10s 요청부 %d자 · 블록 %d자%s"
              % (nm, len(t.partition("{block}")[2]), len(blk),
                 "  ← 재료 축" if blk != block else ""))
    if a.wiring_only:
        print()
        print("(--wiring-only · 모델 호출 0. 각 팔의 요청부 축자:)")
        for nm, t, blk in plan:
            print(" --- %s ---" % nm)
            print(t.partition("{block}")[2].strip())
            if blk != block:
                print("   [블록 추가분] " + blk[len(block):].strip()[:400])
        return 0

    print()
    # 종점 둘: `picked` = 엔진의 집기 규칙(긴 이름 우선·라이브 배달이 그 답을 싣는다) ·
    # `named` = 표적 이름이 답에 **들었나**. B_uniq 는 여럿을 대므로 `picked` 로만 재면
    # 멀쩡한 답이 오답으로 보인다([[76]] ③ 축 섞지 마라).
    print("%-8s %-6s %-24s %-6s %-6s %s"
          % ("팔", "temp", "고른 이름", "picked", "named", "답 머리"))
    print("-" * 128)
    score = {}
    for nm, t, blk in plan:
        body = t.format(block=blk)
        rows = [(0.0, 1)] + ([(a.temp, a.n)] if a.n > 0 else [])
        for tp, n in rows:
            hit = named = 0
            for _ in range(n):
                try:
                    ans = " ".join(str(gen(a.port, body, 96, tp)).split())
                except Exception as e:
                    print("%-8s %-6s 호출 실패: %r" % (nm, tp, e))
                    continue
                p = picked(ans, names)
                ok = (p == target)
                nmd = str(target or "").lower() in ans.lower()
                hit += 1 if ok else 0
                named += 1 if nmd else 0
                print("%-8s %-6s %-24s %-6s %-6s %s"
                      % (nm, tp, p or "원장 밖", "O" if ok else "X",
                         "O" if nmd else "X", ans[:52]))
            score.setdefault(nm, {})[tp] = (hit, named, n)
    print()
    print("## 판정")
    a0 = score.get("A_asis", {}).get(0.0, (0, 0, 1))
    print("  A_asis(temp 0) 표적 picked %d/%d · named %d/%d — 라이브 오지목 재현 %s"
          % (a0[0], a0[2], a0[1], a0[2],
             "**됨**" if a0[1] == 0 else "⛔안 됨 ⇒ 판정하지 마라([[62]] 2b)"))
    for nm, _, _b in plan:
        sc = score.get(nm, {})
        print("  %-10s %s" % (nm, " · ".join("temp %.1f: picked %d/%d · named %d/%d"
                                            % (k, v[0], v[2], v[1], v[2])
                                            for k, v in sorted(sc.items()))))
    print()
    print("⚠N_len 이 처치와 같으면 그 처치는 **길이 효과**다([[57]]).")
    print("⚠종점은 서브의 답 하나다 — 라이브 reward 는 이 프로브가 사지 않는다([[69]]).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
