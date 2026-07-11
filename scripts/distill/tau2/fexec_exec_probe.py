#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""fexec_exec_probe.py — FORMALIZE-EXEC 형식화 프로브의 **실행-기반 채점** 재작성.

★교체 대상 = `fexec_iso_probe.py`의 손작성 단일-gold EM 채점(버그).
  구-버그: 태스크당 hand-written `GOLD[tid]["cons_fields"]` 하나로 전 결정점을 EM 채점.
    - t20은 4품목이 각각 다른 size 값(신발=9·나머지=제약없음)을 요구 → 단일 gold 무의미.
    - t37 gold `{price}`·availability 암묵제약 미반영 → EM 0%/n2 12%는 **채점버그 산물**.
  per-case 정독(2026-07-11) 확정: n2 88% 실패 = gold-깨짐 44% + 애매 32% + 진짜실패 12%.
  ⇒ 그 수치 전량 폐기. 본 스크립트가 실행-기반 채점으로 대체.

올바른 채점(손작성 gold 0): 모델이 형식화한 spec {op,field,constraints}를 **실행**해서
  tau2 진짜 gold `new_item_id`를 내는가로 채점. gold는 tau2 evaluation_criteria의
  modify/exchange write action에서 직접 온다(손작성 아님).
  - 각 결정점 = gold action `..._items(item_ids=[a..], new_item_ids=[A..])`의 위치쌍 (a→A).
  - old item a의 product 변형(카탈로그) 위에서 모델 spec 실행 → 결과 == {A}이면 exec-correct.
  - 실행 우주 = **available 변형**(tau2 환경규칙: 미가용 변형으로는 교체 불가 — 도메인일반·
    손작성 아님). availability 제약은 eref 규약대로 don't-care(우주로 강제). all-변형도 병기.

재사용(중복구현 0): eref_probe.execute_spec / norm_field / norm_value / parse_spec /
  AVAIL_FIELDS / FORMALIZE_SYS / call_server.

usage:
  py -3 fexec_exec_probe.py --gz <comp gz> --tasks 20,37,79 --dry --out cases.jsonl   # 채점없이 매핑검증
  py -3 fexec_exec_probe.py --gz <comp gz> --tasks all --dry                          # 클래스 전수(자동수집)
  py -3 fexec_exec_probe.py --gz <comp gz> --tasks 20,37,79 \
        --base http://localhost:8140/v1 --model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
        --workers 4 --out fexec_exec_32b.jsonl
표준 stdlib만. 커밋 금지(세션 지시).
"""
import argparse
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eref_probe as ER  # noqa: E402  (execute_spec/norm_field/norm_value/parse_spec/AVAIL_FIELDS/FORMALIZE_SYS/call_server 재사용)

WRITE_RE = re.compile(r"(modify|exchange).*_items$")


# ─────────────────────────── 1. 데이터 로드 ───────────────────────────
def load(gz_path):
    op = gzip.open if gz_path.endswith(".gz") else open
    with op(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    sims = data["simulations"]
    tasks_raw = data.get("tasks")
    if isinstance(tasks_raw, list):
        tasks = {str(t.get("id")): t for t in tasks_raw}
    elif isinstance(tasks_raw, dict):
        tasks = {str(k): v for k, v in tasks_raw.items()}
    else:
        tasks = {}
    # 카탈로그: tool 출력의 product_id+variants (eref_probe.load_products 동형·중복 회피 위해 인라인)
    dec = json.JSONDecoder()
    prods = {}
    for s in sims:
        for m in s.get("messages") or []:
            c = m.get("content")
            if m.get("role") != "tool" or not isinstance(c, str) or "variants" not in c:
                continue
            try:
                obj, _ = dec.raw_decode(c)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("product_id") and isinstance(obj.get("variants"), dict):
                prods[obj["product_id"]] = obj
    item2prod = {}
    for pid, pr in prods.items():
        for vid in pr["variants"]:
            item2prod[vid] = pid
    by_task = {}
    for s in sims:
        by_task.setdefault(str(s.get("task_id")), []).append(s)
    return sims, tasks, prods, item2prod, by_task


def gold_writes(task):
    """tau2 evaluation_criteria서 변형-선택 write action 전부(item_ids∧new_item_ids).
    한 태스크가 여러 주문을 교체하면 write action이 복수 → 전량 수확(위치쌍 누락 방지)."""
    out = []
    for a in ((task.get("evaluation_criteria") or {}).get("actions") or []):
        ar = a.get("arguments") or {}
        ii, nn = ar.get("item_ids"), ar.get("new_item_ids")
        if WRITE_RE.search(a.get("name") or "") and isinstance(ii, list) and isinstance(nn, list) \
                and ii and len(ii) == len(nn):
            out.append((a["name"], ii, nn))
    return out


# ─────────────────────────── 2. 컨텍스트(실 궤적) 절단 ───────────────────────────
def _dec_product_ids(msgs):
    dec = json.JSONDecoder()
    out = set()
    for m in msgs:
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, str) and "variants" in c:
            try:
                obj, _ = dec.raw_decode(c)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("product_id"):
                out.add(obj["product_id"])
    return out


def _first_write_touching(msgs, old):
    """old를 건드리는 첫 write assistant 메시지 인덱스(없으면 첫 write·없으면 None)."""
    first_any = None
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            if not WRITE_RE.search(tc.get("name") or ""):
                continue
            if first_any is None:
                first_any = i
            ar = tc.get("arguments") or {}
            if isinstance(ar, str):
                try:
                    ar = json.loads(ar)
                except Exception:
                    ar = {}
            if old in (ar.get("item_ids") or []):
                return i
    return first_any


def pick_context(sims, old, pid):
    """old 결정점 직전 문맥으로 쓸 (sim, 절단 인덱스) 선택.
    선호 = old를 건드리는 write가 있고, 그 직전 문맥에 old-product 카탈로그가 조회된 sim.
    폴백 = 첫 write 직전·없으면 전체 궤적."""
    best = None
    for s in sims:
        msgs = s.get("messages") or []
        wi = _first_write_touching(msgs, old)
        cut = wi if wi is not None else len(msgs)
        has_prod = pid in _dec_product_ids(msgs[:cut])
        # 우선순위: (old-write 존재, product 조회됨, 문맥 길이)
        score = (wi is not None and old in ((_write_items(msgs, wi)) or []),
                 has_prod, cut)
        if best is None or score > best[0]:
            best = (score, s, cut)
    if best is None:
        return None, 0
    return best[1], best[2]


def _write_items(msgs, i):
    if i is None or i >= len(msgs):
        return None
    for tc in (msgs[i].get("tool_calls") or []):
        if WRITE_RE.search(tc.get("name") or ""):
            ar = tc.get("arguments") or {}
            if isinstance(ar, str):
                try:
                    ar = json.loads(ar)
                except Exception:
                    ar = {}
            return ar.get("item_ids")
    return None


def transcript(msgs, limit_chars=13000):
    """실 궤적 dict 메시지 → 텍스트 전사. user 발화=전문 보존(품목 지시 유지),
    tool 출력=truncate(variants 덤프는 후보로 별도 제공 → 강 truncate·타 출력은 완만)."""
    lines = []
    for m in msgs:
        role = m.get("role")
        c = m.get("content")
        if role == "user" and c:
            lines.append("USER: " + str(c).strip())
        elif role == "assistant":
            if c:
                lines.append("ASSISTANT: " + str(c).strip()[:400])
            for tc in (m.get("tool_calls") or []):
                ar = tc.get("arguments")
                if isinstance(ar, (dict, list)):
                    ar = json.dumps(ar, ensure_ascii=False, default=str)
                lines.append("ASSISTANT_CALL %s(%s)" % (tc.get("name"), str(ar)[:200]))
        elif role == "tool" and c is not None:
            s = str(c)
            trunc = 200 if "variants" in s else 1600  # variants=후보로 재공급→강 truncate
            lines.append("TOOL: " + s[:trunc])
    txt = "\n".join(lines)
    if len(txt) > limit_chars:  # 앞(초기 지시)은 남기고 중간을 자름
        head = txt[: limit_chars * 2 // 3]
        tail = txt[-limit_chars // 3:]
        txt = head + "\n...[transcript truncated]...\n" + tail
    return txt


# ─────────────────────────── 3. 프롬프트 ───────────────────────────
def build_prompt(ctx_txt, product, old, old_rec):
    """FORMALIZE_SYS(eref 재사용) + 실 대화 + old 품목 명시 + 후보 = product 변형 전체."""
    pname = product.get("name", "item")
    rec_lines = ["- %s   | record: %s" % (vid, json.dumps(v, ensure_ascii=False, default=str))
                 for vid, v in sorted(product["variants"].items())]
    old_note = ("\n\n=== Item currently being replaced ===\n- %s (a %s)   | record: %s"
                % (old, pname, json.dumps(old_rec, ensure_ascii=False, default=str)
                   if old_rec is not None else "(record not in catalog)"))
    return (ER.FORMALIZE_SYS
            + "\n\n=== Conversation ===\n" + ctx_txt
            + old_note
            + "\n\n=== Candidate records for 'new_item_id' ===\n" + "\n".join(rec_lines)
            + "\n\nThe agent must now choose 'new_item_id' to REPLACE the item %s (the %s above). "
              "Formalize ONLY this item's replacement criterion as the JSON object." % (old, pname))


# ─────────────────────────── 4. 채점(실행-기반) ───────────────────────────
def _norm_spec(spec):
    """실행용 정규화: availability 제약 제거(우주로 강제)·field 정규화(eref 규약 동형)."""
    n_cons = []
    for c in spec.get("constraints") or []:
        nf = ER.norm_field(c["field"])
        if nf in ER.AVAIL_FIELDS:
            continue
        n_cons.append({"field": nf, "op": c["op"], "value": c["value"]})
    return {"op": spec["op"], "field": ER.norm_field(spec.get("field")), "constraints": n_cons}


def _records(product, available_only):
    recs = []
    for vid, v in sorted(product["variants"].items()):
        if available_only and v.get("available") is not True:
            continue
        recs.append((vid, v))
    return recs


def ref_op_family(product, gold_new):
    """자동-도출 참조 op(손작성 0): available 변형서 무제약 argmax/argmin(price)이 gold를
    유일 산출하면 그 op. 아니면 'constrained'(제약 필요·op는 forensic서)."""
    recs = _records(product, True)
    prices = [(float(v.get("price")), vid) for vid, v in recs if v.get("price") is not None]
    if not prices:
        return "none"
    mx = max(p for p, _ in prices); mn = min(p for p, _ in prices)
    top = [vid for p, vid in prices if p == mx]; bot = [vid for p, vid in prices if p == mn]
    if top == [gold_new]:
        return "argmax"
    if bot == [gold_new]:
        return "argmin"
    return "constrained"


def grade(spec, product, gold_new):
    out = {"parsed": spec is not None, "op": None, "has_cons": False, "empty_cons": True,
           "exec_status_avail": None, "exec_status_all": None,
           "exec_correct_avail": False, "exec_correct_all": False, "spec": spec}
    if spec is None:
        return out
    out["op"] = spec["op"]
    ns = _norm_spec(spec)
    out["has_cons"] = len(ns["constraints"]) > 0
    out["empty_cons"] = len(ns["constraints"]) == 0
    for uni, key_s, key_c in (("avail", "exec_status_avail", "exec_correct_avail"),
                              ("all", "exec_status_all", "exec_correct_all")):
        recs = _records(product, uni == "avail")
        res = ER.execute_spec(ns, recs)
        out[key_s] = res["status"]
        out[key_c] = res["status"] == "ok" and set(res["ids"]) == {gold_new}
    return out


# ─────────────────────────── 5. 케이스 수집 ───────────────────────────
def collect(tasks, prods, item2prod, by_task, target):
    """target = ['20','37','79'] 또는 None(=클래스 전수 자동수집)."""
    cases, skipped = [], []
    tids = sorted(tasks, key=lambda x: (len(x), x)) if target is None else target
    for tid in tids:
        t = tasks.get(tid)
        if t is None:
            skipped.append((tid, "no_task"))
            continue
        gws = gold_writes(t)
        if not gws:
            if target is not None:
                skipped.append((tid, "no_variant_select_write"))
            continue
        instr = ((t.get("user_scenario") or {}).get("instructions") or {}).get("reason_for_call") or ""
        sims = by_task.get(tid, [])
        positions = [(widx, name, i, old, gnew)
                     for widx, (name, item_ids, new_ids) in enumerate(gws)
                     for i, (old, gnew) in enumerate(zip(item_ids, new_ids))]
        for widx, name, i, old, gnew in positions:
            n_items = len(gws[widx][1])
            pid = item2prod.get(old)
            if pid is None:
                skipped.append((tid, "old_%s_no_product" % old))
                continue
            product = prods[pid]
            if gnew not in product["variants"]:
                skipped.append((tid, "goldnew_%s_not_in_product" % gnew))
                continue
            gv = product["variants"][gnew]
            if gv.get("available") is not True:
                skipped.append((tid, "goldnew_%s_unavailable" % gnew))
                continue
            sim, cut = pick_context(sims, old, pid)
            ctx = transcript((sim.get("messages") or [])[:cut]) if sim is not None else ""
            old_rec = product["variants"].get(old)
            pname = product.get("name", "item")
            # 품목 지시 존재 heuristic(품명 토큰이 대화에 등장하는가)
            pw = [w for w in re.split(r"\W+", pname.lower()) if len(w) > 2]
            instr_in_prompt = any(w in ctx.lower() for w in pw) if pw else False
            cases.append({
                "task": tid, "write": name, "widx": widx, "pos": i, "n_items": n_items,
                "old": old, "gold_new": gnew, "pid": pid, "product": pname,
                "n_variants": len(product["variants"]),
                "n_available": sum(1 for v in product["variants"].values() if v.get("available") is True),
                "old_opts": (old_rec or {}).get("options"),
                "gold_opts": gv.get("options"), "gold_price": gv.get("price"),
                "ref_op": ref_op_family(product, gnew),
                "instr_in_prompt": instr_in_prompt,
                "ctx_sim_trial": (sim or {}).get("trial"), "ctx_chars": len(ctx),
                "reason_for_call": instr[:400],
                "prompt": build_prompt(ctx, product, old, old_rec),
                "_product": product,  # 채점용(직렬화 시 pop)
            })
    return cases, skipped


# ─────────────────────────── 6. main ───────────────────────────
def _agg(rows):
    n = len(rows)
    if not n:
        return None
    f = lambda k: sum(1 for r in rows if r.get(k)) / n
    return {"n": n, "parsed": f("parsed"),
            "exec_correct_avail": f("exec_correct_avail"),
            "exec_correct_all": f("exec_correct_all"),
            "empty_cons": f("empty_cons"), "has_cons": f("has_cons")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", required=True)
    ap.add_argument("--tasks", default="20,37,79",
                    help="task id 목록 또는 'all'(변형-선택 클래스 전수 자동수집)")
    ap.add_argument("--dry", action="store_true", help="서버 없이 케이스/gold 매핑 검증만")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    _, tasks, prods, item2prod, by_task = load(a.gz)
    target = None if a.tasks.strip().lower() == "all" else [t.strip() for t in a.tasks.split(",") if t.strip()]
    cases, skipped = collect(tasks, prods, item2prod, by_task, target)
    print("== FORMALIZE-EXEC 실행-채점 프로브 ==")
    print("products=%d  cases(decision points)=%d  skipped=%d"
          % (len(prods), len(cases), len(skipped)))
    if skipped:
        print("skipped:", skipped[:20])
    ntask = len(set(c["task"] for c in cases))
    print("tasks covered=%d  empty ref_op dist=%s" % (
        ntask, {k: sum(1 for c in cases if c["ref_op"] == k)
                for k in ("argmax", "argmin", "constrained", "none")}))
    print("instr_in_prompt(품명 등장)=%d/%d" % (sum(1 for c in cases if c["instr_in_prompt"]), len(cases)))

    if a.dry:
        # per-case 매핑 검증 출력 (gold_new가 available·프롬프트에 품목 지시 실렸는지)
        for c in cases:
            print("  t%s p%d/%d old=%s->gold=%s %s ref_op=%s n_avail=%d instr=%s old_opts=%s gold_opts=%s"
                  % (c["task"], c["pos"], c["n_items"], c["old"], c["gold_new"], c["product"][:16],
                     c["ref_op"], c["n_available"], c["instr_in_prompt"],
                     json.dumps(c["old_opts"], ensure_ascii=False, default=str),
                     json.dumps(c["gold_opts"], ensure_ascii=False, default=str)))
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                for c in cases:
                    d = {k: v for k, v in c.items() if k != "_product"}
                    f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
            print("[dry] %d cases -> %s" % (len(cases), a.out))
        return

    # 실측 (서버)
    from concurrent.futures import ThreadPoolExecutor

    def run_one(c):
        try:
            return c, ER.call_server(a.base, a.model, c["prompt"]), None
        except Exception as e:
            return c, None, type(e).__name__

    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (c, ans, err) in enumerate(ex.map(run_one, cases)):
            if err:
                print("  [%d/%d] SERVER_ERR t%s %s" % (i + 1, len(cases), c["task"], err), flush=True)
                continue
            spec = ER.parse_spec(ans)
            g = grade(spec, c["_product"], c["gold_new"])
            row = {k: v for k, v in c.items() if k != "_product"}
            row.update(g)
            row["raw"] = (ans or "")[:800]
            rows.append(row)
            print("  [%d/%d] t%s p%d old=%s ref=%s op=%s cons=%s exec_avail=%s(%s) exec_all=%s"
                  % (i + 1, len(cases), c["task"], c["pos"], c["old"], c["ref_op"],
                     g["op"], not g["empty_cons"], g["exec_correct_avail"],
                     g["exec_status_avail"], g["exec_correct_all"]), flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # 집계
    print("\n== 실행-채점 집계 (primary=available 우주 / secondary=all 변형) ==")
    ov = _agg(rows)
    if ov:
        print("OVERALL n=%d parse=%.2f exec-correct[avail]=%.2f exec-correct[all]=%.2f empty-cons=%.2f has-cons=%.2f"
              % (ov["n"], ov["parsed"], ov["exec_correct_avail"], ov["exec_correct_all"],
                 ov["empty_cons"], ov["has_cons"]))
    bytask = {}
    for r in rows:
        bytask.setdefault(r["task"], []).append(r)
    print("task  n  parse exec[avail] exec[all] empty-cons")
    for tid in sorted(bytask, key=lambda x: (len(x), x)):
        g = _agg(bytask[tid])
        print("t%-4s %2d  %.2f   %.2f       %.2f     %.2f"
              % (tid, g["n"], g["parsed"], g["exec_correct_avail"], g["exec_correct_all"], g["empty_cons"]))
    # exec_status 분포 (available 우주)
    st = {}
    for r in rows:
        st[r.get("exec_status_avail")] = st.get(r.get("exec_status_avail"), 0) + 1
    print("exec_status[avail] 분포:", st)
    # ref_op별 exec-correct (op 능력 격리)
    print("ref_op별 exec-correct[avail]:")
    for fam in ("argmax", "argmin", "constrained"):
        rs = [r for r in rows if r["ref_op"] == fam]
        if rs:
            g = _agg(rs)
            print("  %-12s n=%d exec-correct=%.2f empty-cons=%.2f" % (fam, g["n"], g["exec_correct_avail"], g["empty_cons"]))
    print("\n1차 답: 형식화가 실제로 gold item을 내는가 = exec-correct[avail]. "
          "구 fexec EM(0%/n2 12%)=채점버그·폐기(본 스크립트가 대체).")


if __name__ == "__main__":
    main()
