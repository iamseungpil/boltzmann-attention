#!/usr/bin/env python
"""τ² 전-task × P1-P9 정적 census (zero-cost·리뷰#4b "정확히 2" 방어, 2026-06-15).

autopsy(모델 행동 분류)와 다름: 이건 **task-spec/gold가 *요구*하는 primitive**를 정적 도출.
모델 실행 0 — gold action 시퀀스 + 도구 write/read 분류 + user_scenario 텍스트만 읽음.

목적(매트릭스 §3 리뷰#4b):
  1. 전 task 요구집합 ⊆ {P1..P9} (orphan tool=분류밖 연산 0) 확인 → "P10 없음" 방어.
  2. gap = (학습 미커버) P6+P7 이 20개 autopsy 너머 *전수*서 성립하는지 분포로 확인.

도구 분류 = τ² retail tools.py 의 @is_tool(ToolType.WRITE) (반환시그니처 원칙·prefix 아님).

Usage: tau2_primitive_census.py <domain_dir> [--task-file tasks.json] [--dump-orphans]
  예: tau2_primitive_census.py /home/woori/scratch/tau2-bench/data/tau2/domains/retail
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

# ── 도구 분류: 도메인 tools.py 의 @is_tool(ToolType.WRITE) 동적 파싱 (반환시그니처 원칙) ──
# retail 기본값(tools.py 부재 시 fallback). load_tools()로 도메인별 갱신.
WRITE_TOOLS = {
    "cancel_pending_order", "exchange_delivered_order_items",
    "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
}
AUTH_TOOLS = {"find_user_id_by_name_zip", "find_user_id_by_email"}
GETTER_TOOLS = {
    "get_order_details", "get_product_details", "get_item_details",
    "get_user_details", "list_all_product_types", "calculate",
}
META_TOOLS = {"transfer_to_human_agents", "think"}
ALL_KNOWN = WRITE_TOOLS | AUTH_TOOLS | GETTER_TOOLS | META_TOOLS

META_NAMES = ("transfer_to_human_agents", "think", "transfer_to_human")
AUTH_PREFIXES = ("find_user", "find_customer", "authenticate", "verify_identity")


def load_tools(tools_py):
    """도메인 tools.py서 공개 메서드(tool) 추출·@is_tool(ToolType.WRITE) 표식으로 write 분류.
    반환시그니처 원칙(prefix 아님): 데코레이터가 권위. 미발견 시 retail 기본값 유지."""
    global WRITE_TOOLS, AUTH_TOOLS, GETTER_TOOLS, META_TOOLS, ALL_KNOWN
    if not tools_py or not os.path.exists(tools_py):
        return False
    lines = open(tools_py, encoding="utf-8").read().splitlines()
    writes, getters, auths, metas = set(), set(), set(), set()
    pending = None  # 최근 @is_tool(ToolType.X) 의 X ("WRITE"/"READ"/"GENERIC"/"THINK")
    for ln in lines:
        s = ln.strip()
        if s.startswith("#"):
            continue  # 주석처리된 데코레이터(예: # @is_tool(...)) 무시
        m = re.match(r"@is_tool\(\s*ToolType\.([A-Z]+)", s)
        if m:
            pending = m.group(1)
            continue
        dm = re.match(r"def ([a-z][a-z0-9_]*)\s*\(", s)  # self 다음 줄 가능(멀티라인 sig)
        if dm and pending is not None:
            name = dm.group(1)
            if name.startswith("_"):
                pending = None
                continue
            if pending == "WRITE":
                writes.add(name)
            elif name in META_NAMES or pending == "THINK":
                metas.add(name)
            elif any(name.startswith(p) for p in AUTH_PREFIXES):
                auths.add(name)
            else:  # READ / GENERIC(calculate 등) = getter(값 소스)
                getters.add(name)
            pending = None
        elif s and not s.startswith("@"):
            pending = None
    if not (writes or getters):
        return False
    WRITE_TOOLS, AUTH_TOOLS, GETTER_TOOLS, META_TOOLS = writes, auths, getters, metas
    ALL_KNOWN = WRITE_TOOLS | AUTH_TOOLS | GETTER_TOOLS | META_TOOLS
    return True

# 결정/조건 언어 (P2a gather-for-decision = 출력→조건분기)
COND_RE = re.compile(
    r"\b(if|otherwise|unless|whichever|cheapest|most expensive|compatible|instead|"
    r"depending|in case|as long as|prefer|would go for|only if|provided that|"
    r"as well as|either|in that case)\b", re.I)
# args 중 "fetch해야만 얻는" 슬롯 (2-hop·user가 못 주는 시스템 생산값)
FETCH_ARG_KEYS = {
    "payment_method_id", "item_ids", "new_item_ids", "user_id",
    "amount", "gift_card_id",
}


def _argvals(args):
    out = []
    if not isinstance(args, dict):
        return out
    for v in args.values():
        if isinstance(v, list):
            out += [str(x) for x in v]
        elif v not in (None, ""):
            out.append(str(v))
    return out


def census_task(t):
    """gold actions + scenario서 요구 primitive 집합 도출."""
    ev = (t.get("evaluation_criteria") or {})
    actions = ev.get("actions") or []
    sc = (t.get("user_scenario") or {})
    inst = (sc.get("instructions") or {})
    known = " ".join(str(inst.get(k) or "") for k in
                      ("task_instructions", "reason_for_call", "known_info")).lower()
    unknown = str(inst.get("unknown_info") or "").lower()

    names = [a.get("name") for a in actions]
    writes = [a for a in actions if a.get("name") in WRITE_TOOLS]
    auths = [a for a in actions if a.get("name") in AUTH_TOOLS]
    getters = [a for a in actions if a.get("name") in GETTER_TOOLS]
    orphans = [n for n in names if n not in ALL_KNOWN]

    req = set()
    notes = []

    # P8 auth/provenance — gold에 auth 액션
    if auths:
        req.add("P8")
    # P1 grounding — auth/첫 액션 인자가 known 컨텍스트서 복사(지어내면 안 됨)
    grounded_copy = False
    for a in actions:
        for v in _argvals(a.get("arguments")):
            if len(v) >= 2 and v.lower() in known:
                grounded_copy = True
                break
        if grounded_copy:
            break
    if grounded_copy or actions:
        req.add("P1")  # 모든 task가 ≥1 컨텍스트값 복사 요구
    # P3 시퀀싱 — ≥2 액션(auth→getter→write 의존)
    if len(actions) >= 2:
        req.add("P3")
    # P5 policy-gate · P6 confirm-gate — write 존재
    if writes:
        req.add("P5")
        req.add("P6")
    # P2b gather-for-arg(2-hop) — 하류(write) 인자값이 known-context에 없음 = fetch 필요(도메인-일반).
    #   known 키 + 도메인-일반(write 인자값 ∉ known & 식별자형)으로 검출.
    p2b = False
    for a in writes + getters:
        args = a.get("arguments") or {}
        for k, v in (args.items() if isinstance(args, dict) else []):
            vals = v if isinstance(v, list) else [v]
            for vv in vals:
                s = str(vv)
                key_hit = k in FETCH_ARG_KEYS
                # 도메인-일반: 식별자형(len≥4·영숫자/특수)이고 known에 없음 = 상류서 fetch
                gen_hit = (len(s) >= 4 and s.lower() not in known
                           and any(c.isdigit() for c in s) and not s.replace(".", "").isdigit())
                if (key_hit or gen_hit) and len(s) >= 2 and s.lower() not in known:
                    p2b = True
                    notes.append("2hop:%s=%s" % (k, s[:14]))
                    break
    if p2b:
        req.add("P2b")
    # P4 select-from-output — 변형/항목 선택(product_details + new_item_ids, 또는 list_all)
    sel = False
    if any(n == "get_product_details" for n in names) and \
       any("new_item_ids" in (a.get("arguments") or {}) for a in writes):
        sel = True
    if any(n == "list_all_product_types" for n in names):
        sel = True
    if sel:
        req.add("P4")
    # P2a gather-for-decision — scenario에 조건/결정 언어 + getter 존재
    if getters and COND_RE.search(known):
        req.add("P2a")
    # P7 recovery — *gold엔 없음*(성공경로). 잠재 신호=unknown_info(fallback fetch 강제)
    latent_p7 = bool(unknown) and unknown not in ("", "none")

    return {
        "id": t.get("id"), "n_act": len(actions), "names": names,
        "req": req, "orphans": orphans, "latent_p7": latent_p7, "notes": notes[:4],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain_dir")
    ap.add_argument("--task-file", default="tasks.json")
    ap.add_argument("--tools-py", default=None, help="도메인 tools.py(미지정 시 data→src 경로 유도)")
    ap.add_argument("--dump-orphans", action="store_true")
    ap.add_argument("--show", type=int, default=0, help="첫 N task 상세")
    a = ap.parse_args()

    # 도메인 tools.py 동적 로드 (반환시그니처 분류)
    tp = a.tools_py
    if tp is None and "/data/tau2/domains/" in a.domain_dir:
        tp = os.path.join(a.domain_dir.replace("/data/tau2/domains/", "/src/tau2/domains/"), "tools.py")
    loaded = load_tools(tp)
    print("[tools.py] %s → write=%d getter=%d auth=%d %s" %
          (tp, len(WRITE_TOOLS), len(GETTER_TOOLS), len(AUTH_TOOLS),
           "(동적)" if loaded else "(retail 기본값)"))

    path = os.path.join(a.domain_dir, a.task_file)
    tasks = json.load(open(path, encoding="utf-8"))
    rows = [census_task(t) for t in tasks]

    PRIMS = ["P1", "P2a", "P2b", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    # P7 = gold-부재(잠재만), P9 = τ² sequential(없음)
    cov = Counter()
    for r in rows:
        for p in r["req"]:
            cov[p] += 1
    n_latent_p7 = sum(1 for r in rows if r["latent_p7"])

    print("=== τ² PRIMITIVE CENSUS: %s (n=%d) ===" % (path, len(rows)))
    print("\n[task별 요구 primitive 분포 (정적·gold 기반)]")
    for p in PRIMS:
        if p == "P7":
            print("  %-4s %4d/%d  (★gold-부재=reactive; 잠재 unknown_info=%d)" %
                  (p, 0, len(rows), n_latent_p7))
        elif p == "P9":
            print("  %-4s %4d/%d  (τ² turn-sequential=구조상 0)" % (p, cov[p], len(rows)))
        else:
            print("  %-4s %4d/%d" % (p, cov[p], len(rows)))

    # orphan tools = 분류밖 연산 (P10 후보)
    all_names = Counter()
    orphan_names = Counter()
    for r in rows:
        for n in r["names"]:
            all_names[n] += 1
            if n in [o for o in r["orphans"]]:
                orphan_names[n] += 1
    print("\n[orphan 도구 = P1-P9 밖 연산 → P10 후보]")
    if orphan_names:
        for n, c in orphan_names.most_common():
            print("  ⚠️ %s  ×%d" % (n, c))
    else:
        print("  ✅ 0 — 전 gold 도구가 P1-P9로 매핑됨 (분류 밖 연산 없음)")

    # 요구집합 완전성: 모든 task req ⊆ 분류
    print("\n[요구집합 완전성]")
    over = [r["id"] for r in rows if r["orphans"]]
    print("  P1-P9 밖 요구 task: %d %s" % (len(over), over[:10]))

    # write-task 분포 = P5+P6 gap 직접 관련
    n_write = sum(1 for r in rows if "P5" in r["req"])
    n_p2b = sum(1 for r in rows if "P2b" in r["req"])
    n_p4 = sum(1 for r in rows if "P4" in r["req"])
    print("\n[gap=P6+P7 방어 관련 분포]")
    print("  write 포함(→P5·P6 필요): %d/%d (%.0f%%)" % (n_write, len(rows), 100*n_write/len(rows)))
    print("  P2b(2-hop arg) 필요: %d/%d (%.0f%%)" % (n_p2b, len(rows), 100*n_p2b/len(rows)))
    print("  P4(select) 필요: %d/%d" % (n_p4, len(rows)))
    print("  잠재 P7(unknown_info fallback): %d/%d" % (n_latent_p7, len(rows)))

    # 고유 도구 전체 매핑표
    print("\n[고유 gold 도구 → primitive 매핑]")
    def tmap(n):
        if n in AUTH_TOOLS: return "P8(+P1)"
        if n in WRITE_TOOLS: return "P5+P6(+P2b/P4 인자)"
        if n in GETTER_TOOLS: return "P2a/P2b/P4 소스"
        if n in META_TOOLS: return "meta(non-primitive)"
        return "⚠️ORPHAN"
    for n, c in all_names.most_common():
        print("  %-34s ×%-4d %s" % (n, c, tmap(n)))

    if a.show:
        print("\n[첫 %d task 상세]" % a.show)
        for r in rows[:a.show]:
            print("  task %-4s n=%d req={%s} %s %s" %
                  (r["id"], r["n_act"], ",".join(sorted(r["req"])),
                   ("latentP7" if r["latent_p7"] else ""), " ".join(r["notes"])))


if __name__ == "__main__":
    main()
