# -*- coding: utf-8 -*-
r"""x576 — 072 의 **현 세대 상태**를 정본 변이집합으로 읽는다 (모델 0 · 무료 · 계수만).

## 왜 (2026-08-28 · 사용자 지시 *"072 pass 시켜라"*)

`tasks__20260824/TASK_072.md` 는 t7348 에서 원인 다섯을 CONFIRMED 로 확정했고, 그 뒤 **네
런**(t7356·t7362·t7363·t7368)이 지나갔다. 그 사이 수리 둘이 지어졌다:

    T2_DUP_WRITE           `4404e01f` — 중복 write 를 재생성 채널로 지운다 (드라이버에서 **ON**)
    T2_ACTIONREQ_GROUNDED  `3053e5d3` — 대화에 없는 손님-측 도구 지목을 침묵 (드라이버에 **없음 = OFF**)

⇒ 처방이 어디까지 들었는지를 **재기부터** 한다([[62]]①·[[74]] 세대 뭉개기 금지).

## 무엇을 세나 (판단 0 · 전부 정본 함수)

  ⑴ `mutation_diff` — MISSING/WRONGARG/EXTRA/DUP/BLOCKED 와 **gap**
  ⑵ 그 sim 의 크레딧 계열 호출 전량(도구·인자·성공여부) — 라인 분할·중복·부호반전이 보이게
  ⑶ 원인 다섯의 **라이브 자국**: `[T2_WRITE_SUB] 통과 0건` · `formalized_target=None|손님도구` ·
     `[T2_FORCE_ACTION]` 종결 후 발화 · `[T2_DUP_WRITE]` 실제 제거 · `[ARGS-FORMAT]`

⛔이 프로브는 아무것도 안 고치고 아무 값도 안 만든다. gold 는 `mutation_diff` 안에서만 쓰인다.

사용: (리모트) cd $REPO/scripts/distill/tau2 && PYTHONPATH=. py -3 x576_072_state_census.py
"""
import argparse
import collections
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

import t2_forensic as F                                             # noqa: E402

NL = chr(10)
OUT = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")

# 최근 세대만 — 072 를 담은 런 ([[74]])
TAGS = ("bank_t7368_hard0_20260827", "bank_t7363_hard0_20260827",
        "bank_t7356_grpB3_20260826", "bank_t7348_halfA_20260824")

CREDIT = re.compile(r"apply_\w*_credit\w*", re.I)

# 원인 다섯의 자국 — 문자열은 **로그가 인쇄하는 것 그대로**다(해석 0)
MARKS = {
    "WRITE_SUB_통과0": "근거검산 통과 0건",
    "표적_None": "formalized_target=None",
    "표적_submit_transaction": "formalized_target=submit_transaction",
    "FORCE_ACTION": "[T2_FORCE_ACTION]",
    "DUP_WRITE_제거": "[T2_DUP_WRITE]",
    "ARGS_FORMAT": "[ARGS-FORMAT]",
    "RESOLVE_CAP": "[T2_RESOLVE_CAP]",
}


def credit_calls(sim):
    """그 sim 의 크레딧 계열 호출 전량 — 이름·인자·성공여부. 판단 0."""
    out = []
    msgs = sim.get("messages") or []
    for i, m in enumerate(msgs):
        for tc in (m.get("tool_calls") or []):
            nm = F.nameof(tc) or ""
            args = F.argsof(tc) or {}
            inner = F.inner_name(args) or ""
            target = inner or nm
            if not CREDIT.search(str(target)):
                continue
            # 성공 여부 = 바로 뒤 tool 메시지가 Error 로 시작하지 않는가 (엔진 해석 0)
            ok = None
            for j in range(i + 1, min(i + 4, len(msgs))):
                if msgs[j].get("role") == "tool":
                    ok = not str(msgs[j].get("content") or "").lstrip().startswith("Error")
                    break
            out.append({"msg": i, "tool": target, "args": F.norm_args(args), "ok": ok})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task_072")
    ap.add_argument("--tags", default=",".join(TAGS))
    a = ap.parse_args(argv)

    mut = F.mutating_tools()
    rows = []
    for tag in [t.strip() for t in a.tags.split(",") if t.strip()]:
        try:
            ss = F.sims(tag)
        except Exception as e:
            print("[skip] %s — %r" % (tag, e))
            continue
        txt = F.log_text(tag) or ""
        per_sim_log = collections.defaultdict(list)
        for ln in txt.splitlines():
            m = re.search(r"\[sim=(task_\d+#s\d+)\]", ln)
            if m:
                per_sim_log[m.group(1)].append(ln)
        for s in ss:
            if F.task_id(s) != a.task:
                continue
            st = F.simtag(s)
            d = F.mutation_diff(s, mut, tag=tag) or {}
            gap = sum(len(d.get(k) or ()) for k in ("missing", "wrongarg", "extra", "dup"))
            lines = per_sim_log.get(st) or []
            marks = {k: sum(1 for ln in lines if v in ln) for k, v in MARKS.items()}
            rows.append({
                "tag": tag, "sim": st,
                "reward": (s.get("reward_info") or {}).get("reward"),
                "basis": (s.get("reward_info") or {}).get("reward_basis"),
                "term": F.term_reason(s), "msgs": len(s.get("messages") or []),
                "gap": gap,
                "missing": [F.label(x.get("name"), x.get("arguments")) if isinstance(x, dict) else str(x)
                            for x in (d.get("missing") or ())],
                "wrongarg": [str(x) for x in (d.get("wrongarg") or ())],
                "extra": [str(x) for x in (d.get("extra") or ())],
                "blocked": [str(x) for x in (d.get("blocked") or ())],
                "sidecar": d.get("sidecar"),
                "credits": credit_calls(s),
                "marks": marks,
            })

    if not rows:
        print("그 태스크의 sim 을 못 찾았다 — 판정하지 않는다([[25]])", file=sys.stderr)
        return 2

    print("# x576 — %s 현 세대 상태 (sim %d)" % (a.task, len(rows)))
    print("")
    print("%-26s %-20s %-6s %-4s %-5s %s" % ("tag", "sim", "reward", "gap", "크레", "종료"))
    print("-" * 96)
    for r in rows:
        print("%-26s %-20s %-6s %-4d %-5d %s"
              % (r["tag"][5:31], r["sim"], r["reward"], r["gap"], len(r["credits"]), r["term"]))

    for r in rows:
        print("")
        print("=" * 96)
        print("%s · %s · reward=%s · basis=%s · gap=%d · 사이드카=%s"
              % (r["tag"], r["sim"], r["reward"], r["basis"], r["gap"], r["sidecar"]))
        for k in ("missing", "wrongarg", "extra", "blocked"):
            if r[k]:
                print("  %-9s %d" % (k.upper(), len(r[k])))
                for x in r[k][:12]:
                    print("      %s" % x)
        if r["credits"]:
            print("  크레딧 호출 %d건 (msg · 인자 · 성공)" % len(r["credits"]))
            for c in r["credits"]:
                print("      [%3d] %-38s %-52s %s"
                      % (c["msg"], c["tool"][:38], json.dumps(c["args"], ensure_ascii=False)[:52],
                         c["ok"]))
        print("  자국: " + " · ".join("%s=%d" % (k, v) for k, v in r["marks"].items() if v))

    print("")
    print("=" * 96)
    print("자국 합계 (원인 다섯이 이 세대에 살아 있나)")
    print("=" * 96)
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        for k, v in r["marks"].items():
            agg[r["tag"]][k] += v
    for tag in sorted(agg):
        print("  %-30s %s" % (tag[5:35],
                              " · ".join("%s=%d" % kv for kv in sorted(agg[tag].items()) if kv[1])))

    dst = os.path.join(OUT, "x576_072_state_census_2026_08_28.json")
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump({"probe": "x576_072_state_census", "date": "2026-08-28",
                   "task": a.task, "tags": [t.strip() for t in a.tags.split(",")],
                   "rows": rows,
                   "limits": ["변이집합·gold 는 정본 `mutation_diff` 만 썼다([[69]]).",
                              "크레딧 성공여부는 **바로 뒤 tool 메시지가 Error 로 시작하는지**뿐 —"
                              " 우리 층 재생성 거절은 궤적에 안 남는다(사이드카가 권위·[[30]]).",
                              "자국은 로그 문자열 계수일 뿐 인과가 아니다([[08]])."]},
                  f, ensure_ascii=False, indent=1)
    print("")
    print("-> %s" % os.path.normpath(dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
