# -*- coding: utf-8 -*-
"""Nothing decides by how a value is spelled, and nothing decides by a rule that cannot match.

Two removals are held here. The first is the one the user asked for — pattern matching is
cheating — and the second is the accident found while doing it: `withdrawn_row_check`
declared its id rule as `"\\btxn_…"` in JSON, where `\\b` is a **backspace**, so the rule
matched nothing and `T2_WITHDRAWN_ROW=1` harvested zero settled rows in every run it was
on. The engine's own fallback carried literal backspaces too. A census that reads with its
own regex cannot see that — `x94` did exactly that and reported numbers the engine never
had ([[55]]: an instrument without a negative control is not evidence).

So the checks are:

  no declaration decodes to a control character   the exact bug, in any domain, in any key
  no engine source carries one                    the fallback's form of the same bug
  the harvest reads the engine's own list         `_t2_sg_ids`, not the printed sentence
  submission is membership, not shape             values carried, compared to the settled set
  comparison is a type rule                       whole-string numbers, no substring taken

The last one keeps `task_018`'s real target alive (1113 against a recorded 487 is still a
mismatch) while the pair that only ever differed in rendering still agrees.
"""

import ast
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_transcribe as T      # noqa: E402
import xlib_decl as DECL       # noqa: E402
import gate_interpreter as GI  # noqa: E402

CTRL = "".join(chr(c) for c in list(range(0, 9)) + [11, 12] + list(range(14, 32)))
fail = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", (" — " + detail) if detail else ""))
    if not ok:
        fail.append(name)


# ── ① 선언: 디코드하면 제어문자가 되는 값이 하나도 없어야 한다 ────────────────────────
def walk(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, path + "/" + str(k), out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, "%s[%d]" % (path, i), out)
    elif isinstance(o, str) and any(c in o for c in CTRL):
        out.append(path)


bad_decl = []
for p in sorted(glob.glob(os.path.join(HERE, "a2", "**", "*.json"), recursive=True)):
    try:
        d = json.load(io.open(p, encoding="utf-8"))
    except Exception:
        continue
    hits = []
    walk(d, "", hits)
    bad_decl += ["%s%s" % (os.path.basename(p), h) for h in hits]
check("A2 선언에 제어문자로 디코드되는 값 0", not bad_decl, ", ".join(bad_decl[:3]))

# ── ② 엔진 소스: 원시 제어문자(같은 사고의 소스 형태) ────────────────────────────────
bad_src = []
for f in ("t2_gate_patch.py", "t2_scaffold_get.py", "t2_procedure.py", "t2_transcribe.py"):
    t = io.open(os.path.join(HERE, f), encoding="utf-8").read()
    for i, line in enumerate(t.split("\n"), 1):
        if any(c in line for c in CTRL if c != "\t"):
            bad_src.append("%s:%d" % (f, i))
check("엔진 소스에 원시 제어문자 0", not bad_src, ", ".join(bad_src[:3]))

# ── ③ 수확 경로: 출력 텍스트가 아니라 엔진이 계산한 목록을 받는다 ────────────────────
src = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
tree = ast.parse(src)
# `getattr(self, "_t2_sg_ids", …)`이므로 속성이 아니라 **문자열 상수**로 잡는다.
consts = {n.value for n in ast.walk(tree)
          if isinstance(n, ast.Constant) and isinstance(n.value, str)}
check("게이트가 `_t2_sg_ids`를 읽는다", "_t2_sg_ids" in consts)
check("게이트에 `id_pattern` 참조 0 (ref_iso 제외)",
      not [n for n in consts if n == "id_pattern"] or "withdrawn_row_check" not in src[
          max(0, src.find("id_pattern") - 600):src.find("id_pattern")])
sg = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
check("scaffold_get이 호출 id로 목록을 등재한다", "_t2_sg_ids" in sg)

A2 = GI.load_domain_a2("banking_knowledge") or {}
WDS = A2.get("withdrawn_row_check") or {}
check("withdrawn 선언에 철자 규칙 없음", "id_pattern" not in WDS, str(sorted(WDS)))
check("withdrawn 선언은 두 도구 이름만으로 성립",
      bool(WDS.get("settle_tool") and WDS.get("submit_tool")))

# ── ④ 멤버십: 인자에 실린 값이면 형태와 무관하게 잡힌다 ──────────────────────────────
vals = DECL.arg_values({"user_id": "U1", "transaction_id": "txn_a1",
                        "rows": '[{"id": "txn_b2"}]', "n": 3, "ok": True})
check("인자 값 수집이 중첩 JSON까지 본다", {"U1", "txn_a1", "txn_b2", "3"} <= vals, str(sorted(vals)))
check("불리언은 값으로 세지 않는다", "True" not in vals)

# ── ⑤ 판독기: 선언한 문장 틀로 되읽는다(왕복) ────────────────────────────────────────
tool = WDS.get("settle_tool")
forms = DECL.forms(A2, tool)
check("settle 도구의 문장 틀이 선언돼 있다", bool(forms))
ok_round = True
for tpl, item in forms:
    if "{details}" in tpl and item:
        body = "; ".join(item.format(id=i, actual_int=1, expected_floor=2)
                         for i in ("txn_aa", "txn_bb"))
        text = tpl.replace("{details}", body)
    else:
        text = tpl.replace("{ids}", "txn_aa, txn_bb")
    got = DECL.settled_ids(A2, tool, text)
    if not {"txn_aa", "txn_bb"} <= got:
        ok_round = False
check("문장 틀 왕복 — 낸 id를 그대로 되읽는다", ok_round)
check("빈 결과는 id로 세지 않는다",
      not DECL.settled_ids(A2, tool, forms[0][0].replace("{ids}", "(none)")
                           .replace("{details}", "(none)")))

# ── ⑥ 비교: 타입 규칙 — 부분추출 없음 ────────────────────────────────────────────────
check("같은 수의 다른 표기는 일치", T.same("487.0", 487) and T.same(3, "3"))
check("018의 진짜 표적은 살아 있다", not T.same(1113, 487) and not T.same(488, 487))
check("문자열은 그대로 비교", T.same("EcoCard", " ecocard ")
      and not T.same("EcoCard", "Business Platinum Reward"))
check("한쪽만 수로 읽히면 판정하지 않는다", T.same(487, "487 points"))
# 부분추출이 사라졌다 = 렌더링 안에 든 숫자를 **끄집어내지 않는다**. 옛 규칙은 `Account 3 of 5`를
# 3으로, `$487.99`를 487.99로 읽었다 — 전자는 오독이고 후자는 이 모집단에 존재한 적이 없다(x100).
check("부분추출이 사라졌다", T._num("Account 3 of 5") is None and T._num("$487.99") is None)
check("전체가 수인 표기만 수로 읽는다", T._num(" 487.99 ") == 487.99 and T._num("") is None)

print()
print("결과: %s" % ("ALL PASS" if not fail else "FAIL %d — %s" % (len(fail), fail)))
sys.exit(1 if fail else 0)
