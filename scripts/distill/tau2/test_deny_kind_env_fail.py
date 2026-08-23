# -*- coding: utf-8 -*-
"""거절 본문을 **누가 썼는가** 로 가르는가 (A-7⑵ 2026-08-23 · R1 2026-08-23).

이 파일은 두 결함을 한꺼번에 지킨다.

⑴ **A-7⑵ (원래 계기)** — `deny_kind` 는 env 거절을 `Error:` 접두로만 알아봤다. 이 환경은
   `Failed to …` 로도 거절하고, 그 본문을 성공으로 세면 그 호출이 MATCHED 가 되며 앞선
   성공이 DUP 으로 재분류된다 — 079 의 DUP 주장이 그렇게 태어났다.

⑵ **R1 (`deny_kind` 가 우리 거절을 env 로 찍는다)** — 우리 게이트의 거절은 tau2 규약대로
   `Error: ` 로 시작한다. 표지를 두 개만 든 `OURS_DENY` 는 그 전부를 놓쳐 **환경 탓**으로
   돌렸다. 전 코퍼스 462 파일 실측(수리 직전/직후 같은 코드로 대조):
       env  → ours   **1,295**  (`Error: [POLICY GATE …]` 764 · BYREF 398 · ARGS-FORMAT 108
                                 · PRE-ACTION-KB 24 · RESULT-SIGN 1)
       (없음) → ours **2,554**  (`[DUPLICATE-READ]` 2,340 · `[DUPLICATE-COMPUTE]` 214)
       그 중 **변이 도구 위 27 건**이 `ok=True`(=실행됨)로 세어져 `mutation_diff` 의
       done/dup 칸을 직접 오염시켰다 — *막힌 변이를 실행된 것으로* 센 것이다.
   ⇒ "우리 층이 막은 것은 0 건" 이라는 결론은 구조상 **항상 참**이었고 아무 내용이 없었다.
   이것은 레버가 아니라 **자[尺]의 눈금 수리**다 — 라이브 거동은 한 바이트도 안 바뀐다.

판정 기준은 [[69]] 다: reward 는 궤적 재실행 후 **DB 해시 비교**이므로 막힌 write 는 상태를
안 바꿔 해시에 안 남는다 ⇒ BLOCKED 가 맞다.

실물 코퍼스로도 잰다 — 이 저장소가 반복해서 진 방식이 *자기 픽스처만 통과하는 술어*라서다.
"""

import glob
import gzip
import io
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F          # noqa: E402

fail = []


def check(name, ok, detail=""):
    print("  %-58s %s%s" % (name, "PASS" if ok else "FAIL", (" — " + detail) if detail else ""))
    if not ok:
        fail.append(name)


def legacy(on):
    """수리 전 판정으로 되돌린다(양성대조용 스위치)."""
    if on:
        os.environ["T2_FORENSIC_DENY_LEGACY"] = "1"
    else:
        os.environ.pop("T2_FORENSIC_DENY_LEGACY", None)


# ─────────────────────────────────────────────────────────────────────────────
print("① 술어 — 무엇을 거절로 보는가 (A-7⑵ 원판·거동 보존)")
legacy(False)
check("env `Error:` 는 거절", F.deny_kind("Error: nope")[0] == "env")
check("env `Failed to …` 도 거절", F.deny_kind("Failed to log verification: x")[0] == "env")
check("우리 층 거절은 ours 로 남는다",
      F.deny_kind("[READ-FIRST] fetch it first")[0] == "ours")
check("성공 본문은 거절이 아니다", F.deny_kind('{"ok": true}')[0] == "")
# 접두만 본다 — 본문 **안에** 그 말이 있는 성공 결과를 거절로 접으면 반대 방향 오분류다.
check("본문 중간의 같은 말은 거절이 아니다",
      F.deny_kind('{"note": "Failed to open account is a possible error"}')[0] == "")

# ─────────────────────────────────────────────────────────────────────────────
print("\n② 양성대조 — 수리 **전** 판정에서 결함이 실제로 재현되는가")
# 재현이 안 되면 이 수리는 없는 병을 고친 것이다. 같은 코드에서 스위치만 돌려 확인한다.
legacy(True)
check("옛 판정: 우리 게이트 거절을 env 로 찍는다",
      F.deny_kind("Error: [POLICY GATE G1_AUTH_FIRST] identity not established")[0] == "env")
check("옛 판정: 중복-읽기 스텁을 거절로 아예 안 센다",
      F.deny_kind("[DUPLICATE-READ] This exact call was already executed earlier")[0] == "")
legacy(False)

# ─────────────────────────────────────────────────────────────────────────────
print("\n③ 수리 — 우리가 저작한 거절은 전부 ours 로 간다")
OURS_BODIES = [
    # 동적 표지(코드에 리터럴이 없다·최대 덩어리 764 건)
    "Error: [POLICY GATE G1_AUTH_FIRST] identity not established",
    "Error: [POLICY GATE G4_TRANSFER_MSG] say goodbye before transferring",
    "Error: [POLICY GATE RETRY_LOOP] the same call failed twice already",
    # 엔진 모듈이 저작한 문면
    "Error: [BYREF] could not resolve the @last: reference",
    "Error: [ARGS-FORMAT] the rows argument could not be read as JSON",
    "Error: [READ-FIRST] this calculation depends on records you have not read",
    "Error: [PRE-ACTION-KB] STOP before executing this tool",
    "Error: [DISCOVERY] you have read records but have not called the follow-up",
    # A2 선언이 저작한 문면
    "Error: [RESULT-SIGN] this correction computes to -3.00",
    "Error: [WRITE-EVIDENCE] no tool output in this conversation supports it",
    "Error: [FOLLOW-UP] the credit limit increase procedure is NOT complete",
    # `Error:` 를 안 쓰는 거절 스텁(2,554 건이 여기서 왔다)
    "[DUPLICATE-READ] This exact call was already executed earlier",
    "[DUPLICATE-COMPUTE] This exact call was already executed",
    "[NEAR-DUPLICATE-READ] This query is nearly identical to an earlier one",
    # 소유 원장으로만 잡히는 긴 꼬리(`Error:` + 우리 표지)
    "Error: [T2_WRITE_EVIDENCE] required evidence not found for this write",
    "Error: [UNKNOWN-VALUE] you asserted a boolean you never read",
]
bad = [b for b in OURS_BODIES if F.deny_kind(b)[0] != "ours"]
check("우리 거절 %d 종이 전부 ours" % len(OURS_BODIES), not bad,
      "미분류: %s" % [b[:34] for b in bad[:3]])

# ─────────────────────────────────────────────────────────────────────────────
print("\n④ 부정대조 — 자를 반대 방향으로 틀지 않았는가")
# ⓐ env 저작 본문은 그대로 env 여야 한다(우리 것으로 끌어오면 결손을 우리가 삼킨다).
ENV_BODIES = [
    "Error: Account 'wl94k7m3p8' not found.",
    "Error: Unknown discoverable tool 'apply_for_credit_card'.",
    "Error: Invalid arguments: KnowledgeTools.open_bank_account_4821() missing 1",
    "Error: Insufficient funds. Source account balance is $2500.00",
    "Error: Tool 'deposit_check_3847' has not been given to you by the agent",
    "Failed to open account: preconditions not met",
]
bad = [b for b in ENV_BODIES if F.deny_kind(b)[0] != "env"]
check("env 저작 거절 %d 종은 여전히 env" % len(ENV_BODIES), not bad,
      "%s" % [(b[:30], F.deny_kind(b)[0]) for b in bad[:3]])

# ⓑ **우리 표지를 단 성공 본문**은 거절이 아니다. 표지 소유만 보고 접으면 성공한 호출이
#    `ok=False` 로 찍힌다 — 고치려던 것과 반대 방향의 같은 오분류다(코퍼스 849 건).
NOT_DENY = [
    ("[POLICY_QA] Yes — the policy allows it.\nEvidence (verbatim from KB): ...",
     "a2 `return_template` = 기능-서브가 **답한** 것 (341 건)"),
    ("[GROUNDING WARNING] 1 input value(s) could not be verified: apy.\n{'total': 12.5}",
     "성공 결과 **앞에 덧붙인** 주석 · 뒤에 원 출력이 그대로 붙는다 (508 건)"),
]
for body, why in NOT_DENY:
    check("성공 본문이 거절로 안 찍힌다 — %s" % body[:22], F.deny_kind(body)[0] == "", why)

# ⓒ 모르는 표지는 **모른다**고 한다 — env 로 단정하면 그 칸이 다시 오귀속의 근거가 된다([[25]]).
check("원장에 없는 표지는 unknown (env 로 단정 안 함)",
      F.deny_kind("Error: [NEVERSEEN-TAG] something no file of ours wrote")[0] == "unknown")

# ─────────────────────────────────────────────────────────────────────────────
print("\n⑤ 원장이 **손 목록이 아닌가** — 파일이 바뀌면 자[尺]도 따라오는가")
led = F.ours_deny_prefixes()
check("원장이 비지 않았다", len(led) >= 20, "%d 항목" % len(led))
check("모든 항목이 출처 파일을 댄다", all(bool(v) for v in led.values()))
check("동적 표지가 접두로 잡혔다", "Error: [POLICY GATE" in led)
# ★핵심 검정: 원장을 이 파일에 **적어 둔 것이 아니라 소스에서 읽는다**는 것을 증명한다.
#   가짜 엔진 모듈 하나를 만들고 유리(glob)만 그쪽으로 돌려 새 표지가 저절로 따라오는지 본다.
tmp = tempfile.mkdtemp(prefix="denyledger_")
try:
    with io.open(os.path.join(tmp, "t2_fake_gate.py"), "w", encoding="utf-8") as f:
        f.write("# -*- coding: utf-8 -*-\n"
                "def deny(tc):\n"
                "    return ToolMessage(id=tc.id, role='tool',\n"
                "                       content='Error: [ZZ-BRAND-NEW-GATE] blocked for %s' % tc)\n")
    os.makedirs(os.path.join(tmp, "a2"))
    with io.open(os.path.join(tmp, "a2", "x.json"), "w", encoding="utf-8") as f:
        json.dump({"gates": [{"feedback": "Error: [ZZ-A2-NEW] do X first, then retry."}]}, f)
    old_here = F.HERE
    try:
        F.HERE = tmp
        fresh = F.ours_deny_prefixes(refresh=True)
        check("새 엔진 표지가 원장에 저절로 들어온다",
              "Error: [ZZ-BRAND-NEW-GATE]" in fresh, "%d 항목" % len(fresh))
        check("새 A2 표지가 원장에 저절로 들어온다", "Error: [ZZ-A2-NEW]" in fresh)
    finally:
        F.HERE = old_here
        F.ours_deny_prefixes(refresh=True)          # 원장 복구
finally:
    shutil.rmtree(tmp, ignore_errors=True)
check("복구 후 원장이 원래 크기", len(F.ours_deny_prefixes()) == len(led))

# ─────────────────────────────────────────────────────────────────────────────
print("\n⑥ 실물 — 이 코퍼스에서 그 접두가 실제로 env 실패인가 (A-7⑵ 원판)")
pats = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                    "sim_results", "bank_t73*_2026*.results.json.gz")
files = sorted(glob.glob(pats))
if not files:
    print("  · 코퍼스 없음 — 실물 검정 skip")
else:
    shapes, n = set(), 0
    for p in files:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        for s in (d.get("simulations") or []):
            for m in (s.get("messages") or []):
                if m.get("role") != "tool":
                    continue
                b = " ".join(str(m.get("content") or "").split()).lstrip()
                if b.startswith("Failed to "):
                    n += 1
                    shapes.add(b.split(":")[0])
    check("`Failed to ` 본문이 실재한다", n > 0, "%d건" % n)
    # 이 접두로 잡히는 것이 전부 실패 서술인지 눈으로 확인 가능한 수의 형상인가.
    check("형상이 소수의 실패 동사구뿐", len(shapes) <= 5, " / ".join(sorted(shapes)))

# ─────────────────────────────────────────────────────────────────────────────
print("\n⑦ 실물 — 수리 전/후를 **같은 코드로** 대조한다 (귀속이 어느 쪽으로 움직였나)")
try:
    corpus = F.all_result_files()
except Exception:
    corpus = []
corpus = [p for p in corpus if "bank_" in os.path.basename(p)][-40:]
if not corpus:
    print("  · 코퍼스 없음 — 실물 검정 skip")
else:
    moved = {}
    seen = 0
    for p in corpus:
        try:
            sims = list(F.sims(p))
        except Exception:
            continue
        for s in sims:
            for m in (s.get("messages") or []):
                if m.get("role") != "tool":
                    continue
                b = " ".join(str(m.get("content") or "").split())
                if not b:
                    continue
                seen += 1
                legacy(True)
                o = F.deny_kind(b)[0] or "-"
                legacy(False)
                nw = F.deny_kind(b)[0] or "-"
                if o != nw:
                    moved[(o, nw)] = moved.get((o, nw), 0) + 1
    print("     본문 %d 건 · 이동: %s" % (seen, dict(moved)))
    # 허용되는 이동은 **두 방향뿐**이다. 다른 이동이 하나라도 있으면 자를 반대로 튼 것이다.
    illegal = {k: v for k, v in moved.items() if k not in (("env", "ours"), ("-", "ours"))}
    check("이동 방향이 env→ours · 무판정→ours 뿐", not illegal, "위반 %s" % illegal)
    check("우리 거절이 실물에서 회수됐다", sum(moved.values()) > 0,
          "회수 %d 건" % sum(moved.values()))
    check("ours 를 잃은 본문이 없다", not any(k[0] == "ours" for k in moved))
    check("env 로 새로 밀린 본문이 없다", not any(k[1] == "env" for k in moved))

print("\nRESULT: %s" % ("ALL PASS" if not fail else "FAIL %s" % fail))
sys.exit(1 if fail else 0)
