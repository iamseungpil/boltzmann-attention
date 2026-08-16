# -*- coding: utf-8 -*-
r"""x342 — savings 오답의 축 가르기: **부하(표현)인가 [[63]] 빼기 결손인가**. `t2_probe` 정본 호출.

## 사건 (t7304·원장 C505)

배달은 됐다(savings 본문 40,649자 도달 4/8) · gold `Silver Plus` 는 **배달물 안에 5회** 있고
**커밋 궤적에도 8/8 sim** 등장한다. 그런데 gold 0/8. 부하 0 인 격리 서브(ctl)도 `Gold Account`
**8/8** 오답. 결정적 축자(`s554706` msg47):

    "the **Gold Account** … a minimum balance requirement of **$10,000**,
     which is manageable for your current budget"

손님은 *"five to six grand, maybe seven max"* 라고 말했다 ⇒ **위반 사실을 스스로 말하면서
그 후보를 고른다**. 정보 부족이 아니라 **배제 실패**([[63]]).

문서 축자로 요구를 만족하는 클래스는 **`Silver Plus` 유일**(인출 15 · 최소잔액 $2,500 ·
APY 3.0/4.5 · daily). 오답들은 **각각 한 조건에서만** 탈락한다(Green 인출 8 · Gold 잔액 $10,000).

## 셀 4 (컷 = 손님이 savings 요구를 진술한 **직후** · 개설 write **직전**)

    A_REF      개입 없음                         ← 이 자리에서 스스로 내는가(기준선)
    B_TABLE    같은 숫자를 **표로 압축**          ← 부하 축소만(판단 0·제거 0)
    C_REMOVE   요구 **위반 후보를 지운** 목록      ← [[63]] 제거 레버(이유 병기·최종 선택은 모델)
    D_NEG      **무관한 이유로** 후보를 지움       ← ★부정통제(오르면 계기 무효)

## 판정 (사전 고정 · 잡음 바닥 ±4 ⇒ 차 ≥5 만 인용)

    A_REF ≥18                  → 이 자리에선 이미 낸다 ⇒ 라이브 실패는 다른 곳(전달·경합)
    A 낮음 ∧ B ≥18             → **부하(표현)** ⇒ 제거기 짓지 마라([[62]]②·표만 주면 된다)
    A≈B 낮음 ∧ C ≥18           → **[[63]] 빼기 확정** ⇒ 제거 레버 정당(단 최종 선택은 모델)
    전부 낮음                  → 이 자리에선 아무도 못 연다 ⇒ 다른 축
    D_NEG ≥18                  → 부정통제 실패 = 이 프로브 무효

⚠표·목록의 숫자는 **env 문서에서 실행 시각에 파싱**한다(하드코딩 0). 요구 3항은 손님 발화
  축자에서 사람이 읽어 상수로 두되 **인용을 함께 인쇄**한다 — 실제 레버라면 이 형식화는
  LLM 이 하고 엔진은 substring 검산만 한다([[66]] 인용-근거·[[22]]).
⚠[[62]]③ 경계: `C_REMOVE` 가 열려도 **후보가 하나만 남는 배치**는 엔진이 답을 낸 것과 같다.
  그래서 제거는 *"손님이 축자로 말한 제약을 위반하는 후보만"* 으로 좁히고 생존 후보를 **전부**
  보여 준다. 이 프로브는 그 배치가 실제로 여는지까지만 잰다.

실행(리모트·8141):
  T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x342_savings_subtract_iso.py [k] [nb]
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_probe as P                                              # noqa: E402

TAG, TASK = "bank_t7304_treat_20260816j", "task_055"
CUT = 50            # s554706: 손님 savings 요구 msg 48 → 개설 시도 msg 51 사이
DOCS = "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge/documents"

# 손님 축자(궤적에서 읽음 · 실행 시 인쇄해 눈으로 검산한다)
REQ = [("잔액 상한", 7000, "five to six grand, maybe seven max"),
       ("월 인출 횟수", 12, "tap into my savings 3-4 times a week"),
       ("APY 하한", 3.0, "at least 3% APY")]

MARKS = {"SILVERPLUS": "Silver Plus", "GREEN": "Green Account", "GOLD": "Gold Account"}
ASK = ("\n[instruction] Recommend exactly ONE savings account class for this customer. "
       "Answer with the class name only.")


def facts():
    """env 문서에서 savings 클래스별 (최소잔액·월 무료인출·APY) 파싱 — 하드코딩 0.

    ★2026-08-16 1차 실행에서 **내 계기가 틀렸다**: 산문 전체를 훑고 `max()`/`min()` 으로 뭉갠
      탓에 `Green Account (savings)` 의 무료인출이 **8 대신 20** 으로 들어갔다(판별 필드 바로
      그 칸). 오답을 지지하는 표를 처치로 주고 *"안 열린다"* 고 말할 뻔했다([[25]]·[[55]]).
      ⇒ 값은 **클래스 자신의 파이프 표 행**(`| Ongoing minimum balance | $2,500 |`)에서만 읽고,
      산문의 조건부 값(*"reduced to $5,000 for … holders"*)은 `cond` 로 **따로** 담는다.
    ⚠이건 분석 프로브라 정규식을 쓴다. **레버로 옮길 때는 금지**([[59]]) — 값은 A2 카탈로그에
      정책 축자 출처와 함께 선언되어야 한다(기존 `catalog_filter` 패턴).
    """
    out = {}
    for fn in sorted(os.listdir(DOCS)):
        if not fn.startswith("doc_savings_accounts_"):
            continue
        k = re.sub(r"_\d+$", "", re.sub(r"^doc_savings_accounts_", "",
                                        re.sub(r"\.json$", "", fn)))
        c = (json.load(io.open(os.path.join(DOCS, fn), encoding="utf-8")).get("content") or "")
        e = out.setdefault(k, {"wd": set(), "mb": set(), "apy": set(), "cond": set()})
        for m in re.finditer(r"\|\s*(?:Maximum |Monthly )?[Ff]ree withdrawals per month\s*\|\s*(\d+)", c):
            e["wd"].add(int(m.group(1)))
        for m in re.finditer(r"\|\s*Monthly withdrawal limit\s*\|\s*(\d+)", c):
            e["wd"].add(int(m.group(1)))
        for m in re.finditer(r"\|\s*(?:Ongoing m|M)inimum balance[^|]*\|\s*\$([\d,]+)", c):
            e["mb"].add(int(m.group(1).replace(",", "")))
        for m in re.finditer(r"\|\s*APY\s*\|\s*([\d.]+)%", c):
            e["apy"].add(float(m.group(1)))
        for m in re.finditer(r"(?:Tier \d(?: APY)?|You earn|earns an APY of)[^0-9%]{0,12}([\d.]+)%", c):
            e["apy"].add(float(m.group(1)))
        for m in re.finditer(r"reduced (?:from \$[\d,]+ )?to \$([\d,]+)[^.]{0,80}", c):
            e["cond"].add("min balance $%s only for %s" % (m.group(1), m.group(0)[-60:].strip()))
    return out


def disp(k):
    return " ".join(w.capitalize() for w in k.replace("_", " ").split())


def table_text(F_, keys=None):
    rows = ["| account class | min balance | free withdrawals/mo | APY |", "|---|---|---|---|"]
    for k in sorted(keys or F_):
        e = F_[k]
        rows.append("| %s | %s | %s | %s |" % (
            disp(k),
            ("$%s" % min(e["mb"])) if e["mb"] else "—",
            (max(e["wd"]) if e["wd"] else "—"),
            (", ".join("%s%%" % a for a in sorted(e["apy"]) if a >= 1.0) or "—")))
    return "\n".join(rows)


def violates(e):
    """요구 위반 사유(닫힌 술어·전부 문서 값 대 손님 축자 값 비교)."""
    why = []
    if e["mb"] and min(e["mb"]) > REQ[0][1]:
        why.append("minimum balance $%s exceeds the stated maximum $%s"
                   % (min(e["mb"]), REQ[0][1]))
    if e["wd"] and max(e["wd"]) < REQ[1][1]:
        why.append("%d free withdrawals/month is fewer than the stated need (%d+)"
                   % (max(e["wd"]), REQ[1][1]))
    good = [a for a in e["apy"] if a >= 1.0]
    if good and max(good) < REQ[2][1]:
        why.append("top APY %.1f%% is below the stated floor %.1f%%" % (max(good), REQ[2][1]))
    return why


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    nb = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 3
    F_ = facts()
    if len(F_) < 5:
        print("문서 파싱 실패(%d 클래스) — 중단(계기 결함)" % len(F_))
        return 1

    kept, removed = [], []
    for key in sorted(F_):
        why = violates(F_[key])
        (removed if why else kept).append((key, why))

    print("x342 · %s/%s · cut=%d" % (TAG, TASK, CUT))
    print("요구(손님 축자):")
    for name, val, quote in REQ:
        print("   · %-10s %-8s ← \"%s\"" % (name, val, quote))
    print("\n생존 후보 %d: %s" % (len(kept), ", ".join(disp(x[0]) for x in kept)))
    print("제거 후보 %d:" % len(removed))
    for key, why in removed:
        print("   − %-26s %s" % (disp(key), " · ".join(why)))
    if len(kept) == 1:
        print("⚠[[62]]③ 경고: 생존 후보가 **하나**다 — C_REMOVE 가 열리면 그것은 엔진이 답을 "
              "낸 것과 구분되지 않는다. 판정문에 반드시 병기할 것.")
    print()

    site = P.site(TAG, TASK, CUT)
    tbl = table_text(F_)
    keep_txt = "\n".join(
        "- %s (min balance $%s · %s free withdrawals/mo · APY %s)"
        % (disp(key), min(F_[key]["mb"]) if F_[key]["mb"] else "—",
           max(F_[key]["wd"]) if F_[key]["wd"] else "—",
           ", ".join("%s%%" % a for a in sorted(F_[key]["apy"]) if a >= 1.0) or "—")
        for key, _ in kept)
    rem_txt = "\n".join("- %s — removed: %s" % (disp(key), "; ".join(why))
                        for key, why in removed)
    # 부정통제: 위반 여부와 **무관한** 규칙으로 지운다(이름 길이) — 같은 개수를 지운다.
    neg_removed = sorted(F_, key=lambda x: (-len(x), x))[:len(removed)]
    neg_kept = [x for x in sorted(F_) if x not in neg_removed]

    P.run("x342", site, [
        ("A_REF", ""),
        ("B_TABLE", "[reference] Documented parameters of the savings account classes:\n" + tbl),
        ("C_REMOVE", "[reference] Classes that do NOT meet what the customer stated "
                     "(excluded, with the reason):\n" + rem_txt
                     + "\n\nRemaining classes that meet everything stated:\n" + keep_txt),
        ("D_NEG", "[reference] Classes excluded from consideration:\n"
                  + "\n".join("- %s — excluded" % disp(x) for x in neg_removed)
                  + "\n\nRemaining classes:\n"
                  + "\n".join("- %s" % disp(x) for x in neg_kept)),
    ], MARKS,
        "A≥18 → 이 자리에선 이미 낸다(라이브 실패는 다른 곳) · A낮음∧B≥18 → **부하(표현)**"
        " ⇒ 제거기 짓지 마라 · A≈B낮음∧C≥18 → **[[63]] 빼기 확정** ⇒ 제거 레버 정당"
        "(단 생존 후보가 1개면 [[62]]③ 경고 병기) · 전부 낮음 → 다른 축 · D_NEG≥18 → 계기 무효",
        ASK, None, k, nb)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
