# -*- coding: utf-8 -*-
r"""x511 — **합쳐서 한 번 부르는가** 격리 (x509 S2 · 사용자 승인 순서 ①금액 → ②범주).

## 물음
`apply_checking_account_credit_5829` 는 정책상 **계좌당 한 번**이고 여러 보정은 **합쳐서** 한 건으로
내야 한다. 라이브에서 모델은 계좌당 1~7 회 나눠 부른다. x509 S2 의 물음: **전달로 닫히나.**

## 왜 산수가 아닌가 (최신 3런 실측)

    모드 B  분할하되 **합계는 정확**   073 t0: 3건 합 9.50 / gold 9.50 인데 **reward 0.0**
    모드 D  계좌당 1건               073 t0(t7346): 9.50·9.00·1.50 → **reward 1.0**

결손은 **덧셈이 아니라 write 의 모양(1 vs N)** 이고, 모델은 할 수 있다(D 가 실재).

## 재료 — **전부 궤적에서 축자로 뜯는다**

세 조각 모두 통과 sim(t7346 · 073 · t0)의 메시지 원문이다. 문서 id 를 코드에 적지 않는다
(그것은 도메인-특화 스캐폴드다·[[05]]·훅 ISOLATION CONTRACT). 재는 것은 *모델이 그 자리에서
실제로 받은 것*이므로, 궤적이 유일하게 옳은 출처다(t2_gap 규율과 동일).

    비교기 출력  `ATM fee lines whose charged amount does NOT match…` 로 시작하는 tool 메시지
    정책 문단    `may only be called ONCE per checking account` 를 담은 tool 메시지의 그 문단
    도구 스키마  `Parameters: - account_id` 를 담은 unlock 출력

못 찾으면 **돌리지 않는다** — 대체물을 지어 넣지 않는다([[25]]).

## 팔

    A_bare        비교기 출력만
    B_policy      + 정책 문단(궤적과 같은 자리 = 앞쪽)
    C_policy_last + 정책 문단을 **요구 직전**에 (최신성·타이밍)
    N_neg         + 정책과 **같은 대역·규칙 0** 인 문장(부정통제·[[57]])

## 규율

엔진이 합계를 계산해 주지 않는다 — 프로브는 모델이 낸 **호출 수와 금액**만 센다([[62]] ③).
gold 는 채점에만 쓴다([[23]]). 프롬프트에 정답 금액을 넣지 않는다.

## §격리 계약 자기점검 ([[71]] 4문 · 훅이 묻는다)

    1) 기능 하나인가        예 — 결정은 **"이 계좌에 크레딧을 몇 건으로 낼 것인가"** 하나다.
                            금액을 고르는 것도, 계좌를 고르는 것도 이 서브의 일이 아니다.
    2) 재료의 출처          **궤적 축자**. 문서 id 리터럴은 제거했다(초판이 그것으로 훅에 막혔다).
                            A2 선언이 아니라 궤적인 이유: 이 정책 문단은 A2 에 없고, 재려는 것이
                            *모델이 그 자리에서 실제로 받은 것*이라 궤적이 유일하게 옳은 출처다.
    3) 전달 방식            검색 0. bm25·embedding **안 쓴다**. 궤적에서 앵커 문구로 조각을 집을 뿐.
    4) 엔진의 해석·선택·순위 없음. 합계도 안 낸다. **호출 수와 금액을 세기만** 한다.

사용: py -3 x511_combine_credit_iso.py [--port 8141] [--n 8]
"""
import argparse
import collections
import gzip
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

MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
SIMS = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
SRC = os.path.join(SIMS, "bank_t7346_halfA_20260822.results.json.gz")

# 궤적에서 조각을 찾는 **앵커 문구**(검색 키일 뿐 · 판단 0). 문서 id 는 쓰지 않는다.
A_COMP = "ATM fee lines whose charged amount does NOT match"
A_POLICY = "may only be called ONCE per checking account"
A_SCHEMA = "Parameters: - account_id"
A_POLICY_END = "## Important Restrictions"


def materials():
    """통과 sim 의 메시지에서 세 조각을 **축자로** 뜯는다."""
    d = json.load(gzip.open(SRC, "rt", encoding="utf-8"))
    sim = None
    for s in (d.get("simulations") or []):
        if str(s.get("task_id")) == "task_073" and s.get("trial") == 0:
            sim = s
            break
    if sim is None:
        raise SystemExit("073 t0 없음 — 재료 출처가 없다")
    comp = schema = policy = None
    for m in (sim.get("messages") or []):
        c = " ".join(str(m.get("content") or "").split())
        if comp is None and c.startswith(A_COMP):
            comp = c
        if schema is None and A_SCHEMA in c and "apply_checking_account_credit" in c:
            schema = c[:900]
        if policy is None and A_POLICY in c:
            i = c.find(A_POLICY)
            j = c.find(A_POLICY_END, i)
            policy = c[max(0, i - 60):(j if j > i else i + 700)].strip()
    if not (comp and schema and policy):
        raise SystemExit("재료 미확보: comp=%s schema=%s policy=%s"
                         % (bool(comp), bool(schema), bool(policy)))
    return comp, schema, policy


# 부정통제 — 정책과 같은 대역이되 **합치라는 규칙은 한 글자도 없다**([[57]]).
FILLER = ("Credits are recorded in the account transaction history with a transaction identifier, "
          "the account identifier, the credit type and the credit amount. The resulting balance is "
          "reported back to the agent so that the customer can be informed of the new balance. "
          "Credit records are retained for audit and appear in subsequent transaction retrievals "
          "for the same account, alongside the fee lines that were charged earlier.")

SYS = ('You are a bank agent. Reply with ONE JSON object only: '
       '{"calls": [{"tool": "<tool name>", "account_id": "<id>", '
       '"amount": <number>, "credit_type": "<type>"}]}. '
       'No prose, no explanation.')


def ask(port, body, maxtok=700, temperature=0.0):
    payload = {"model": MODEL, "temperature": temperature, "max_tokens": maxtok,
               "messages": [{"role": "system", "content": SYS},
                            {"role": "user", "content": body}]}
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % port,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def parse(txt):
    """호출 목록만. 파싱 실패는 **실패로 센다** — 관대하게 굴면 결과가 부풀린다."""
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
    except Exception:
        return None
    cs = o.get("calls")
    return cs if isinstance(cs, list) else None


ARMS = ("A_bare", "B_policy", "C_policy_last", "N_neg")


def build(arm, comp, schema, policy):
    ask_line = "# What to do now\nApply the credit(s) for this account. Return the tool call(s)."
    head = "# Tool\n%s\n\n# Fee comparison result for this account\n%s\n" % (schema, comp)
    if arm == "A_bare":
        return head + "\n" + ask_line
    if arm == "B_policy":
        return "# Policy\n%s\n\n" % policy + head + "\n" + ask_line
    if arm == "C_policy_last":
        return head + "\n# Policy\n%s\n\n" % policy + ask_line
    if arm == "N_neg":
        return "# Policy\n%s\n\n" % FILLER + head + "\n" + ask_line
    raise SystemExit("unknown arm %r" % arm)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="롤아웃이므로 T>0 — 결정론 1회는 팔 사이 차이를 못 잰다")
    a = ap.parse_args(argv)
    comp, schema, policy = materials()
    print("재료 — 비교기 %d자 · 정책 %d자 · 스키마 %d자 · 부정통제 %d자"
          % (len(comp), len(policy), len(schema), len(FILLER)))
    print("비교기 축자: %s" % comp[:170])
    print("정책 축자  : %s" % policy[:170])
    out = {"arms": {}, "n": a.n, "temperature": a.temperature}
    for arm in ARMS:
        body = build(arm, comp, schema, policy)
        cnt = collections.Counter()
        rows = []
        for k in range(a.n):
            try:
                txt = ask(a.port, body, temperature=a.temperature)
            except Exception as e:
                cnt["오류"] += 1
                rows.append({"i": k, "error": repr(e)[:120]})
                continue
            cs = parse(txt)
            if cs is None:
                cnt["파싱실패"] += 1
                rows.append({"i": k, "raw": (txt or "")[:160]})
                continue
            ncall = len(cs)
            cnt["합침(1건)" if ncall == 1 else "분할(%d건)" % ncall] += 1
            rows.append({"i": k, "n": ncall,
                         "amounts": [c.get("amount") for c in cs if isinstance(c, dict)]})
        combined = sum(v for k, v in cnt.items() if k.startswith("합침"))
        print("== %-14s 합침 %d/%d   %s" % (arm, combined, a.n, dict(cnt)))
        for r in rows[:4]:
            print("     %s" % json.dumps(r, ensure_ascii=False)[:150])
        out["arms"][arm] = {"combined": combined, "counts": dict(cnt), "rows": rows}
    dst = os.path.abspath(os.path.join(SIMS, "..", "x511_combine_credit_iso.json"))
    with io.open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("=== 정산 (합침 = 계좌당 한 건) ===")
    for arm in ARMS:
        print("   %-14s %d/%d" % (arm, out["arms"][arm]["combined"], a.n))
    print("-> %s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
