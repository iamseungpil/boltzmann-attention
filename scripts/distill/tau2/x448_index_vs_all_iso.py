# -*- coding: utf-8 -*-
r"""x448 — **A2/A3 색인 + shell 정확 전달** ↔ **bm25·dense 검색 전달** (2026-08-21·격리·무료)

## 우리 방식의 정의 (사용자 축자·2026-08-21)
*"bm25나 embedding 이 아닌 A2 A3 를 이용한 정확한 격리 에이전트에 문서 전달 방식이 우리 방식이다."*
*"아니 shell 검색이다."* ⇒ **A2/A3 가 문서 id 를 선언하고, 그 id 를 `shell` 로 정확히 집어
격리 서브에이전트에 전달한다.** 유사도 검색이 아니다.
*"A2 A3 에서 하는 것은 어떤 서브에이전트에서 어떤 결정을 위해 필요한 문서가 뭔지 정확하게 기술하는
것만이다."* ⇒ 선언은 **무엇을 배달할지**만 말하고 어느 범주가 맞는지는 말하지 않는다.

## 이 프로브가 고치는 교락
C569(110편 전달·`x446`) ↔ C571(색인 12편 전달·`x447`)의 0/4 ↔ 4/4 는 **다른 스크립트·다른
프롬프트**에서 났다. 여기서는 모든 팔이 **같은 프롬프트·같은 계약·같은 인용 검산·같은 손님 발화**를
쓰고 **재료만** 다르다. 재료 바이트는 전부 **샌드박스 실물**(라이브와 같은 도구·같은 형식)에서 뽑는다.

## 팔 (전부 전달 관용구 — 서브에이전트는 판단만 한다)
    Z_none    문서 없음                                    현행 재현(C570: 결정 시점에 KB 가 없다)
    R_bm25    `KB_search_bm25(손님 발화, k=선언편수)`         벤치의 검색 — **예산 일치**
    R_dense   `KB_search_dense(손님 발화, k=선언편수)`        벤치의 검색 — **예산 일치**
    B_shell   **A2 선언 id → `shell: cat <id>.md`**          ★우리 방식(정확 전달)
    N_sham    선언 **밖** 같은 편수를 같은 `shell` 로          부정통제([[57]]) — *양*인가 *무엇*인가
    W_all     계열 문서 전부(문맥 초과 → 절단 표시)            참조: *"그냥 전체 주기"* 는 안 들어간다
  ⚠엔진은 **읽어 전달만** 한다. 해석·선택·순위 0([[59]]). 정책 문서 읽기가 우리 층 몫인 것은
    확정된 경계다(`t2_search.py` §경계 · C405ⓔ 사용자 결정). 규칙 0 은 **DB 도구 출력**에 대한 것이다
    (`SCAFFOLD_AUDIT_RULE0_2026_07_08` 축자).
  ⚠A2 색인의 출처는 **문서 제목**뿐·gold 미참조([[23]]). 오라클 팔(`x417.doc_slice` 계열)은 없다.

## 채점 (닫힌 술어만·[[59]]ⓐ)
    correct              참조 라벨 일치 (진단용·[[69]]·gold 카드에서 유도·pass 아님)
    quote_real           인용이 **그 팔이 받은 재료** 안에 실재하는가 — **형태만** 정규화(영숫자 외 전부
                         공백). 목록 구두점 차이로 같은 문장이 탈락하던 것을 고쳤다(1차 실측 결함).
    quote_from_customer  인용이 **손님 발화**에서 왔는가 — C569 가 걸린 실패 양식
  참조: 003→travel(gold `Silver Rewards Card`=travel 4%) · 024→None(gold `Business Bronze`) ·
        063→None(gold `Silver Rewards Card`).

사용: (리모트·cwd=tau2 · PYTHONPATH=src:...) py x448_index_vs_all_iso.py --port 8141
"""
import argparse
import glob
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

import x423_choice_isolation as I       # noqa: E402  케이스 원천(카드축)
import x431_spec_selects as X           # noqa: E402  ask 정본
import x437_declaration_isolation as P  # noqa: E402  손님 턴 추출
import x447_indexed_category_iso as IX  # noqa: E402  A2 선언 읽기(사본 금지·[[67]])
import x446_category_citation_iso as W  # noqa: E402  계열 전체 읽기(사본 금지·[[67]])
import x430_account_facts as FT         # noqa: E402  DOCDIR

REP = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026")
FAMS = ("credit_cards", "business_credit_cards")

# 참조 라벨 = gold 카드에서 유도(진단용). A2 저작에는 쓰이지 않았다([[23]]).
REF = {"003": "travel", "024": None, "063": None}

SYS = ("You are the module that decides ONE thing for a Rho-Bank support agent: which documented "
       "spending category the customer's spending belongs to. Reply with ONE JSON object only: "
       "{\"spend_category\": \"<one of: travel, software, operations, media_advertising, green>\" "
       "or null, \"quote\": \"<one sentence copied word for word from the '# Documents' section "
       "that shows this spending qualifies>\"}. The quote MUST come from the documents, never from "
       "the customer. If no document sentence supports a category, set spend_category to null.")

_ALNUM = re.compile(r"[^0-9a-z]+")


def form_norm(t):
    """**형태만** 정규화 — 영숫자 외 전부 공백. 목록 구두점·대시·강조 차이를 지운다([[59]] 안전)."""
    return " ".join(_ALNUM.sub(" ", str(t or "").lower()).split())


def as_cat(v):
    """모델이 문자열 `"null"`/`"none"` 을 내는 경우를 **None 과 같게** 본다(1차 실측 계기 결함)."""
    s = str(v or "").strip().lower()
    return None if s in ("", "null", "none", "n/a") else s


class Sandbox(object):
    """라이브와 **같은** 환경의 도구를 그대로 쓴다([[25]]). 엔진은 이 도구로 **읽기만** 한다."""

    def __init__(self):
        from tau2.domains.banking_knowledge.environment import get_environment
        self.env = get_environment(retrieval_variant="alltools")
        have = sorted(t.name for t in self.env.get_tools()
                      if t.name in ("KB_search_bm25", "KB_search_dense", "shell"))
        if have != ["KB_search_bm25", "KB_search_dense", "shell"]:
            raise SystemExit("도구 지형이 다르다: %r" % (have,))
        self.files = [x for x in str(self.env.use_tool("shell", command="ls")).split() if x]

    def file_for(self, doc_id):
        """선언된 id → 샌드박스 파일명. **우리가 이름을 짓지 않는다** — 실제 목록에서 찾는다."""
        for f in self.files:
            if f.rsplit(".", 1)[0] == doc_id:
                return f
        for f in self.files:
            if doc_id in f:
                return f
        return None

    def cat(self, doc_ids):
        """★우리 방식 — 선언된 id 를 `shell` 로 **정확히** 집어 온다(검색 아님)."""
        out, missing = [], []
        for did in doc_ids:
            f = self.file_for(did)
            if not f:
                missing.append(did)
                continue
            out.append(str(self.env.use_tool("shell", command="cat %s" % f)))
        return "\n\n".join(out), missing

    def search(self, tool, query, k):
        return str(self.env.use_tool(tool, query=query, k=k))


def sham_ids(declared_ids, n):
    """부정통제 — 선언 **밖**에서 같은 계열·같은 편수. 규칙 하나(이름 순)."""
    out = []
    for p in sorted(glob.glob(os.path.join(FT.DOCDIR, "doc_*credit_card*.json"))):
        did = os.path.basename(p)[:-5]
        if did in declared_ids:
            continue
        out.append(did)
        if len(out) >= n:
            break
    return out


def all_blob():
    ds = []
    for fam in FAMS:
        ds.extend(W.docs_for(fam))
    return "\n\n".join("### %s — %s\n%s" % (i, t, b) for i, t, b in ds), len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tag", default="iva2")
    ap.add_argument("--arms", default="Z_none,R_bm25,R_dense,B_shell,N_sham,W_all")
    ap.add_argument("--maxchars", type=int, default=90000)
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]

    sb = Sandbox()
    declared_ids = [d[0] for d in IX.index_docs()]
    K = len(declared_ids)
    b_blob, b_missing = sb.cat(declared_ids)
    n_blob, _ = sb.cat(sham_ids(set(declared_ids), K))
    w_blob, w_n = all_blob()
    if b_missing:
        print("⚠선언됐는데 샌드박스에 없는 문서: %r" % (b_missing,))

    print("=" * 100)
    print("x448 · 한 프롬프트·재료만 다름 · 예산 k=%d편 · 팔 %s" % (K, ",".join(arms)))
    print("   B_shell  %3d편 %7d자   (A2 선언 id → shell cat · ★우리 방식)" % (K, len(b_blob)))
    print("   N_sham   %3d편 %7d자   (선언 밖·같은 도구)" % (K, len(n_blob)))
    print("   W_all    %3d편 %7d자%s" % (w_n, len(w_blob), "  ⚠절단됨" if len(w_blob) > a.maxchars else ""))
    print("=" * 100)

    seen, cs = set(), []
    for c in I.cases(60):
        if c["arg"] != "card_type":
            continue
        k = (c["task"], c["trial"])
        if k in seen:
            continue
        seen.add(k)
        cs.append(c)

    rows = []
    for c in cs:
        said = " \n".join(P.turns(c))[:5000]
        ref = REF.get(str(c["task"]).split("_")[-1], "?")
        print("\n%s t%s   참조=%s" % (c["task"], c["trial"], ref))
        for arm in arms:
            if arm == "Z_none":
                body = ""
            elif arm == "R_bm25":
                body = sb.search("KB_search_bm25", said, K)
            elif arm == "R_dense":
                body = sb.search("KB_search_dense", said, K)
            elif arm == "B_shell":
                body = b_blob
            elif arm == "N_sham":
                body = n_blob
            else:
                body = w_blob
            cut = len(body) > a.maxchars
            body = body[:a.maxchars]
            head = ("# Documents\n%s\n\n" % body) if body else ""
            ans = X.ask(a.port, SYS, head + "# What the customer said\n%s\n" % said, maxtok=400) or {}
            cat = as_cat(ans.get("spend_category"))
            q = form_norm(ans.get("quote"))
            real = bool(q) and q in form_norm(body)
            from_cust = bool(q) and q in form_norm(said)
            rows.append({"task": c["task"], "trial": c["trial"], "arm": arm, "ref": ref,
                         "cat": cat, "quote": str(ans.get("quote") or "")[:240],
                         "quote_real": real, "quote_from_customer": from_cust,
                         "chars": len(body), "truncated": cut,
                         "correct": (cat == ref) if ref != "?" else None})
            print("   %-8s cat=%-16s 참조일치=%-5s 인용실재=%-5s 손님인용=%-5s %6d자%s"
                  % (arm, cat, (cat == ref) if ref != "?" else "-", real, from_cust, len(body),
                     "  ⚠절단" if cut else ""))
            if ans.get("quote"):
                print("        인용: %s" % str(ans.get("quote"))[:140])

    p = os.path.abspath(os.path.join(REP, "x448_%s.json" % a.tag))
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("\n" + "=" * 100)
    print("%-8s %-9s %-9s %-9s %-9s %s" % ("팔", "참조일치", "인용실재", "손님인용", "범주유지", "평균자수"))
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        print("%-8s %-9s %-9s %-9s %-9s %d"
              % (arm,
                 "%d/%d" % (sum(1 for r in rs if r["correct"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["quote_real"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["quote_from_customer"]), len(rs)),
                 "%d/%d" % (sum(1 for r in rs if r["cat"]), len(rs)),
                 sum(r["chars"] for r in rs) / len(rs)))
    print("→ %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
