# -*- coding: utf-8 -*-
"""회수 궤적(role=tool)의 KB_search 출력에서 문서 블록을 걷어 로컬 코퍼스 스냅샷을 만든다.
탐색용 프로브(엔진 아님). 블록 경계 = `N. 제목` **다음 줄이 `ID:`, 그 다음이 `Score:`** 인 헤더.
헤더 위치로 split 하므로 본문의 번호 목록(`1. …`)에 걸리지 않는다."""
import gzip, glob, json, re, io
OUT = "reports/facet_rft_2026/x767_corpus_snapshot.json"
HDR = re.compile(r"^[ \t]*\d+\.[ \t]*(.*)\n[ \t]*ID:[ \t]*(\S+)\n[ \t]*Score:[ \t]*[-\d.eE]+\n[ \t]*Content:[ \t]*",
                 re.M)
TAIL = re.compile(r"\n\[(?:Timing|axis)[^\n]*$")
corpus, titles, nblk = {}, {}, 0
nf = 0
for p in sorted(glob.glob("reports/facet_rft_2026/sim_results/*.results.json.gz")):
    try: raw = gzip.open(p, 'rt', encoding='utf-8', errors='replace').read()
    except Exception: continue
    if "   ID: doc_" not in raw: continue
    try: d = json.loads(raw)
    except Exception: continue
    nf += 1
    for s in d.get("simulations", []) or []:
        for m in s.get("messages", []) or []:
            if m.get("role") != "tool": continue
            c = str(m.get("content") or "")
            hs = list(HDR.finditer(c))
            if not hs: continue
            for i, h in enumerate(hs):
                end = hs[i + 1].start() if i + 1 < len(hs) else len(c)
                cont = c[h.end():end]
                cont = TAIL.sub("", cont.rstrip()).rstrip()
                did = h.group(2); nblk += 1
                if len(cont) > len(corpus.get(did, "")):
                    corpus[did] = cont; titles[did] = h.group(1).strip()
json.dump({"n": len(corpus), "n_blocks_seen": nblk, "titles": titles, "docs": corpus},
          io.open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print("files", nf, "blocks", nblk, "docs", len(corpus), "->", OUT)
