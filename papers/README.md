# Papers — portfolio (tool-use / small-model + deterministic offload)

> 단일 진입점. 연구 기록·도구·결과는 `../scripts/distill/`(권위)·`../reports/facet_rft_2026/sim_results/`(결과). 이 디렉터리 = 논문화. branch `facet-rft-2026`.
> **2026-06-26: Papers 1+4 병합** → 과학(scale이 무엇을 사나)과 경제(그래서 어떻게 싸게 사나)가 *한 인과 호*(compliance scale-invariant → 모든 scale에 gate 필요 → 소형+gate가 신뢰성 대등 → 비용우위)로 완결. 포트폴리오 = **3편**.

## 포트폴리오 골격

| # | 디렉터리 | 성격 | 한 줄 | 정본 설계문서 |
|---|---|---|---|---|
| **1 (=1+4)** | `paper1_capability_scale_lever/` | **이론·지도 + 시스템·비용** (ICLR 타깃·메인) | 기능×스케일(7B–235B)×레버×**비용** 지도: scale이 *무엇을 사고* / LLM이 *무엇을 진짜 못하고*(compliance=scale-invariant) / scaffold+A2가 *무엇을 싸게 메우나* + **총비용 knee·TCO ~23×**. 벤치=측정도구 | `EXPERIMENT_DESIGN §0★★/★★★`·`LOAD_THEORY_DESIGN`·`MAKEORBREAK_VERDICT`·`CAPABILITY_LEVER_ALLOCATION`·`TCO_TABLE_DESIGN` |
| **2** | `paper2_a2_generation/` | 학습법 (proposal) | NL 정책 → `GATE_SPEC` 컴파일러 학습. #1이 *given*으로 쓰는 A2를 *만드는* 방법(공통 적 A2 비용 절감) | `A2_FRONTEND_DISTILL_DESIGN` |
| **3** | `paper3_path_selection/` | 학습법·직교축 (proposal) | 도구폭발 하 path-search를 *전이가능 학습 휴리스틱*으로(탐색=offload·LLM=②적용op인식·③value). #1(provenance축)과 직교 | `PATH_SELECTION_AXIS_DESIGN` |

## Paper 1 — 확정 (ICLR 메인)
- **제목**: *What Scale Buys in Tool-Use Agents, and How to Buy It Cheaply: A Capability×Scale×Lever×Cost Map Where Compliance Is Scale-Invariant*
- **파일(canonical)**: `paper1_capability_scale_lever/what_scale_buys.{md, tex, pdf}` + `references.bib`(229+) + `REFERENCES.md`(테마별) + `refs_dr.md`·`refs_lit.md`(추출 원본).
- **PDF**: `what_scale_buys.pdf`(md→xhtml2pdf 렌더). latex→pdf = Overleaf 컴파일(`pdflatex what_scale_buys; bibtex; pdflatex ×2`).
- **빌드**: `../build_pdf.py out.pdf what_scale_buys.md REFERENCES.md`.

## 구조 논리
- **#1 = 과학(scale이 무엇을 사나) → 경제(그래서 어떻게 싸게 사나)** 한 호. P1의 핵심발견(compliance scale-invariant)이 *바로* 비용전략이 성립하는 이유 → 1+4 병합이 인과로 옳음.
- **#2·#3 = #1 지도의 "학습이 답"인 칸들의 방법** — #2(A2 생성)·#3(path 휴리스틱). 둘 다 #1 프레임 인용·별도 양성결과 필요라 분리 유지.
- **#1·#3 직교**: #1 tau2 = provenance/grounding 축 / #3 = path-search 축. *섞지 않음.* load 발견(L_branch=scaffold-저항)이 #1→#3 handoff.

## 상태 (2026-06-26)
- **#1 (=1+4)**: **초안 완성**(md/pdf/latex). 백본 = make-or-break + load 분해 + **F3/F4 compliance scale-invariant**(★g2 per-opportunity rate+CI = §7-1 무료·**제출 선결**) + 비용(TCO ~23×·fleet 2.1×·knee). §7 funded forward plan(≤$1k). [EST]=72B·235B·multi-bench 전이.
- **#2**: proposal. S0 창발·S1 기각·dose-response 설계까지 실측.
- **#3**: proposal(학술버전만·**CDP/특허 specifics는 GitHub 금지**·confidential은 로컬).

## 규율
- 헤드라인 = 결정론 scaffold + 작은 base + A2 + 비용. **"소형>대형"은 rival(ToolOrchestra 2511.21689)이 선점 → 우리 = method + "scale이 못 푸는 guarantee" + 비용 knee.**
- 결과 인용 = `sim_results/`(전부 gpt-4.1 0). 추정은 [EST] 명시. refs `% UNVERIFIED`는 제출 전 재확인.
