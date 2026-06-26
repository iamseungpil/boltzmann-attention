# HANDOFF 2026-06-26 PM — load theory arc 종결 + Paper 1(=병합 1+4) md/tex/pdf 완성 + compliance strong-form settled + high-scale sweep 요청

> **진입 = `06-NOW`(단일진실원) + `EXPERIMENT_DESIGN §0★★★(PM)`(마스터) + `papers/paper1_capability_scale_lever/what_scale_buys.md`(논문 본문) + `MAKEORBREAK_VERDICT_2026_06_26`.** 직전 = `HANDOFF_2026_06_26`(AM·make-or-break). ★전부 gpt-4.1 0(로컬)·결과 영속화=`reports/facet_rft_2026/sim_results/`.

## 0. 이 세션 서사 (한 단락)
AM의 make-or-break(operand SFT NO-GO·잔여=orchestration) 위에서 **load theory**를 세웠다: 부하=measurable 5-feature·**binding 차원집합이 scale 함수**(7B/14B 4→32B 2·interference/state 먼저 은퇴·length/conditional 잔존)·통제 probe가 관측 confound 적발(L_interf). **★핵심 = F3/F4 compliance scale-invariance**(g1 붕괴 57→6 vs g2 잔존 31→41·gate 모든 scale 0). 사용자 directive로 **4-편 논문 포트폴리오**를 GitHub `papers/`에 생성→**Paper 1과 4를 인과로 병합**(compliance scale-invariant=비용전략 성립 이유). 권위 related-work(`RELWORK_AND_DIRECTION`)에 정렬·229-ref 서지·md/tex/pdf 3형태. 리뷰(사용자) 후 **무료 g2 per-write rate 측정→strong-form 생존**(제출 선결 통과)·R1-R8 라벨 수정·**두 동기(규제/compliance·싼A2/멀티도메인) 명시**. 마지막 = **high-scale compliance sweep(32B-fp16+72B+235B) coworker 요청**(첫 유료·승인됨).

## 1. ★최상위 결과 (정본)
- **compliance is scale-invariant = STRONG form settled** (`sim_results/g2_per_opportunity_rate_2026_06_26`): g2(confirm-before-write) **per-write-opportunity rate가 scale 무관 flat** — 7B 0.103[.080,.132]·14B 0.070·32B 0.075[.058,.097]·CI 겹침. 절대count(31→41) 상승은 write수(526→680) 때문(confound)·rate 정규화하면 flat. → **제목 strong-form 방어됨**(차등형 fallback 불요). gate=모든 scale 위반0.
- **load 분해**(`load_obs`·`load_graded_probe`·`plan_probe`): 차원 은퇴 순서·관측 confound(통제 필수)·orchestration 잔여=plan/execute 분리로 절반 닫힘·절반 planning miss(→Paper 3).
- **헤드라인**: scale은 capability를 사지만 guarantee 못 삼 → frontier도 gate 필요 → 소형+gate 신뢰성 대등 → 비용우위(TCO ~23×·fleet 2.1× [EST]). cost knee.

## 2. ★논문 (canonical·GitHub `papers/`·branch facet-rft-2026)
- **Paper 1 = 병합 1+4** `paper1_capability_scale_lever/what_scale_buys.{md,tex,pdf}` + `references.bib`(229) + `REFERENCES.md`.
  - 제목: *What Scale Buys in Tool-Use Agents, and How to Buy It Cheaply: A Capability×Scale×Lever×Cost Map Where Compliance Is Scale-Invariant*.
  - 구조: abstract→§1(동기 2종+목적함수+기여6)→§2 relwork(권위맵 정렬·ToolOrchestra 좁힘·인지아키텍처)→§3 framework→§4 method→§5(operand/load/**compliance**/cheap-repl/map/cost)→§6 lever allocation→§7 funded plan(≤$1k)→§8-10.
  - latex→pdf = **Overleaf**(로컬 TeX 없음·정적 정합만 확인·`pdflatex what_scale_buys;bibtex;pdflatex×2`).
- **Paper 2/3/4** = proposal. #2=A2생성(NL→GATE_SPEC·NL2CA선행)·#3=path-selection(CDP·학술버전만·[[32]])·#4 비용=병합돼 #1에 흡수(README 참조).
- **README**(`papers/README.md`)=3-편 포트폴리오(1+4 병합)·골격·정본문서.
- **빌드도구**: `papers/build_pdf.py`(md→html→xhtml2pdf). `*.pdf binary`(`.gitattributes`).

## 3. ★다음 (우선순위)
1. **[유료·요청됨] high-scale compliance sweep 회수**: `COWORKER_REQUEST_2026_06_26_highscale_compliance.md` — {32B-fp16(quant통제)·72B·235B} × floor+g15. 회수 시 `t2_compliance.py`+`g2_rate.py`로 F3/F4+g2/write rate → §5.3에 점 추가(strong 확장 or ≤32B 한정·둘다 정직반영). 예산통제: floor nt1 먼저(~$45-120).
2. **[무료·R5] SOPBench compliance 둘째 점**: 단일벤치 약점(리뷰 R5). **단 SOPBench 위반 taxonomy=dirgraph/constraint ≠ tau2 g2(confirm)** → 별 driver·정의 매핑 필요(무료 one-liner 아님). §9 단일벤치 한계로 정직 보고 중.
3. **[무료] §7 나머지**: 7B/14B scale축(같은 scaffold·learn 재취득 경계)·cheap-replication ΔL 독립추정·multi-field 전이(부분).
4. **[제출 전] refs `% UNVERIFIED` 재확인**(post-cutoff·내가 보완한 4 벤치 entry·τ²-bench id 2406 vs 2506).
5. Paper 2/3/4 = 각자 실험 프로그램(별도).

## 4. 자산 (commit·push됨·branch facet-rft-2026)
- **도구(전부 무료·gpt-4.1 0)**: `tau2/`(load_obs·load_graded_probe·plan_probe·**g2_rate**·t2_compliance·claude_user_batch·operand_controlled·escape_det_census)·`papers/build_pdf.py`.
- **결과 영속화**: `sim_results/`(g2_per_opportunity_rate·load_obs_multiscale·load_graded_interf·plan_probe_phase0·f3f4_scale_invariant_compliance·cu_batch3_failall).
- **정본 doc**: `LOAD_THEORY_DESIGN`·`ORCHESTRATION_CAPABILITY_LEVER_DESIGN`·`MAKEORBREAK_VERDICT`·`RELWORK_AND_DIRECTION`(권위 relwork맵·make-or-break whitespace-shift 배너)·`REGULATORY_DETERMINISM_SOURCING`(M1 정밀). 마스터=`EXPERIMENT_DESIGN §0★★★(PM)`.

## 5. In-flight / 환경
- **32B vLLM 서버(GPU0:8360)·7B(GPU1:8361) 가동 중**(load probe용). 다음 세션 불필요시 정리(`nvidia-smi`로 PID·`pkill -f` 금지=ssh부모죽임).
- coworker sweep = 외부(coworker)·회수 대기.
- 미완: high-scale sweep 수치·R5 SOPBench·§7 무료항목·refs 재검증.

## 6. 불변 (행동 전·[[06]] 단일진실원)
- [[05]] A2만 도메인특화·scaffold 도메인분기0 · [[11]] tau2 학습0(전이=A2-swap) · [[08]] 집계→결론 직행 금지·전수포렌식(g2-rate가 그 실증: 절대count→strong오인 직전 per-write 정규화로 교정) · [[09]] gpt-4.1 0 로컬 먼저·유료=확인1회·승인후(sweep=첫 유료) · [[06]] pass^1 노이즈·robust만 · [[30]] 리모트·영속화·distinct tag · [[32]] CDP/특허 GitHub 금지(Paper 3 학술버전만).
- ★리뷰 규율(이 세션 반복): "약점은 데이터 아니라 라벨" — strong주장/SETTLED/measured를 본문 caveat과 *반 칸* 앞서지 말 것. 미구현→[EST]·추정→[EST] 꼬리표.
