# neurips2026_steering_ko

**Canonical 한국어 원고**. 영문판 `paper/neurips2026_steering_v2/`가 이 파일의 1:1 번역 mirror다.

## 중심 주장 (한 줄)

> 정지 key-side 증폭(SEKA, AdaSEKA)은 멀티-툴 도구 선택에서 방출된 도구 상태를 구조적으로 기억할 수 없다(정리 4.1). 우리는 이 한계를 우회하는 두 개의 history-free 연산자족 — Q-coverage와 Layer-Adaptive K+Q — 을 제시하고, MetaTool Subtask4와 $\tau^2$-bench retail에서 Qwen과 Llama에 대해 동일 프로토콜로 SEKA/AdaSEKA를 이긴다.

## 디렉터리 구조

```
paper/neurips2026_steering_ko/
├── main.tex, content.tex, refs.bib, neurips_2026.sty
├── main.pdf                     ← 빌드 산출물 (13 페이지)
├── README.md                    ← 이 파일
├── MANIFEST.md                  ← 모든 표/그림의 결과 JSON 및 스크립트 매핑
├── FACT_BASE.md                 ← locked 수치와 JSON 스키마
├── sections/
│   ├── 01_abstract.tex
│   ├── 02_introduction.tex
│   ├── 03_related_work.tex
│   ├── 04_method.tex            ← Definition 3.1-3.4, Assumption A1-A6
│   ├── 05_theory.tex            ← Theorem 4.1-4.4, Proposition 4.5, Corollary 4.6
│   ├── 06_experiments.tex       ← E1-E6 (SEKA 비교 축)
│   ├── 07_discussion.tex        ← Regime-split, limitations
│   ├── 08_conclusion.tex
│   └── 09_appendices.tex        ← 증명 + SEKA 재현 + 매니페스트 + 코드
├── figures/
│   ├── fig1_concept.pdf         ← 세 연산자 개념도
│   ├── fig2_delta_vs_k.pdf      ← Regime-split (placeholder)
│   ├── fig3_stepwise.pdf        ← Stepwise coverage (placeholder)
│   ├── fig4_basis.pdf           ← Basis ablation
│   ├── fig5_size_sweep.pdf      ← Qwen 크기 스윕 (placeholder)
│   └── fig_{main,qbias,stability}_*.pdf   ← 레거시
└── scripts/
    └── build_placeholder_figures.py
```

## 관련 실험 코드 및 데이터 위치

| 역할 | 파일 |
|---|---|
| 정지 K (SEKA-style) | `scripts/ocq/eval_metatool_subtask1.py::install_kbias_hooks` |
| Q-coverage (ours) | `scripts/ocq/eval_metatool_subtask1.py::install_q_bias_hooks` |
| Layer-Adaptive K+Q (ours) | `scripts/ocq/eval_metatool_subtask1.py::install_layer_adaptive_hooks` |
| Subtask4 평가 | `scripts/ocq/eval_metatool_subtask4.py` (stepwise 블록) |
| $\tau^2$-bench 평가 | `scripts/ocq/eval_tau2_bench.py` |
| Canonical SEKA on Subtask4 | `scripts/ocq/eval_subtask4_with_real_seka.py` |
| Canonical AdaSEKA | `scripts/diagnostics_2026_04_16/eval_subtask4_with_adaseka.py` |
| PCA basis builder (E4) | `scripts/ocq/build_pca_baseline_basis.py` |
| 크기 스윕 드라이버 (E5) | `scripts/ocq/run_tau2_size_sweep.sh` |

결과 JSON은 `reports/` 하위에 저장되며, `MANIFEST.md`가 (표/그림) → (JSON 경로 + 실행 스크립트)를 일대일 매핑한다.

## 실험 계획

`reports/steering_paper/EXPERIMENT_PLAN_UNIFIED_2026_04_16_v2.md`. 6개 주 실험(E1--E6)의 의도/가설/검증을 모두 기록.

## 빌드

```bash
python scripts/build_placeholder_figures.py
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```
