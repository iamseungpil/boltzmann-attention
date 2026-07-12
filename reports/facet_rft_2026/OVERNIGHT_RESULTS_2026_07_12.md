# 밤샘 실험 결과 종합 (2026-07-11 밤 → 07-12 아침·전부 무료·user-sim 0)

> [[09]] 준수: gpt-4.1 user-sim 0(격리 프로브·로컬 32B/사다리)·비용 0. C단계(유료)는 주간·승인.

## ★결과 1 — E-REF scale 사다리 (논문 핵심 figure·6모델·무료)
> 재현: `eref_scale_ladder.sh` · `sim_results/eref_ladder_{tag}_{clean,loadC,distA}.jsonl` · 짝지은 36 시나리오/셀.

| 모델 | clean(P1/P2 bind) | 부하C | distractorA | parse |
|---|---|---|---|---|
| 0.5B | **0.04** | 0.01 | 0.02 | 0.92 |
| **1.5B** | **0.89** | 0.96 | 0.90 | 1.00 |
| 3B | 0.99 | 0.97 | 1.00 | 1.00 |
| 7B | 0.99 | 1.00 | 1.00 | 1.00 |
| 14B | 0.97 | 1.00 | 1.00 | 1.00 |
| 32B | 1.00 | 1.00 | 1.00 | 1.00 |

**[M] 두 발견**:
1. **참조-바인딩은 ~1.5B에서 급격 emergent** (0.5B 0.04→1.5B 0.89→3B+ 포화). parse 0.92라 파싱문제 아님 = 0.5B는 진짜 못 함. ⇒ **tool-use의 추상 능력(deictic binding)은 초소형(1.5–3B)에서 이미 산다** = [[00-thesis]] 직접 증거.
2. **정적 오염(부하 20k·distractor 10)은 어느 scale서도 바인딩 안 부숨**(능력 있으면 1.00 유지). ⇒ **범인은 정적소음 아님** = "lost-in-the-middle" 류 기각·V2 재확인.
- ⇒ **명제**: clean 바인딩은 싸고(1.5B) 정적소음에 강건 → **in-vivo 붕괴는 *동적* 오염 몫**(자기-프레임·멀티턴·패러프레이즈·⋈). 그 fexec 프록시 = 0.79(§결과3).
- **정직 caveat**: clean 합성 과제. 동적-공격 축(near-miss/paraphrase) 미실행(밤샘 GPU0 배치 스펙 미집행) → 재발사 필요.

## ★결과 2 — 4지선다 출처선언 처방-비교 (base/prompt/loop·n=60·전부 producer存)
> 재현: `c51_fourway_prescriptions.jsonl`. 결정점마다 {GET producer호출·FIND문맥·INFER·ASK} 선언 + 갈래별.

| arm | GET | ASK | FIND정답 | FIND오답 | **옳은-출처선언(GET∪FIND정답)** |
|---|---|---|---|---|---|
| base | 1 | 21 | 22 | 8 | **0.38** |
| prompt | 30 | 11 | 15 | 2 | **0.75** |
| loop | 23 | **0** | 23 | 10 | **0.77** |

**★per-case 검증(포렌식 가드·[[08]])**: `find_exact` payload==gold **60/60**·`find_wrong`≠gold **20/20** (라벨 정확). ⚠️**단 지표=*출처선언 정확도*이지 최종값 아님** — GET 54건은 payload=도구명(get_*_details)이라 실제 조회값 미검(producer 존재 시 옳은-출처로 계상). gold_val=None 7/60(순수 pick·채점 애매). 최종값 검증분은 find_exact(직접확인)뿐.

**발견**:
- **base = ASK 남발**(21/60·t17형 시스템적)·GET 1 = 소형모델 default가 "물어보기". 옳은-출처 0.38.
- **prompt = 대폭 개선**(0.38→0.75·GET 1→30·FIND오답 8→2). ★단 이건 **격리 단일턴**(C42 regime=짧은문맥서 prompt 작동)이라 [[42]]와 모순 아님 — **in-vivo(멀티턴) 유지 여부는 미검**(C-stage가 판정).
- **loop = ASK 완전제거(0)**·옳은-출처 0.77(prompt와 유사)·**단 FIND오답 10↑**(강제가 ASK 피하되 문맥 오값 FIND로 샘=⋈ 누출). forced_GET 3회.
- ⇒ **1차 판정**: 격리서 **prompt≈loop 둘 다 base 2배**. loop의 강점=ASK 0(문맥길이 불변·[[45]] load-invariant)이라 **in-vivo서 prompt 붕괴 시 loop이 이길 것**(C-stage 예측). loop 약점=find_wrong(⋈ 경계 누출).
- **learn arm**: 미실행(설계만·C42 데이터게이트) — base가 이미 prompt로 0.75면 격리서 학습여지 작음·**in-vivo 오염 데이터로 필요성 先실증** 필요.

## ★결과 3 — fexec 형식화 실행-채점 (버그 폐기·재측정)
> `fexec_exec_probe.py`(실행-기반·hand-gold 0). 상세 = §0a of `E_REF_BOUNDARY_DESIGN`.

| set | exec-correct |
|---|---|
| target(t20·t37·t79) | **0.88** (t20 4/4·t37 3/3·t79 0) |
| 전체 변형-선택 87건 | **0.79** |

- 기존 "EM 0.00/12%"는 **채점버그**(태스크당 단일 hand-gold·멀티품목 무시) → 폐기(3문서 반영). 형식화는 실제 gold를 79~88% 낸다.
- 잔여 = 제약형 0.73·t79(딴 병 색 오바인딩=⋈). ⇒ **FORMALIZE-EXEC "미편입" 판정 붕괴** → 재편입 대상(B-max②).

## ★결과 4 — 동적 바인딩-공격 축 (GPU0 배치 완료·2026-07-12·[M])
> `eref_gpu0_{nearmissB,paraphraseP,fexec_all}.jsonl`(리모트 persist `539b36d`). 32B GPTQ·infra 0(parse_fail=0·exec_fail=0). 판정도구 = scratchpad `eref_agg.py`/`eref_case.sh`.

**축 B (near-miss same-dim distractor·동적오염)** — bind = 옳은 값이 제약에 바인딩된 비율:

| level (오염밀도) | n | bind | op | cons |
|---|---|---|---|---|
| 0 (clean) | 36 | **1.000** | 1.000 | 1.000 |
| 1 | 36 | 0.722 | 1.000 | 0.722 |
| 2 | 36 | 0.750 | 1.000 | 0.750 |
| 4 | 36 | **0.472** | 1.000 | 0.472 |

**축 P (paraphrase)**: lv0 bind 1.00·em 1.00 → lv1 **bind 1.00·em/cons 0.75**(바인딩 생존·제약-exactness만 저하 = B보다 약함).

- **★per-case 정독([[08]]·B lv4 19/36 fail 전건 정독)**: 실패 = 전부 **anchor(near-miss) 값이 gold 대신 바인딩**(Backpack "large"→"small"·Helmet "M"→"S"·Kettle "2L"→"1.5L"). 랜덤오류 아닌 **정박치환**(C43 동형). `op`=1.00 불변 = 연산구조 온전·**제약 값만 오염원에 포획**.
- **판정 [M]**: 정적오염(결과1·부하/distractor)=전 scale 1.00 강건인데, **동적오염(near-miss B)이 바인딩을 1.00→0.47로 부순다**(오염밀도 단조·infra 오염 0). paraphrase는 더 약함(바인딩 생존). ⇒ **"동적오염이 바인딩 부순다" 실증** = E-REF 완성·[[00-thesis]]·2509.09677(self-conditioning scale-불변)과 수렴.
- **fexec_all(형식화 실행-채점 전체 87건)**: exec_correct_avail **0.770**(밤샘 0.79 재확인)·제약형 0.74 vs 무제약 0.93 = 바인딩이 손실 지점(축 B와 정합).

## ✅ 갭 닫힘 (2026-07-12)
1. ✅ **동적 바인딩-공격 축** = 결과4(위)·[M] 확정.
2. ✅ **딥리서치 synth**(scale=오염내성 선행) = 수동종합 완료 → 정본 `SCALE_DYNAMIC_CONTAMINATION_PRIORWORK_2026_07_12.md`. 판정: scale=horizon 구매([지지]·2509.09677=F6 수렴)·강한형 균일면역=반박·whitespace 미선점(2509.09677 인접·구분). 축 b/c/d 소스=[미검](verdict 미실행·인용 전 검증필요).

## 종합 — 오늘 명제 상태
> **tool-use = 저-추상·고-간섭**: 추상능력(바인딩) 1.5B서 emergent·정적소음 강건(결과1)·**단 동적오염이 정박치환으로 바인딩 부숨(결과4·[M])** → 격차는 *동적* 간섭. 규율(출처선언)은 prompt/loop로 격리서 2배 개선(결과2)·형식화도 실제 작동(결과3). 선행지형(synth)이 3중 수렴(2509.09677 self-cond scale-불변·Laban 멀티턴 보편붕괴·C43 정박치환). **남은 [M]화**: in-vivo(C-stage) prompt-vs-loop 붕괴곡선(유료·승인).
