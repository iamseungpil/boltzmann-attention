# 신스택 이득/부작용 per-step 포렌식 (직접실행·2026-07-11 밤)

> 사용자 지시(정체된 에이전트 대체·직접 수행). [[08]] 마커-귀속 + Δspurious + per-case 정독.
> 데이터: `t5c_b78c2.results.json.gz`(신스택 78 task nt=1·라이브) + `.stderr.log.gz`(마커) vs `comp_retail_t4`(구스택 nt=4).
> 재현: `newstack_perstep_census.py`. **방법 규율**: 개입=생성-레벨(silent)→마커는 stderr(sim-id 없음)→(tool,val) 값-매칭 귀속.
> **★nt 불일치(신 nt=1 vs COMP nt=4) → sim-비율 직접비교 무효**. 태스크-단위 + Δspurious(nt-무관) 중심.

## §0 결론 (한 줄)
**신스택은 부작용을 순증시키지 않았다**: Δspurious **음성**(3.8% < COMP 5.7%)·유일 레버-유발 부작용 = E-PLAN deny-루프(cap 19회로 완화). 레버 발화-sim은 baseline 위(약한 양성·귀속 노이즈).

## §A 마커 census — 전 레버 라이브 발화 확인
| 레버 | 발화(b78c2·78 sim) |
|---|---|
| DISAMB (subcall silent) | 111 |
| E-PLAN L2 (미검토 sibling read-강제) | 93 |
| GROUND (\|C\|=1 치환) | 78 |
| E-PLAN walk (종결 리마인더) | 39 |
| PROV_regen (per-arg) | 22 |
| **E-PLAN cap (무한루프 방지 발동)** | **19** |
| PROV rescue·E-PLAN L1 | 3·3 |
- 신규 레버(PERARG·E-PLAN·conflation) 전부 라이브 발화. NLNUM·CALCX·P2 마커 0(off 또는 미발화).

## §B 이득 개연 — 레버 발화-sim ∩ 통과 (귀속·nt=1)
| 레버 | 귀속 sim | 통과 | 통과율 |
|---|---|---|---|
| DISAMB | 71 | 39 | **55%** |
| E-PLAN L2 | 45 | 25 | **56%** |
| PROV_regen | 3 | 1 | 33% |
- 전체 b78c2 통과율 = **52.6%** → DISAMB/L2 발화-sim이 **약간 위**(55~56%) = 약한 양성 신호.
- **주의(정직)**: val-매칭 귀속은 다른 sim에 같은 order_id면 과대계상 → 상한. "통과=이득 개연"이지 인과 아님. 인과는 짝-대조(v25e nt=4·별도) 필요.

## §C Δspurious — 부작용 핵심 지표 (nt-무관·깨끗)
| | OVER_ACTION(gold-없는 write 실행) |
|---|---|
| **신스택 b78c2** | **3/78 (3.8%)** |
| 구스택 COMP | 26/456 (5.7%) |
- **신스택이 더 낮다 → 레버가 안 시킨 write를 순증 안 시킴**(Δspurious ≤ 0 실측·GO 조건 충족).
- **per-case 정독 3/3 = 전부 기존 B-class**(레버-유발 아님):
  - **t34**: 조건체인("부분취소 안 되면…") 오해석 → 주문 취소(gold=주소변경). C25 계열.
  - **t57**: 조건체인 끝=무행동인데 취소 + **"gift card 환불" 허위 발화**(C63/기존 t57).
  - **t99**: 사용자 *"스케이트보드는 내가 직접 취소"* 명시 → 에이전트가 수행(철회-요청 수행). C25.
  - ⇒ 셋 다 **대화-semantic 잔여**(C50·scaffold 밖)·신규 레버와 무관.

## §D 유일 레버-유발 부작용 = E-PLAN deny-루프 (완화됨)
- E-PLAN L2 deny가 무한 반복→max_steps 소진(t27/t103·`T27_T103_PERSTEP` 정본). **cap(19회 발동)이 방지장치로 실작동**.
- L2-deny 귀속 45 sim 중 실패 20 — 단 대부분 **기존 실패**(t20/t57/t102/t109 등=B/능력 잔여)이고 deny-루프 순유발은 t27/t103류 소수(cap이 흡수). 귀속 노이즈로 상한.
- **conflation 수리(과발화 −45%·커밋됨)가 이 부작용의 근원(품목↔주문 수량 혼동)을 줄임** — 단 b78c2는 **수리 전** 스택이라 이 census엔 미반영. 수리-후 재측정은 C단계.

## §E 대차대조 (나아진 것 vs 부작용)
| 축 | 판정 | 근거 |
|---|---|---|
| **이득** | DISAMB·E-PLAN L2 발화-sim 통과율 baseline 위(+2~3pp) | §B (약한 양성·귀속 상한) |
| **Δspurious** | **음성**(안 시킨 write 순증 0·3건은 기존 B-class) | §C per-case 3/3 |
| **레버-유발 부작용** | E-PLAN deny-루프 1종 (cap 완화·conflation 수리로 근원 축소) | §D·`T27_T103` |
| **미측정(정직)** | 인과·robust는 nt=1이라 미확정 | v25e 짝대조·C단계 pass^k 필요 |

## §F 남은 것
1. **인과 확정**: v25e(nt=4·COMP 짝) trial-flip 또는 C단계(456 사이클) — nt=1 신호를 pass^k로 경화.
2. **수리-후 재측정**: b78c2는 conflation 수리 전 → 수리-후 스택(현재 working tree)으로 재런 시 deny-루부작용 추가 감소 기대.
3. per-case 정독 확장: DISAMB 55%·L2 56%의 "귀속 통과"가 진짜 이득인지 실패-표적서 fail→pass 짝 정독(C단계 per-case).
