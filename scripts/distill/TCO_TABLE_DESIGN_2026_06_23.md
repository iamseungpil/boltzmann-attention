# TCO 비용 표 — 북극성 산출물(#6) 골격 + 첫 실제 셀 (2026-06-23)

> 진입: `EXPERIMENT_DESIGN §0★★`(목적함수·평가#6) · `CAPABILITY_LEVER §0''–§3d`(생애주기 비용축) · 리뷰(2026-06-23).
> ★리뷰 메타-지적(반영): 우리는 "비용-효율"을 표방했으나 **efficiency의 분모(조립된 비용: $·VRAM등급·레이턴시·인간시간)를 한 번도 만들지 않았다**. pass^1·false-block·P/R·A2필드수 = 전부 capability/메커니즘 또는 비용의 *한 항*이지 조립된 비용 아님. 이 표(#6)가 "비용-효율 논문" vs "소형 tool-use 재탕"을 가르는 핵심 산출물. **이게 북극성 deliverable.**

## 0. 기준선 = (c) 3열 (리뷰 권장·채택)
한 표에 **인간노동 / frontier-API / 소형-on-prem**을 나란히 → "인간보다 싸고, frontier-API보다 싸고(=Palantir류 대비), 동등 compliance" = 논문 헤드라인. (a)시스템간만/(b)인간만이 아니라 셋 다.

## 1. 비용 행 (실측 vs 추정 명시)
| 행 | 출처 | 상태 |
|---|---|---|
| compliance pass^1 | 측정 | 실측(g15 32B 0.573 등) |
| **latency /req** | sim `duration` | **실측** |
| **tool-roundtrips /req** | 궤적 tool_calls | **실측(~8)** |
| VRAM / GPU등급 | 모델 known | 실측 |
| **$ /req** | 토큰×단가(API)·GPU상각/처리량(on-prem)·시급(인간) | **추정**(token 미포착→가정·정밀화 §4) |
| ⑤ 일반화비용/새도메인 | role-sourcing 실험 | 미측정(다음) |

## 2. ★첫 실제 셀 (retail·기존 런·새 GPU 0)
| | 인간 상담원 | gpt-4.1 (frontier-API) | **32B-int8+g15 (on-prem)** |
|---|---|---|---|
| retail compliance pass^1 | ~1.0(가정·숙련) | 0.82(leaderboard·bench) | **0.573**(compliant=bench·위반0) |
| latency /req | 분~십분 | **30.5s**(실측) | **178s**(실측·~6× 느림) |
| tool-roundtrips | – | 7.6 | 8.1 |
| VRAM/HW | – | API(on-prem 불가) | **~33GB·1×A6000** |
| **$ /req** (computed·`tco_cost.py`) | ~$5–7(콜센터·문헌가정) | **$0.044**(18.3k in/1.0k out × $2/$8 per 1M) | **$0.0019**(A6000 $0.3/hr ÷ 162 req/hr@conc8) |
| 데이터반출 | – | ❌(API 전송) | ✅ on-prem |
| 감사가능 | 부분 | ❌ black-box | ✅(결정론 게이트·궤적) |

(7B on-prem $0.0008/74s·14B $0.0012/119s — 더 싸고 빠르나 compliance↓. 전 행 `tco_cost.py` 실측 latency + 추정 token/\$.)

**★computed 헤드라인(assumption robust)**: 32B-on-prem **$0.0019/req = gpt-4.1 API $0.044보다 ~24× 쌈**(GPU $0.2–0.5/hr 범위서 16–40×)·**인간 ~$6보다 ~3000× 쌈** — on-prem·감사가능·결정론게이트. 트레이드오프 = 레이턴시 ~6×(178s vs 30s). ★단 **compliance 0.573 < 0.82 = "동등" 아님**(아래 fleet).

### ★fleet 투영 = "동등 compliance를 더 싸게" (정직한 cost-efficiency 셀·`CAPABILITY_LEVER §10`)
- 순수 32B는 0.573(< 0.82)이라 "동등 더 쌈" 주장 *불가*. **fleet(쉬운req=32B·어려운req=frontier escalate)**로 blended:
  - blended $ ≈ p·$0.0019 + (1−p)·$0.044 (p=32B가 처리하는 비율). p=0.57이면 ≈ **$0.020/req**·blended compliance≈0.82(32B 성공분 + frontier가 나머지). vs 순수 frontier $0.044@0.82 = **~2.2× 쌈·동등 compliance**.
- ⇒ **두 헤드라인 셀**: (A) "동등 compliance(fleet)를 ~2× 싸게" (B) "근접 compliance(0.573)를 순수-on-prem ~24× 싸게". 단 fleet는 *cheap·decidable 라우터* 전제(§10 크로스오버·미구현=다음 instrument).

## 3. ★헤드라인 재서술 (리뷰#2 반영)
- ❌ 옛: "32B가 0.82 맞췄다/맞추는 중"(=ToolOrchestra 패리티 재탕·rival 영역).
- ✅ 새: **"동등 compliance를 frontier-API 대비 ~X배·인간 대비 ~Y배 싼 req당 TCO로(on-prem·감사가능·결정론 게이트)" + "능력별 최소비용 레버 배정 가이드라인".** pass 0.573/0.82 = 본문 증거 한 셀(레이턴시·정확도 트레이드오프 곡선의 한 점).

## 4. 정밀화 (추정→실측 fix·다음 런 적용)
- **$ /req 정밀화 = token/cost 포착**: `t2_run_gated`가 agent_cost/user_cost를 0으로 남김(litellm cost tracking off). → litellm `completion_cost`/usage 캡처 ON(또는 message content로 token 근사) → API$·on-prem 토큰처리량 실측. (소규모 재계산·gpt-4.1 $는 retail_gpt41_nogate 재집계로.)
- **on-prem $ 정밀화**: GPU 상각(자본/수명/전력/가동률) 명시 모델 + 실 throughput(run wall-clock/tasks×concurrency). 가동률 가정이 $/req 지배 → 범위로 보고.
- **인간 baseline**: 콜센터 contact 비용 문헌값(지역별 $3–8) 인용·가정 명시.
- **레이턴시**: on-prem 178s = int8+enforce-eager+conc8 → 실배포는 양자화·배칭 최적화로 단축 가능(별도 측정).

## 5. 시퀀싱 (리뷰#5·#1 반영)
1. **이 TCO 표(①②③④열) 먼저** = 현 런에 비용열 붙이기(새 GPU 0). nt=3 denoise 회수 시 pass 옆에 latency·$·VRAM 즉시 부착.
2. 헤드라인 재서술(§3)로 격상.
3. **그 다음 role-sourcing(1안/2안)** = ⑤일반화비용 열 채우기(새 field A2-swap 비용). 헤드라인 아님·TCO 표 ⑤열 instrument.
4. token capture fix → $ 행 추정→실측.

## 6. 정직 caveat
- $ 행 = **추정**(token 미포착·GPU 가동률 가정). latency/roundtrips/VRAM = 실측. → order-of-magnitude는 강건하나 정확 배수는 §4 정밀화 후.
- 32B 0.573 < gpt-4.1 0.82 = compliance 격차 존재 → "동등"이 아니라 "근접+훨씬 쌈"이 정직. 격차의 capability 잔여(§1b ~25% addressable 외)는 정직 보고. (또는 fleet: 쉬운 req=32B·어려운 req=escalate로 blended compliance↑·blended $↓ = `CAPABILITY_LEVER §10` 크로스오버를 TCO로.)
- 레이턴시 6×는 실 SLA 제약 가능 → 트레이드오프 명시(저비용 vs 저지연).
