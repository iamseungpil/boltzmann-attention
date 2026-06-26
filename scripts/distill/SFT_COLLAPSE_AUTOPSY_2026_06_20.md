# SFT 붕괴 원인 정밀 확정 (§3.5) — 전수 full-message 실독 결과 2026-06-20

> 선결 과제(`HANDOFF_2026_06_20 §3.5`): "SFT가 의미없다"가 진짜 catastrophic forgetting인지, format/harness mismatch인지, lr/rank confound인지 **메커니즘 확정**. 사용자 의심 정당. 도구=`results.json` raw_data 전수 실독(mtx_*_{tbox,abox} 5어댑터).

## ★판정 (한 줄)
**(b) parser/format mismatch = 기각. (a) 학습-분포-이동(interference) = 확정·단 *legible*(랜덤 망각 아님). (c) lr/rank = 비주도(severity만 변조).** ⇒ "SFT 무의미"는 **거짓**: 어댑터들은 *학습한 대로 충실히 emit*하나 그게 retail ABI와 안 맞음 = **실험 mismatch를 메커니즘 수준서 확증**(narrow 단일-추상-도구 전문가를 14-도구 full 에이전트로 평가).

## 증거 (raw_data 전수)
tool_call은 **transport/parse 정상**(raw `choices[].message.tool_calls`에 name+arguments JSON 깨끗·hermes 파서 OK). 실패는 전부 *tool 실행 의미* 단계. ⇒ (b) 기각의 직접 근거.

### fact_full / fact_prov (r32·α64) — 추상 arg-스킴 emit
- `get_product_details(product_id={"$ref": "0#.T-Shirt"})` = **학습한 심볼릭 provenance 포인터**(이전 출력 참조)를 concrete id 대신 emit → retail은 str 기대 → `unhashable type: 'dict'`·`'dict' object has no attribute 'lower'`.
- 더 지배적: `get_product_details()` 등을 **빈 arg `{}`**로 호출 → `missing N required positional argument`(fact_full tbox 261·fact_prov tbox 310). $ref은 일부(37~49/~470).
- **ABox(resolve1+grounding)도 못 고침**(pass 1/50 동일·$ref 미-dereference·missing-arg 잔존). ⇒ 단순 "dereference 스캐폴드 누락"만이 아니라 모델이 concrete retail-arg **populate 능력을 추상패턴으로 덮어씀**.
- term: too_many_errors 지배(에러루프).

### solo_sts (r64·α128) — 값 날조
- tool_call **well-formed**(빈 arg 아님): `find_user_id_by_email(email="example@example.com")`·`get_product_details(product_id="6086499569")`·`find_user_id_by_name_zip(John,Doe,12345)` = **placeholder/날조 값** → not-found(108~244). 일부 spurious arg(`list_all_product_types(title=..)`).
- 학습=`resolve_selection` 단일 도구 **3001/3001**. 14개 실 retail 도구는 *한 번도 안 봄* → well-formed 호출은 base Qwen-Instruct 능력 누출·날조 값은 gather 미학습 탓.

### solo_cfb_mid (r16·α32) — 최소 손상
- well-formed 호출·arg-shape 에러 0·실 대화("주문 ID 알려주세요")·우아한 human transfer. 실패=날조→not-found(91~175). pass 6/50(7B SFT 중 최고). = grounding 실패지 구조 붕괴 아님.

## (c) 기각 근거
rank: solo_sts(r64) > fact_*(r32) > cfb_mid(r16). 만약 rank가 원인이면 r64가 최악·r16 최선이어야 하나 — **r64 solo_sts는 빈-arg 병리 전혀 없음(well-formed)·r16 cfb_mid가 최소손상**. 질적 병리(빈-arg vs 날조)는 rank 아닌 **데이터 타깃**이 결정. rank=severity만 변조.

## base_tbox 무효
`mtx_base_tbox`·`mtx_base_abox` = 50/50 `infrastructure_error`(serve 실패). 매트릭스 base 셀 **재실행 필요**(비교 기준 공백).

## 함의 (ReST·§4 게이트 통과)
1. **(b) 기각 → ReST 정당화 유효**(SFT가 살아있어 방향 재고할 필요 없음). 단 "catastrophic forgetting"이라는 *단정*은 부정확 — 정확히는 **narrow 단일-추상-도구 SFT가 arg-분포를 추상스킴으로 이동**. 평가 자체가 OOD(mismatch).
2. **ReST 데이터 설계 교정**: 반드시 **실 retail 14-도구·concrete arg 전체 궤적**으로 학습(추상 단일도구 금지). 검증기=DB-match + **concrete-arg populate 확인**(빈-arg/$ref emit 0).
3. **$ref provenance는 thesis-정합**(날조 대신 참조) — 단 *scaffold dereference 스텝* 있어야 유용. 현재 TBox·ABox 둘 다 미-dereference. 설계옵션: provenance를 원하면 resolver가 $ref 해소(decidable→offload).
4. 평가 규율: **narrow 선택 전문가를 full-agent로 돌리지 말 것**(mismatch 재발 방지·`03-anti-drift`).

## 도구·재현
`/tmp` 분석 스니펫(classify·exc_detail·raw_dump·abox_ref·cfg). 입력=`/home/woori/scratch/tau2-bench/data/simulations/mtx_{fact_full,fact_prov,solo_sts,solo_cfb_mid}_{tbox,abox}/results.json`. 학습데이터=`/home/woori/scratch/fc_build/sft_solo_*.jsonl`·어댑터=`scratch/{sft_runs/fact_*,adapters/qwen7b_solo_*}`.
