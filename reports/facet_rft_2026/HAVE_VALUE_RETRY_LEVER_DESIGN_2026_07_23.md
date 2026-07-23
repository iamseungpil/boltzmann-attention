# "have-value → act" 일반 레버 설계 — 2026-07-23

> 이번 세션 반복 발견의 통합: **시각-루프·052·last-4·050 flail = 한 가족** =
> *에이전트가 write의 필수 인자를 못 채워/이미 값이 있는데도 재확인·재요청을 반복하고 write를 재시도 안 함.*
> 개별 fix(A2 met_template·CLI verdict-gate·dedup redirect)로 하나씩 잡았으나, 근저의 **단일 일반 레버**를 정식화.
> 격리 확정(bank_039_last4_iso·idx55): "값 이미 있다·지금 file하라" 지시 → file_dispute **8/9** vs 무지시 재요청 8/9.

## 0. 문제 (4 인스턴스·전부 격리 확정)
| 인스턴스 | write | 필수 인자 | 루프 형태 |
|---|---|---|---|
| 시각(054) | log_verification | time_verified | get_current_time 미호출→고객에 시각 물음 반복 |
| CLI(052) | approve/deny | (판정) | 쿨다운 직접 오판·판정도구 우회 |
| last-4(039) | file_dispute | card_last_4_digits | 값(1652) 손에 쥐고도 "last-4 필요" 반복·재-file 안 함 |
| flail(050) | (검색) | (도구명) | 함수명 BM25 검색 반복 |

공통 = **필수 입력 미충족 → 텍스트/도구 반복(두번째 say) → temp0 고착 → write 미도달**.

## 1. 일반 레버 = provenance-satisfied retry nudge
**정의**: write W가 인자 A를 요구(A2 WEV/WRITE_ARG_GROUND 선언)하고, ①A가 *이전엔 미충족*(deny 발화 or 에이전트
재요청)이었다가 ②*지금 대화에 실재*(grounding 검증기 통과=tool 출력/user 발화에 값 존재) ③그런데 에이전트가
A를 **또 재요청**(그 write 미시도) → **넛지 주입**: *"A는 이미 대화에 있다(값 실재). 다시 묻지 말고 W를 그 값으로
지금 호출하라."* 에이전트가 W를 **직접 emit**(강제 0).

## 2. [[05]] 3질문 (필수·[[17]])
1. **도메인-특화 순증?** — **No(엔진)**. 엔진=WEV specs(이미 A2)의 grounded_args를 재사용해 "요구 인자 + 이제
   실재 + 재요청" 판정. 도구/인자명=A2(기존). 리터럴0. (지시문에 값 "1652"는 대화서 추출·정적 A2 아님.)
2. **유동판단 동결?** — **No.** 어느 값을 쓸지는 여전히 에이전트가 판단(대화의 실재값)·엔진은 "이미 있으니
   재요청 말고 써라"만. 값 선택 안 함.
3. **스캐폴드가 write 수행?** — **No.** 넛지만·W는 에이전트가 emit. write강제 금지(§1.5) 준수.

## 3. 기존 fix와의 관계 (통합)
- 시각 met_template hint·CLI verdict-gate·dedup redirect = **이 레버의 인스턴스별 조기 배선**(각각 met_template/
  WEV/dedup 채널). 이 일반 레버 = **provenance 채널서 통일**(값 실재 신호=이미 있는 grounding 검증기).
- ⇒ 개별 A2 hint 증식 대신 **엔진 1개소**(provenance retry)로 수렴 = minimize-A2·[[13]] 흡수우선.

## 4. 트리거 정밀화 (오발·derail 방지)
- **조건 AND**: (a) 그 write의 grounded arg가 **직전 deny/재요청 이력** 있음 (b) 값이 **지금 grounding 통과**
  (c) 직전 에이전트 턴이 **그 arg 재요청/그 write 미시도**. → 값 없을 때 발화 0(오발 방지).
- **cap**: 1회(넛지 후에도 미시도면 통과·강제 없음). 리마인더=생성-레벨 비커밋(대화 오염 0·[[73]]).

## 5. 위험·계측
- **Δspurious**: 넛지는 "이미 있는 값으로 재시도"만 유도 → 새 write 안 만듦(상한=요구된 write). 계측.
- **오발**: 값 미실재면 미발화(grounding 검증기가 게이트). last-4가 실은 틀린 값이면 grounding이 이미 차단.

## 6. 격리 검증 상태·다음
- ✅ **결정점 확정**(last-4·idx55): 값-실재+지시 → file 8/9 vs 무지시 재요청 8/9. 원리 인과 확정.
- 다음(구현): provenance 검증기(t2_gate_patch _write_arg_ground_deny/_first_fab_call)에 "이전-미충족 ∧ 지금-실재
  ∧ 재요청" 감지 + 생성-레벨 넛지. 오프라인 replay(시각 E-arm·last-4 idx55 동형)로 shipped 문구 검증 후 라이브.
- 한계: 값이 **정말 없는** 경우(획득 경로 자체가 막힘·user-tool 핸드오프 실패)는 이 레버 밖 = 별도(경로-표면화).
