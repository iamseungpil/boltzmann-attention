# HANDOFF — 2026-08-21 저녁 · 다음 세션이 이어받을 것

> 등대 = `ba-frft/reports/facet_rft_2026/RESEARCH_MASTER.md`
> 앞 문서 = `HANDOFF_2026_08_21_PM.md` → 이 문서. 원장 = **C585~C586**(2건)
> 이 세션은 **전부 무료**(로컬 vLLM 8141·user-sim 0·유료 런 0).

---

## §0 지금 도는 것 — **없다**

```
GPU 0/1  유휴 · vLLM 8140/8141 상주
유료 런  없음 · 리모트 미커밋 = 종전 3건 그대로(이 세션 것 아님)
로컬 미커밋 = test_action_reminder.py(종전 것) + RESEARCH_MASTER/핸드오프(이 문서 커밋에 포함)
```

⚠**세션 시작 함정 하나 잡았다**: PM 핸드오프의 A3 커밋 `58c26b84` 이 **로컬 미러에 없었다**
(리모트만 push 됨·로컬 HEAD 는 앵커 없는 옛 판). fetch·ff 로 맞췄다. **핸드오프가 가리키는
커밋은 세션 시작 때 로컬 존재부터 검산할 것.**

---

## §1 PM 핸드오프 §1 순서의 소화 상태

```
2 엔진 전달 교체        ✅ 완료(커밋 a4120fa5 · de52d31d) — 아래 §2
3 x456 재측정           ✅ 완료 — A 45% ↔ C_docs(v2) 87.5% (C585·한계 셋 필독)
4 read gold 축 감사     ✅ 완료(x461·C586) — 원인 = 선언 위상
4b requires_reads 저작  ✅ 완료(커밋 a0b8ddea·C587) — 4 write·행마다 정책 축자·아래 §5
5 감사 표-행 계약       ⏳ 미착수(사실표용·배달과 무관)
6 24 태스크 A/B         ⏳ 유료·승인 필요. 격리 근거(C585) + 선언 보강(C587) 둘 다 실렸다
```

### 다음 세션 첫 후보 = **⑥ 승인 요청** (또는 ⑤)

- **⑥**: `T2_SG_DOCS=1` 를 라이브 스택에 넣은 24 태스크(1단계 20+4) A/B — go_stack 에 플래그
  추가부터(아직 안 넣었다·격리 갈림 확인 전 라이브 배선 금지 원칙 지킴). 판정선 Δ ≥ 4/40(E-MFIX).
  C587 의 새 선언은 라이브 코드가 그대로 읽으므로(요건 큐→핀) 별도 플래그 불요 — A/B 는
  T2_SG_DOCS 하나로 가르되, **requires_reads 는 양팔 공통**(선언 저작은 처방이지 실험 변수가 아님).

### §5 (4b) requires_reads 저작 — C587

```
행 4 (relations.declarations + edges + by_tool 동기·두 층 동일)
  apply_checking_account_credit        ← get_bank_account_transactions
  apply_savings_account_credit         ← get_all_user_accounts + get_bank_account_transactions
  submit_interest_discrepancy_report   ← get_all_user_accounts + get_bank_account_transactions
  file_debit_card_transaction_dispute  ← get_bank_account_transactions
출처(행마다 note 에 축자): doc_017 · doc_043 · doc_044 · doc_031 (KB documents 직독·
  ⛔tasks/gold 미개봉 — 도구명이 tasks 에서만 잡혔을 때 documents/db 로 우회)
검산: 검정 5종 + 핀 자기검정 10/10 ALL PASS · 피의존 1→5 ·
  4 write 전부 first_step/frontier = get_bank_account_transactions 유일 해소
한계: 오프라인까지. 라이브에서 큐가 이 write 를 표적 삼는 턴이 오는지는 ⑥이 판정.
050 은 대상 아님(이미 선언돼 있고 실패 원인이 선언 부재가 아님).
```

---

## §2 이 세션이 만든 것 (원장 C585~C586)

### C585 — 엔진 전달 교체 + x456 재측정

```
배선(a4120fa5): T2_SG_DOCS=1 ∧ isolate.docs → 서브에게 검색 안 시킴
  · 클래스 선택 = 별도 서브·by_class 38 닫힌 목록·엔진은 소속 검산만([[65]][[22]])
  · 엔진 = always 전량 + 선택 클래스 content-범위 자르기 + 앵커 40자 검산
    (불일치 = 문서 전량 폴백 + 로그·실측 앵커폴백 0)
  · getter 미노출·지시가 재료 앞(C578)·실패 = 종전 검색 폴백(거동보존)
단위검정: test_sg_docs_delivery.py 6종 + 회귀 3종 ALL PASS
  ★검정이 착수 전에 잡은 함정 = Record ID: 계수기가 엔진-배달분에도 서서 답 폐기(C581 동형)
    → docs 모드 미적용 + 배달 원문을 _ok_outs 시드(마감검증이 배달분 상대로)
x456 3팔(같은 ref 8):  A_repaired  9/20(45%)   B_prerepair 0/8(부정통제 성립)
                       C_docs v1   9/15 — 순증 0·실패 자리가 픽커로 이동
                       C_docs v2   14/16(87.5%) — ref 짝 vs A = +5
픽커 v2(de52d31d): "항목 전수 + 모호 이름 = 전부 포함" — v1 은 계좌 자신의 클래스를 빠뜨림
                   (관문1 문면 base=0.0 으로 확인·gold 무관)
```

**⛔인용 규율**: ⑴n=8 격리·같은 ref ⑵v2 는 v1 실패 문면 보고 고친 1회 반복 = out-of-sample
아님 ⑶gate1_kept 는 **접지 진단이지 reward 아님**([[69]]) ⑷라이브 효과 미측정(=⑥이 판정).

### C586 — read-gold 축 생존 감사 (x461·새 레버 0)

```
T2_PIN_READ       074·079·094·085·073 에서 0줄 (타 6 태스크 51줄 — 레버 자체는 산다)
T2_DEMANDED_STEP  read-미수행 6 태스크 전부 ABSENT
                  발화 표적 분포 = verify 22 · 계좌목록 12 · referral 6 · 카드적합성 3 — 4종 전부
구조 원인          get_bank_account_transactions 피의존 1 (dep=get_interest_correction 뿐)
                  get_payment_history            피의존 1 (dep=check_cli_eligibility 뿐)
                  ⇒ 이 태스크들의 실제 write 에 read 요구 선언이 없다 → 큐가 수요를 못 냄
```

---

## §3 산출물 (전부 tracked·push 완료)

```
a4120fa5  엔진 전달 교체 + A3 docs.instructions(두 층 동일) + 단위검정 + isofb 검정 정렬
de52d31d  픽커 v2 + x456 --arms
b6d88155  x461 감사 스크립트 + JSON
f46e35a7 / 1a7db973  x456 결과 2벌(reports/facet_rft_2026/x456_kb_sub_liveness_cdocs{,_v2}.json)
x461_readgold_lever_liveness.json
```

## §4 이 세션의 자기 결함 (반복 금지)

```
① 로컬 미러가 옛 A3 를 들고 있는 채 배선을 시작할 뻔 — 커밋 존재 검산으로 잡음(§0)
② 픽커 v1 문구가 계좌 자신을 빠뜨림 — 격리 1차가 잡음. 서브 지시는 "항목 전수"를 명시할 것
③ isofb 검정이 2026-08-14 이후 상시 빨간색(수리를 검정이 안 따라감) — 발견 즉시 정렬.
   ⇒ 수리 커밋에는 그 수리가 깨뜨리는 기존 검정 목록 확인을 포함할 것
④ ssh 폴러 PipeTimeout 2회([[30]] 기지 함정) — 원격 완주 대기는 짧은 확인 반복이 아니라
   원격측 파일-존재 루프 + setsid 로그로
```
