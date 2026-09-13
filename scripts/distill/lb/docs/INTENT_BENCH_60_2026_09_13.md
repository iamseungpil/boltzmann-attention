# 손님 의도 소형 벤치 (60 턴, 2026-09-13)

출처: `x768/iso_intent2.py` 가 고른 base 승 궤적(x806 nt4)의 어시스턴트 턴 60개(무작위, seed 7, 앞 60).
라벨은 대화 꼬리(직전 3 발화)를 읽고 사람이 붙였다(Claude 1차, 사용자 검토 대기). 정의:
- **ACTION** — 손님이 계좌·카드·기록의 **변경**을 청했고 아직 안 됨(그 변경을 위한 본인확인 단계도 ACTION).
- **INFO** — 질문·조회·추천·설명을 기다림. 조회는 도구가 필요해도 INFO. 아직 결정 중이면 INFO.
- **HANDOFF** — 사람을 청했고 아직 안 됨.
- **NONE** — 아무것도 남지 않음(마지막 요청 완료 · 감사 · 손님이 직접 하겠다고 함).

| # | task t @idx | 라벨 | v2 답 | 근거(손님 마지막 말) |
|---|---|---|---|---|
| 00 | 049 t0 @65 | ACTION | ACTION | 잔액 갚았으니 (카드 폐쇄) 진행 |
| 01 | 056 t3 @45 | ACTION | INFO | "now I also need a business …" (다음 개설) |
| 02 | 022 t1 @5 | INFO | INFO | 리워드 이상 점검 |
| 03 | 089 t3 @13 | INFO | INFO | ATM 거절 원인 |
| 04 | 025 t3 @17 | INFO | INFO | $100k 결제 가능 여부 |
| 05 | 044 t3 @58 | ACTION | ACTION | "I'd like to apply for the Platinum" |
| 06 | 093 t1 @30 | ACTION | ACTION | 이자 오류 발견 → 정정 |
| 07 | 043 t3 @31 | ACTION | ACTION | "please go ahead and pay that off" |
| 08 | 098 t2 @23 | INFO | INFO | "please check my eligibility" |
| 09 | 093 t2 @13 | INFO | INFO | 이자 확인용 본인확인 |
| 10 | 044 t1 @80 | ACTION | ACTION | 신청 의사 |
| 11 | 022 t1 @102 | ACTION | ACTION | 분쟁 제출 진행 중(남은 건) |
| 12 | 023 t0 @8 | INFO | INFO | 리베이트 자격 확인 |
| 13 | 064 t1 @19 | INFO | INFO | 계좌 추천 조사 중 |
| 14 | 047 t1 @28 | ACTION | ACTION | "close my Silver Zoom Card" |
| 15 | 070 t1 @39 | ACTION | ACTION | Sky Blue 개설(본인확인) |
| 16 | 016 t1 @15 | INFO | INFO | 보너스 미지급 이유 |
| 17 | 080 t3 @79 | ACTION | ACTION | 나머지 두 카드 취소·재발급 |
| 18 | 006 t2 @17 | INFO | INFO | 카드 추천 |
| 19 | 016 t2 @40 | INFO | INFO | 추가 질문 |
| 20 | 064 t1 @40 | INFO | ACTION | "constraints before we commit" (결정 중) |
| 21 | 089 t1 @19 | INFO | INFO | 본인확인 뒤 진단 |
| 22 | 022 t1 @70 | ACTION | ACTION | 분쟁 제출 진행 |
| 23 | 037 t2 @33 | ACTION | ACTION | 교체 카드 주소 + 분쟁 |
| 24 | 028 t0 @64 | ACTION | ACTION | 분쟁 제출 진행 |
| 25 | 017 t3 @25 | INFO | INFO | 리워드 확인 |
| 26 | 005 t2 @17 | ACTION | ACTION | 이메일 변경(우회코드) 요청 |
| 27 | 043 t3 @25 | ACTION | ACTION | 07 과 같음 |
| 28 | 089 t3 @15 | INFO | NONE | 다른 ATM 시도 상담 |
| 29 | 044 t1 @72 | ACTION | ACTION | 신청 의사 |
| 30 | 015 t0 @31 | ACTION | INFO | 추천 링크 생성 요청 |
| 31 | 049 t3 @84 | HANDOFF | HANDOFF | "want to speak to a supervisor" |
| 32 | 004 t3 @20 | ACTION | ACTION | 이메일 변경 |
| 33 | 095 t0 @21 | INFO | INFO | 이자 확인 |
| 34 | 019 t0 @41 | ACTION | INFO | 분쟁 제출 진행(남은 건) |
| 35 | 044 t0 @17 | ACTION | ACTION | Gold 폐쇄 요청, 자격 확인 중 |
| 36 | 093 t1 @23 | INFO | INFO | 본인확인 |
| 37 | 005 t0 @21 | ACTION | ACTION | 26 과 같음 |
| 38 | 047 t1 @63 | INFO | ACTION | "not sure about downgrading" (결정 중) |
| 39 | 052 t0 @81 | INFO | INFO | "when will I be eligible again?" |
| 40 | 098 t3 @16 | INFO | INFO | 추천 자격 확인 |
| 41 | 096 t2 @18 | INFO | INFO | "go ahead and check" |
| 42 | 100 t2 @18 | INFO | ACTION | 추천 보너스 최대 옵션 조회 |
| 43 | 022 t1 @76 | ACTION | ACTION | 분쟁 제출 진행 |
| 44 | 100 t0 @48 | INFO | ACTION | "still need to wait longer?" |
| 45 | 023 t2 @13 | INFO | INFO | 리베이트 확인 |
| 46 | 043 t3 @23 | ACTION | ACTION | 07 과 같음 |
| 47 | 031 t0 @33 | ACTION | ACTION | 분쟁용 카드 뒷자리 제공 |
| 48 | 017 t3 @36 | ACTION | ACTION | "please submit cash-back disputes" |
| 49 | 002 t3 @9 | INFO | INFO | 카드 추천 |
| 50 | 022 t2 @63 | ACTION | ACTION | 분쟁 확인, 다음 건 |
| 51 | 081 t2 @130 | HANDOFF | HANDOFF | "asked to speak with a human" |
| 52 | 017 t1 @18 | INFO | INFO | 리워드 확인 |
| 53 | 003 t1 @17 | INFO | INFO | 카드 추천 |
| 54 | 072 t1 @14 | INFO | INFO | 본인확인(조회) |
| 55 | 096 t1 @20 | INFO | INFO | "let's verify" (조회) |
| 56 | 008 t0 @12 | ACTION | ACTION | 전단 혜택 적용 요청 |
| 57 | 100 t0 @7 | INFO | INFO | 추천 보너스 조회 |
| 58 | 022 t1 @119 | NONE | INFO | 마지막 분쟁 제출 완료 |
| 59 | 047 t1 @42 | ACTION | ACTION | 폐쇄 진행 + 다른 카드 관심 |

**v2(정의 축소) 일치: 51/60 = 85%** (불일치 9: #01 #20 #28 #30 #34 #38 #42 #44 #58). 불일치의 절반은 "결정 중/조회"를 ACTION 으로(#20 #38 #42 #44), 나머지는 남은 반복 작업을 INFO 로(#01 #34) 또는 완료를 INFO 로(#58).
채택 기준(설계서 §5.3): 이 벤치에서 ≥95%(57/60). 라벨 이의는 사용자 검토로 고친다.
