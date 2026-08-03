# 검색 도구·습관 전이 지도 (2026-08-03)

> B4 런 실패 정독에서 나온 것: 32B의 검색 실패는 능력보다 **도구 모양**에서 온다.
> 같은 모델이 `grep -r` 하나로 11,562자를 받으면 탐색을 여러 번 못 돌린다.
> 이 문서는 Claude Code 하네스의 검색 방식을 **전수 나열**하고, 각 항목이
> ⑴도구 모양(기계화 가능) ⑵습관(학습/scale) 중 어디인지, ⑶[[05]] 감사, ⑷오늘의 근거를 붙인다.
>
> ⚠**구현 승인 문서가 아니다.** 근거가 있는 항목과 측정이 필요한 항목을 구분해 적는다.

## 0. 왜 이 문서인가 — 오늘의 실측

| 관측 | 값 |
|---|---|
| shell grep 패턴 길이별 무매칭률 | 1단어 **0%**(n=2) · 2단어 **42%**(n=12) · 3단어+ **71%**(n=7) |
| 무매칭 패턴의 성격 | 전부 자연어 구절 — `'sign-up bonus'`(문서는 `New-Customer Bonus`) · `'get card last 4 digits'`(실제 식별자는 `get_card_last_4_digits`) |
| 0을 받은 뒤의 행동 | **존재 부정**(007) · **무관 문서로 답변 생성**(012) · **앱 경로 날조**(014·015) |
| KB_search 질의 길이 분포 | 1~2단어 10% · 3단어 36% · **≥4단어 54%** |
| 구 매칭 vs 내용어 AND 퇴화율 | **72% → 25%** (질의 200건 전수) |

## 1. 도구 모양 — 기계화 가능

| # | Claude Code | 우리 하네스 현재 | 전이 형태 | [[05]] | 근거 |
|---|---|---|---|---|---|
| T1 | **기본 출력 = 경로만**(`files_with_matches`) | `shell grep -r` = **본문 전체** · `KB_search` = top-k 본문 | 기본을 경로/제목 목록으로, 본문은 요청 시 | 3 NO | 11,562자 1회(005) · 서브 **172,731토큰** 사고(RATE_SUBAGENT §2d) |
| T2 | **결과 수 캡**(`head_limit` 기본 250·`offset` 페이징) | 캡 없음 | 캡 + "N개 중 M개 표시" | 3 NO | 문맥 초과 종료 1건(A 022/t0) |
| T3 | **`output_mode:"count"`** = 개수가 1급 모드 | 없음 | **④가 이것**(이미 구현·`T2_MATCH_COUNT`) | 3 NO | 195 vs 4 · 라이브 부착 149회 |
| T4 | **`glob`/`type` 필터** + 별도 `Glob` 도구 | shell 파일 glob은 있으나 **파일명을 추측** | 파일명 인덱스 노출 | 3 NO | 007/t1 [8] `doc_platinum_rewards_card_*` → 파일 없음 → [10] `cat INDEX.md`로 복구 |
| T5 | **정규식**(ripgrep) | 평문 구(phrase) | **부분문자열 AND**를 기본 매처로 | 3 NO | `'get card last 4 digits'` 구 0 → AND **3건·타깃 포함** |
| T6 | **부분 읽기**(`offset`/`limit`·2000줄 기본) | `cat` 전문 | 페이지 읽기 | 3 NO | 미측정 |
| T7 | **대소문자·부분 매칭 기본** | `-i`는 쓰나 구분자 가정 | 정규화(소문자·영숫자)로 구분자 무시 | 3 NO | `Sign-up`↔`signup` 정규화로 회수 |

**T3는 이미 구현·라이브 검증됨**(B4). T1·T2·T5·T7은 같은 계열이고 서로 보강한다.

## 2. 습관 — 도구로 사지 못하는 것

| # | 습관 | 오늘 32B의 반례 | 성격 |
|---|---|---|---|
| H1 | **관측된 리터럴을 키로 삼는다** | 문맥에 `get_card_last_4_digits`가 있는데 `'get card last 4 digits'`로 침 | 부분 기계화 가능(문맥 식별자 토큰 추출) |
| H2 | **식별자 > 설명** | `'sign-up bonus'`(손님 어휘)로 침 | 학습/scale |
| H3 | **넓게 시작해 좁힌다** | `EcoCard referral`·`Platinum Rewards Card referral`로 **처음부터 좁힘** | 학습 + **후퇴는 기계화 가능** |
| H4 | **0을 "없음"으로 읽지 않는다** | 0 → 존재 부정 / 날조 / 포기 | 학습 · **개수 표면화가 재료 제공** |
| H5 | 구조로 친다(`def`·데코레이터) | 미관측 | 학습 |

**H4가 지배 결함이고, 도구로는 재료만 줄 수 있다.**

## 3. 선행연구 (초록 직독·[S-abstract])

- **[2505.12694] LLM-based Query Expansion Fails…** 축자: *"when the query is ambiguous, causing **biased refinements that narrow search coverage**"*
  ⇒ **H3 반례(015)에 이미 이름이 붙어 있다.** 노벨티 주장 금지·인용 의무([[41]]).
- **[2505.08450] IterKey** 축자: *"generating keywords … validating the answers. **If validation fails, the process iteratively repeats with refined keywords**"* · [M] BM25-RAG 대비 **5~20%**
  ⇒ **후퇴/재정련 처방 선점.** 차이는 *LLM이 정련·LLM이 검증* vs 우리 *기계적 후퇴·엔진이 개수 발급*.
- **[2605.15184] Is Grep All You Need? How Agent Harnesses Reshape Agentic Search** 축자:
  *"**how tool outputs are presented to the model** … remain **under-explored** in agent loops"* ·
  *"grep generally yields higher accuracy than vector retrieval in our comparisons in experiment 1; … overall scores still **depend strongly on which harness and tool-calling style is used**"*
  ⇒ **T1~T2(제시 형식)가 부분 선점.** 단 벤치가 LongMemEval(대화 기억)이고, 비교 축이 **형식**이지 **경계 신호**가 아니다.
- **[2606.11864] CORE-Bench** 축자: *"**sharp drop** from traditional code search to code retrieval in agentic coding settings"*
  ⇒ 배경. 우리 기전과 직접 연결 아님.

**남는 좁은 자리**: 세 편 어디에도 **"회수 결과에 총 매칭 수를 붙여 모델이 자기가 전부를 봤는지 알게 한다"** 는 없다. IterKey는 *답*을 검증하고, 2605.15184는 *형식*을 비교한다. ④의 델타는 그대로다(설계서 §10).

## 4. [[05]] 총괄

§1 전 항목이 3질문 NO다 — 도메인 어휘 0(전부 검색기 일반 기능)·판단 동결 0(모델이 여전히 질의와 해석을 함)·엔진의 도메인 행동 0(검색은 모델이 발주).
⚠단 **인벤토리 densification 위험**은 T4에 있다(파일명 인덱스 노출 = ADB 84.56→66.47 계열). 개수만 주는 T3와 달리 T4는 후보를 늘린다 ⇒ **Δspurious 필수**.

## 5. 순서 제안

1. **T3(④) 완주 판정** — 진행 중. 기본형 효과를 먼저 본다.
2. **T5+T7을 shell로 확장** — 오늘 실패 10건 중 대부분이 shell이고 ④는 `KB_search` 전용이다. shell도 `mutating=False`라 P13 규약상 부착 허용.
3. **후퇴(backoff) 보고** — `[matches] 0 … largest matching subset {referral} → 37`. IterKey 인용 하에.
4. **T1·T2** — 반환 형식. 측정 필요(문맥 감소 vs 재검색 증가).
5. T4는 마지막 — densification 비용이 유일하게 실재.

## 6. 열린 항목

- ④ 코퍼스(`T2_KB_DOCS_DIR`=json 원본)와 **shell이 보는 디렉터리(md 사본+INDEX.md)** 가 같은 집합인지 미확인. 다르면 ④의 분모가 검색기 분모와 어긋난다.
- **§1 O1 정정**: 설계서에 *"INDEX.md 부재"* 로 적었으나 **존재한다**(007/t1 [10]이 `cat INDEX.md`로 698 문서 목록 수신). 설계서 수정 필요.
- 2605.15184 experiment 1 본문 정독(어떤 제시 형식이 이겼는지) — 인용 전 필수.
