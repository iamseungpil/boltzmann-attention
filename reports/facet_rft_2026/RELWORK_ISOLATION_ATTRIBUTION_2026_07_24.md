# RELWORK — 격리-귀속 논문(Track C 후보) 노벨티 지형 (2026-07-24 딥리서치)

> 상위 = `RESEARCH_MASTER.md` §3 C127. 딥리서치 하네스(105 에이전트·claim별 3-표 적대검증·refuted 1건 제외
> 전 생존). 질문 = "의미-선택 오류로 보이는 에이전트 실패가 실은 (a)인접-행 전사 슬립 (b)자기-정박 전파이고
> (c)정보-맞춘 격리로 해리 가능하며 (d)기존 taxonomy는 과잉-귀속" 4요소의 선점 여부.
> 근거 실측 = C124(039 격리 9/9 vs 9/9)·C125(rall20 41735 재발+날조).

## 1. 요소별 노벨티 판정 (합성 F7)

| 요소 | 판정 | 최근접 선행 |
|---|---|---|
| (a) **인접-행 속성-id 전사 슬립** (라이브 에이전트 파이프라인 내) | **OPEN** | PI-LLM(같은-키 간섭·단일턴)·DRE(표 행/열 혼동·단일턴 QA)·Entity Binding Failures(NL 모호성 기반·전사 기전 없음). **에이전트 궤적 내부에서 이 기전을 식별한 논문 없음** |
| (b) 자기-정박 결정론 재현 | **부분 선점** | **Binding Drift(arXiv:2607.18316·6일 전!)**: 다중스텝 도구 에이전트서 wrong-entity 전파·entity-lock 증폭(주입 오류 3.0×) 형식화. 잔여 노벨티 = **자발(비주입) 자기-정박의 결정론 재현 + 전사-슬립과의 해리** |
| (c) **정보-맞춘 격리 귀속 방법론** | **OPEN as applied** (방법론적 선구 존재) | snowballing의 새-세션 재질의·lost-in-multi-turn의 CONCAT 대조(정보-동일·95.1% 회복)·self-conditioning의 반사실 주입. **라이브 도구-에이전트의 per-failure 최소-문맥 vs 전체-궤적 replay는 미청구** |
| (d) **taxonomy 과잉-귀속 비판** | **FULLY OPEN** | aptitude/reliability 분해(대화)가 전제 지지. **격리 replay가 귀속을 뒤집는 것을 보인 논문 없음**. ToolCritic·Entity-Binding = 순수 관찰 라벨링(반사실 검증 0)이라 ready-made foil |

## 2. 필수 인용 계보 (전부 3-0 검증 생존)

- **자기-정박 계보**: Snowballing(2305.13534·새-세션 재질의로 67/87% 자기-오류 인지=격리 프로브의 조상)·
  Self-conditioning(2509.09677·ICLR26·반사실 오류-주입으로 "장문맥 한계 아님" 입증)·
  Lost-in-multi-turn(2505.06120·CONCAT 대조+aptitude −16% n.s. vs unreliability +112%)·
  Self-correction 실패(2310.01798·단 RL-추론 모델은 헤지 필요).
- **바인딩/간섭 계보**: PI-LLM(2506.08184·같은-키 간섭 log-linear·**Track B와 공유 인용=중복 리스크**)·
  DRE(2606.32029·표 참조 오류가 별도 클래스+한 번 틀리면 표 재참조 없이 자기-출력 복사=질적 관찰).
- **라이벌(같은 저자쌍·최근 4주·미검증 프리프린트)**: Entity Binding Failures(2606.30531·wrong-tool 0% vs
  wrong-entity 24-26% 해리·NL 모호성 taxonomy)·**Binding Drift(2607.18316)**. ⚠그들의 re-verifier −79%
  수치는 본 검증서 1-2 refuted — 프레이밍 선점은 실재하나 수치 의존 금지.
- **foil**: ToolCritic(2510.17052·"Required Arguments" 한 버킷에 오타~의미오류 합침·반사실 검증 0·
  teacher-forcing이 자기-정박을 측정에서 구조적으로 제거)·Hallucination Cascade(2606.07937).

## 3. 전략 함의 (Track C 설계)

1. **시급**: 라이벌 저자쌍이 4주에 2편 — 이 라인은 빠르게 움직인다. E-F3-ISO(소급 재감사)가 논문의
   정량 코어이고 open question #4("wrong-entity 중 전사-슬립 vs 모호성 비율")가 정확히 그 실험이다.
2. **프레이밍**: (a)+(c)+(d)를 코어로, (b)는 Binding Drift와의 **해리**(자발·결정론 재현·전사와 구분)로
   재배치. 행동/귀속 수준 유지 — 연상기억 기전 서사는 Track B에 위탁(F8 중복 리스크: PI-LLM 공유 인용
   시 "한 발견의 분할" 인상 → 간섭 단위 차이(record row vs rule clause)를 명시 교차인용으로 방어).
3. **검증 필요 잔여**(open questions): MAST/TRAIL/tau2 부록의 반사실-검증 부재는 **생존 증거 부재에 의한
   주장**이라 1차 문헌 직접 재확인 후에만 (d)를 단정할 것. Binding Drift의 자발-오류 커버리지도 정독 필요.
4. **헤지**: RL-추론 모델의 within-trace 재검증(자기-정박 전제 약화 가능)·DRE 자기-정박은 부록 1례(질적).

## 4. 출처

핵심 8: 2305.13534 · 2509.09677 · 2505.06120 · 2310.01798 · 2506.08184 · 2606.32029 · 2606.30531 ·
2607.18316. foil 2: 2510.17052 · 2606.07937. (전체 소스 목록·원문 인용 = 워크플로 산출
`tasks/wr2gnxoou.output`·세션 로컬.)
