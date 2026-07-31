# claim_prov · completion_guard 엔진 승격 설계 (2026-07-31 · **리뷰 대기 · 미구현**)

> 상위 = `A2_THREE_LAYER_SPLIT_DESIGN_2026_07_31.md`(3층) · [[05]] 고정-변경 경계 · [[23]] A2 출처 규율.
> 발단 = [[23]] 소급 감사에서 **이 두 키만 "정책 근거 없음"으로 확정**됐다(§1).

---

## 1. 왜 승격 후보인가 (감사 결과)

[[23]] 소급에서 banking 33키 중 **정책 근거가 아예 없다고 확정된 것은 이 둘뿐**이다.

| 키 | 후보였던 정책 문장 | 판정 |
|---|---|---|
| `claim_prov` | 7항 *"Do not give intermediate responses… that would give away internal rho-bank information/policies"* | **불일치** — 내부정보 유출 금지이지 완료-주장 대조가 아니다 |
| `completion_guard` | (없음 — 정책에 "all of the"·"each"·"every" **0회**) | **없음** |

그런데 **gold 경유도 아니다**(리터럴이 전부 env 도구명·접두사 패턴). 즉 위반이 아니라 **분류
오류**다 — 이 둘이 담은 것은 banking 사실이 아니라 **도메인-일반 무결성 원리**다:

- `claim_prov` = *"한 일을 했다고 말했으면 실행 원장과 대조한다"*
- `completion_guard` = *"고객이 실행해야 하는 것을 내가 했다고 말하지 않는다"*

이건 tau2의 어느 도메인에서도 참이고, 실제로 **선언-우선 R5**(`done_report.tool` ∈ 실행 원장)가
같은 원리를 엔진에서 이미 구현하고 있다. 같은 원리가 **엔진과 A2 양쪽에 따로** 있는 상태다.

## 2. 무엇이 도메인-불변이고 무엇이 아닌가 (내용 실측)

두 키의 값은 **산문 템플릿 + 도메인 결합**이 섞여 있다.

| 조각 | 도메인-불변? | 근거 |
|---|---|---|
| `claim_prov.question` 본문 | **불변** | "audit yourself · list every assertion that YOU have ALREADY performed" — 업종 명사 0 |
| `claim_prov.event_map` 값 | **도메인** | `{"search": ["KB_", …], "give": ["give_"], "transfer": ["transfer_", "request_human"]}` = **env 도구명 접두사** |
| `claim_prov` kind enum | **도메인 일부** | `search|verify|record_update|dispute_file|give|transfer|write` — `dispute_file`이 업종색 |
| `completion_guard.claim_question` | **불변** | "Does your reply state or imply that some action has already been completed…" |
| `completion_guard.feedback` | **불변** | "only the CUSTOMER can, by calling the tool you gave them" — 업종 명사 0 |
| `completion_guard.user_execution_tool` | **도메인** | `call_discoverable_user_tool` = **env 도구명** |

⇒ **산문 전체가 불변이고, 도메인인 것은 이름 결합뿐이다.**

## 3. 설계 — 3층 경계로 가른다

```
L1  a2/base/shared.json          claim_audit = {question, schema, completion_question, feedback}
                                  (산문·불변 · 새 도메인 비용 0)
L2  <domain>.settings.json       claim_bindings = {kinds: [...], event_map: {...},
                                                   user_execution_tool: "..."}
                                  (구조 동일·값만 도메인 · 새 도메인 = 템플릿 채우기)
L3  (없음)                        ← 이 두 키는 L3에서 **사라진다**
```

엔진은 L1 산문에 L2 결합을 **주입**해 서브콜을 만든다(`{kinds}` 자리표시자 치환). 도메인 이름은
엔진 소스에 등장하지 않는다(리터럴 0 유지).

**핵심 이득**: 지금은 새 도메인이 이 원리를 쓰려면 **산문 전체를 복사**해야 한다(banking에만 있고
retail·airline엔 **둘 다 없다** — 실측 확인). 승격 후에는 **결합 3개만** 채우면 된다.

## 4. 왜 R5와 합치지 않는가 (경계)

선언-우선 R5는 **봉투의 `done_report.tool`**을 원장과 대조한다 — 모델이 형식화한 **닫힌 값**을
비교하므로 서브콜이 필요 없다. `claim_prov`/`completion_guard`는 **산문 응답**을 대상으로 하므로
**형식화 서브콜이 필요**하다(열린 술어를 닫는 비용·[[22]]).

⇒ 둘은 **같은 원리의 다른 입력**이다. 합치지 않고 **L1에 나란히** 둔다. 다만 arm③(`GUIDE=1`)에서
봉투가 항상 나오면 R5가 상위 호환이므로, **그때는 서브콜을 끄는 것**이 조정 후보다(중복 개입 =
[[19]] 간섭 감시점).

## 5. [[05]] 결정 3질문 ([[17]] 상설 의무)

1. **scaffold/A2의 도메인-특화를 순증시키나?** — **아니다. 감소한다.** banking L3에서 2키가 빠지고
   L1(비용 0) + L2(템플릿) 로 재배치된다. 산문 바이트가 도메인 회계에서 빠지는 것이 이 설계의 요점이다.
2. **모델이 할 수 있는 유동 판단을 결정론에 동결하나?** — **아니다.** 판정 구조 불변: 모델이
   형식화하고 엔진은 원장 대조만 한다([[10]]). 바뀌는 것은 **산문이 어느 파일에 있느냐**뿐이다.
3. **scaffold가 모델 대신 도메인 행동을 수행하나?** — **아니다.** 실행 경로 변경 0.

⇒ 셋 다 no. 그리고 **행동은 바이트 단위로 같아야 한다** — 다르면 버그다(§6 게이트).

## 6. 등가 게이트 (구현 시 필수)

- 승격 전후로 **엔진에 전달되는 최종 문자열이 동일**함을 단위테스트로 강제한다
  (`render(L1.question, L2.bindings) == 구 A2.question`).
- 3층 병합 등가(`x18 --verify`)·`x17` 층 등가는 그대로 통과해야 한다.
- retail/airline은 이 키가 **없었으므로** 승격 후에도 **비활성**이어야 한다(`claim_bindings` 미선언 =
  레버 skip·U2′ 안전측). 새 도메인에 조용히 켜지면 그게 회귀다.

## 7. 범위 밖 (하지 않는 것)

- **행동을 바꾸는 어떤 것도** 이 커밋에 넣지 않는다. Y2가 도는 중이며, 승격은 Y2 완주 후 반영한다.
- `dispute_file` kind를 일반명(`file`)으로 바꾸는 것은 **별건**이다 — enum 값을 바꾸면 모델 출력이
  달라져 행동이 바뀐다(§6 등가 게이트 위반).
- x6b의 CORE/EXT 분류표 갱신은 승격 구현과 같은 커밋에서 한다(생성물 동기화·`--emit`).

## 8. 한계

- 도메인이 3개뿐이고 이 두 키는 **banking에만** 있다. "불변"은 산문에 업종 명사가 없다는
  **구조적 근거**이지, 다른 도메인에서 실제로 쓰여 검증된 것은 아니다. retail/airline에서 켜보는
  것은 별도 실험이다.
- 승격은 **비용 회계를 정직하게 만들 뿐 성능을 바꾸지 않는다**. pass가 움직이면 등가 게이트가
  깨진 것이다.
