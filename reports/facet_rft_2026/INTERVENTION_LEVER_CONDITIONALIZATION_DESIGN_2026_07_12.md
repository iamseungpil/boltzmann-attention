# 개입레버 조건부화 설계 — 결핍-조건 발화 (2026-07-12)

> 소유: `TRIVIAL_REGRESSION_ABLATION`이 특정한 개입레버(DISAMB·EPLAN·GROUND·PRINCIPLE)의 spurious-misdirection을 도메인일반 조건부화로 제거하는 설계.
> 불변: [[05]] A2만·엔진 도메인일반 · Δspurious≤0 계측 필수(등대 모트) · gold-independence · 무료우선 절단→유료 e2e 승인.
> 상위: `RESEARCH_MASTER §3` · NIGHT 핸드오프 §5(retail-B 개선=레버 조건화) · `B78C_FORENSIC §3`(레버 부작용 vs 진짜 잔여).

## 0. 문제 정의
개입레버가 **가능한-애매성(possible-ambiguity)**에 발화 → 에이전트가 *이미 옳게* 선택한 쉬운 태스크서도 발화해 **오도**. 절단 실증: COMP+DISAMB=0/2·개입레버 각자 trivial 회귀(`TRIVIAL_REGRESSION_ABLATION §4`). 가드레버(PROV-rescue·cap)는 무해=발화조건이 이미 결핍-조건(deny/방지)이라서.

**핵심 재프레이밍**: 가드레버가 안전한 이유 = **검증가능 결핍**에만 발화(막을 것이 실재할 때만). 개입레버를 같은 원리로 전환하면 안전해짐.

## 1. 설계 원리 — 애매성 트리거 → 결핍 트리거
| 레버 | 현재 발화조건(과발화) | 조건부화(결핍-조건·도메인일반) |
|---|---|---|
| **DISAMB** subcall | 후보 \|C\|≥2 = 무조건 재해소 | 에이전트 arg값이 **유효후보 집합 밖**(무효)·또는 판별레코드 **미조회**일 때만. 유효-가용 옵션을 골랐으면 **존중**(재해소 금지). |
| **GROUND** 값치환 | 문맥 후보와 불일치 시 치환 | 에이전트 값이 **출처에 부재**(증명가능)일 때만 치환. 문맥에 존재하면 존중(오탈자-근사만·C64/C65 St≠Street 한정). |
| **PRINCIPLE_DEFAULT** | write operand 기본값 주입 | 필드가 **실제 누락/공란**일 때만 채움. 에이전트가 값을 제공했으면 **덮어쓰지 않음**. |
| **EPLAN** walk deny | 미검토 sibling 있으면 deny | A2-spec이 **관련**으로 표기한 미검토 레코드가 **실재**할 때만 deny. 무관 sibling엔 무발화(현 무조건-walk가 t41 spurious·§2d). |

**공통 원리**: 발화 트리거 = *possible-ambiguity* → **verifiable-deficiency**. 결핍은 gold 없이 도메인일반으로 검증가능:
- **무효(invalid)**: arg값 ∉ 조회레코드의 유효후보집합.
- **누락(missing)**: 요구 필드가 공란/부재.
- **미검토(unexamined)**: A2-relevant 레코드 미조회.

이 셋은 전부 **fetched-record + A2-spec**로 판정 = 도메인일반 엔진 규칙([[05]] 준수·태스크특화 0).

## 2. [[05]] 준수 논증
- 조건부화 로직 = **엔진 고정**(도메인일반 결핍-판정). 변경분 = A2 gate_spec(어느 tool/arg·유효후보 출처·관련-레코드 정의)뿐 = ABox.
- 엔진이 **값을 생성하지 않음**: DISAMB/GROUND/PRINCIPLE은 여전히 deny+피드백 or 에이전트-제공-값 존중이지, 엔진이 write-인자 날조 금지(autofetch류 금지·`CENSUS_LEVERS §1` 정합).
- ⇒ 개입레버 → **가드레버화**(무효/누락/미검토를 막되, 유효선택은 통과). 등대 "레버는 하나 사면 하나 판다"에서 **파는 쪽(trivial spurious)을 0으로**.

## 3. Δspurious≤0 계측 (모트·필수)
각 조건부화 레버는 **재-add 전** 다음 게이트 통과 필수:
- **trivial 무회귀**: 6-fail(또는 36-trivial)서 COMP 대비 pass-rate 저하 0(비결정성 커버·nt4·CI).
- **hard-78 이득 유지**: 조건부화가 하드셋서 원 개입레버의 이득을 죽이지 않음(무효/누락/미검토는 하드서 실재하므로 여전히 발화해야 함).
- 채택 = **회복−퇴행>0**(robust·NIGHT §5 원칙).

## 4. 검증 계획 (무료 절단 → 유료 e2e)
1. **오프라인 결핍-판정 유닛**(무료): DISAMB 무효-검출·PRINCIPLE 누락-검출·EPLAN 관련-미검토 검출을 격리 케이스로 확증(유효선택 오발화 0·결핍 검출율).
2. **task106 절단 재현**(소액): 조건부화 DISAMB → COMP+DISAMB_cond가 106을 **통과**하는지(현 0/2 → 목표 pass). 유효선택(red L도 valid) 존중이 관건.
3. **6-fail nt4**(유료·승인): 조건부화-full vs full vs comp → trivial 회귀 0 ∧ 하드 이득 유지 확인.
4. GO 시 → retail-B 스택 = **COMP + 조건부화 개입레버 + 가드**(regression-safe).

## 5. 리스크·미해결
- **결핍-판정 오탐**: 유효값을 무효로 오판 시 새 오발화 → 유닛(§4.1)서 오탐율 0 확인 필수.
- **DISAMB 유효-후보집합 정의**: "유효-가용"의 출처(product variants available=true 등)를 A2-spec로 도메인일반 표기 가능한가 — retail exchange(variants)·banking(계좌)·airline(편) 4도메인 불변성 확인 필요([[48]]).
- **EPLAN 관련성**: "A2-relevant 미검토"의 관련성 판정이 semantic이면 [[05]] 경계(§2d 기각된 entity-특정 강화와 구분) — 관련성=구조적(같은 요청 스코프 내 미조회)만 허용.
- **잔여 경계**: 조건부화 후에도 안 닫히는 변형-⋈(C56 체계핵·thinking-flat)은 **진짜 경계**=E7 판단실험(learn vs map).

## 6. 다음
- 본 설계 리뷰(4층 적대·`INTEGRATION_PLAN_REVIEW` 규율) → §4.1 오프라인 유닛 구현 → §4.2 task106 절단 → 승인 후 §4.3.
- 실세계 nt4(B·진행중)가 개입/가드 이분을 기대값서 확정하면 본 설계의 표적(어느 레버 우선 조건부화)이 확정됨.
