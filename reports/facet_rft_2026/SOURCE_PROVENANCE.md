# 선행논문 원문 provenance 원장 — **줄번호 인용의 재현 경로** (2026-07-18 개설)

> **왜 이 문서가 생겼나** (사용자 지적·2026-07-18): `NABAOS_PREEMPTION_AUDIT` v2가 원문 줄번호를 **~30개**
> 인용했는데 그 원문이 **내 scratchpad에만** 있었다 → **누구도 재현·반증할 수 없었다.**
> ★**이게 v1이 죽은 원인 그 자체다**: v1의 *"[S] grep 전수 2회"* 거짓이 통과한 건 grep 패턴이 나빠서가
> 아니라 **아무도 그 grep을 재현할 수 없어서**다. v2는 인스턴스 5건만 고치고 **원인을 안 고쳤다.**
> `RESEARCH_MASTER` 갱신 프로토콜 위반: **"수치는 정본 doc에 provenance와 영속 · scratchpad-only 인용 금지."**

## 규율 (이후 모든 정독에 적용)
1. **줄번호를 인용하려면 이 원장에 등재**한다. 미등재 줄번호 = **인용 무효**(다음 세션의 나는 그것을 믿지 말 것).
2. **원문 PDF/텍스트는 repo에 커밋하지 않는다** — 저작권 저작물의 재배포다. 대신 **SHA-256 + 정확한 추출
   명령**을 싣는다. 같은 PDF에 같은 명령이면 **줄번호가 결정론적으로 재현**된다(검증 30초).
3. 축자 인용은 **감사 문서 본문에** 남긴다(학술 인용 범위). 원장은 **재현 경로**만 책임진다.
4. **[S] 부재 주장**(=grep 0 hits)은 **검색어를 그대로 기재**한다. 검색어를 안 적은 부재 주장 = 무효.
   (v1 교훈: 내 패턴이 `trained`(과거분사)를 놓쳐 *"학습 언급 2회"*라는 거짓을 **grep의 권위로** 말했다.)

## 재현 절차 (공통)
```bash
curl -sL -o <name>.pdf https://arxiv.org/pdf/<id>       # 아래 sha256과 대조
sha256sum <name>.pdf
pdftotext -layout <name>.pdf <name>.txt                  # poppler pdftotext 4.00
wc -l <name>.txt                                         # 아래 line_count와 대조
```
⚠️ **sha256 또는 line_count가 다르면 줄번호 인용을 신뢰하지 말 것**(arXiv 버전 갱신·poppler 버전차).

---

## 등재 목록

### 1. `2603.10060` — NabaOS ★감사 대상
| 항목 | 값 |
|---|---|
| **arXiv ID** | `2603.10060` (**v1**·`[cs.CR] 9 Mar 2026`) |
| **제목** | *Tool Receipts, Not Zero-Knowledge Proofs: Practical Hallucination Detection for AI Agents* |
| **저자** | Abhinaba Basu (**단독저자**·`mail@abhinaba.com`) |
| **URL** | `https://arxiv.org/pdf/2603.10060` |
| **취득일** | 2026-07-18 |
| **PDF SHA-256** | `9891928acc39ce211c155b11aa4b7eb81d9385bcb4e11d4bd49639c357de3a8a` |
| **PDF bytes** | 327,469 |
| **추출** | `pdftotext -layout` (poppler **4.00**) |
| **line_count** | **891** |
| **인용처** | `NABAOS_PREEMPTION_AUDIT_2026_07_18`(전체) · `RELWORK §4c-7` · `COMPLETION_EVIDENCE_LEARN_DESIGN §0a-1` · 원장 **C111** |
| ⚠️**추출 결함** | **Table 1(`200-210`) = pdftotext가 행 정렬을 깨뜨린다**(Upamāna/Śabda/Abhāva 라벨이 어긋남). **그 표에서 줄번호 인용 금지** — 같은 내용의 산문(`213-219`)을 인용할 것. |

**이 원장 개설 前에 감사 문서가 인용한 줄번호**(위 sha256/line_count에서 재현 가능·핵심만):
`17` 초록(*"receipts that the LLM cannot forge"*) · `144-146` 레인 선점 · `152-153` VerifierQ/SVIP(**검증기** 학습) ·
`225` *"The user can then apply their own judgment"* · `230-232` 런타임이 실행(LLM 아님) · `242` `input_hash` ·
`256` HMAC 서명 · `268` 위조불가 · **`286-287` `facts` = *"from the structured tool output"*(★❷ 판정 근거)** ·
`289` tool adapter가 facts 정의 · `309-311` Stage 4 self-tag(*"the receipt ID it claims as evidence"*) ·
`315-316` Stage 5 pratyakṣa 대조 · `318-319` anumāna=전제 존재만 · `328-330` Stage 6 trust 주석 ·
`336-344` **Verification Prompt 전문**(`evidence:`·`checkable:`) · `347-350` **준수율 92/88/85% = *"those that do
not include the verification metadata"*(★❻ 판정 근거·포맷 발화율)** · `352-358` cooperative 가정 + *"not the sole
verification mechanism"* · `374-375` Computation replay · `394-397` 벤치=튜플 1,800 · `489-490` **SVIP-style =
*"A lightweight classifier **trained** on response features"*(★v1 [S] 거짓의 반증)** · `500` clean FPR ·
`541` 94.2% · `559-561` mis-tagging(준수했는데 틀림) · `598-607` **Table 6 Actual Correctness**(★v1 *"결과 안 쟀다"*의
반증) · `686-688` TOML constitution(block/warn/pass) · `689` *"verification policy rather than behavioral alignment"* ·
`695-698` Lim 1 · `708-710` Lim 4 · `712` Lim 5 · `717-719` Lim 6 · `724-751` threat model ·
`739-741` **Compromised tools = *"If a tool itself returns incorrect data"*(★v1 오귀속의 반증)** ·
`743-745` *"remain effective because they do not rely on self-tags"*(★§3 강등 근거) · `749-751` Reasoning errors.

### 2. `2606.05806` — ToolMaze
| 항목 | 값 |
|---|---|
| **arXiv ID** | `2606.05806` (**v1**) |
| **제목** | *When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents* |
| **저자** | Dongsheng Zhu, Xuchen Ma, Yucheng Shen, Xiang Li, Yukun Zhao, Shuaiqiang Wang, Lingyong Yan, Dawei Yin |
| **URL** | `https://arxiv.org/pdf/2606.05806` |
| **취득일** | 2026-07-18 |
| **PDF SHA-256** | `d62361bf89b15e47d12741cee7848c358f0e6bb939bf93e456e6cc3e0711d70f` |
| **PDF bytes** | 3,287,255 |
| **추출** | `pdftotext -layout` (poppler **4.00**) |
| **line_count** | **2495** |
| **인용처** | `RELWORK §4c-5` · 관리표 행1 · 원장 **C110** |
| **핵심 줄** | `486-496` C1/C2 축자 + 3.66×(TSR +17.85pp vs PRR +4.88pp) · `1686` `33.40`=**Gemini TSR(NP)** 셀 · `1696` `50.57`=MiniMax **C1/P1 PRR**(★유일한 `50.5x`) |
| **[S] 부재** | `50.54` = **0 hits**(검색어: `50\.54`·`50\.5[0-9]`) ⇒ **PRR 33.40→50.54% = DR 스플라이스·인용 금지** |

### 3. `2505.23662` — ToolHaystack
| 항목 | 값 |
|---|---|
| **arXiv ID** | `2505.23662` (**v1**·EMNLP 2025 Findings) |
| **제목** | *ToolHaystack: Stress-Testing Tool-Augmented Language Models in Realistic Long-Term Interactions* |
| **URL** | `https://arxiv.org/pdf/2505.23662` |
| **취득일** | 2026-07-18 |
| **PDF SHA-256** | `a0ed2ff2f92d94008ba3f6e58044f2963b7fece92348786425f8570a3ffe35f9` |
| **PDF bytes** | 3,173,576 |
| **추출** | `pdftotext -layout` (poppler **4.00**) |
| **line_count** | **2041** |
| **인용처** | `RELWORK §4c-6` · 관리표 행5 · 원장 **C110** |
| **핵심 줄** | `271-292` CR-Single/CR-Multi 정의(*"consolidate information from various parts of the conversation"*) · `487` recency bias 94.44%@거리0 · `570-575` In/Out-of-context 분류 · `574` *"CoT prompting is not universally effective"* |
| **[S] 부재** | `parameter hallucination` = **0 hits**(검색어: `parameter hallucination`·대소문자 무시) ⇒ **서베이 경유 표현·그들 명명 아님** |
