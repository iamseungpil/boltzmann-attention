# CP2 Stage 0 — R2·R3·R4 단일 적용 계획 (2026-08-23)

**대상 파일** `C:\workspace\ba-frft\scripts\distill\tau2\t2_gate_patch.py` (HEAD `c5d873b1` 시점 12,403줄)
**재료** `CP2_QUEUE_AUDIT_2026_08_23.md` §5 R1~R6 · R2/R3/R4 명세 3건 · 각각의 반증 3건 · **본 계획자의 독립 재검증**
**조건** 전부 오프라인. GPU 접촉 0 · tau2 런 0 · **이 문서 말고 리포 파일 편집 0**.
**표기** ★ = 내가 직접 코드/로그를 돌려 얻은 값. 확정 못 한 것은 "확인 안 됨" 이라고 적었다.

> ⛔ **가장 먼저 읽을 것 — 이번 재검증이 세 명세의 공통 전제를 하나 무너뜨렸다.**
> 감사 §1-A6 과 R3·R4 명세가 공유하는 전제 *"ASUB 우회 턴의 `work` 는 통째로 버려진다 =
> 배달물이 모델에 간 적이 없다"* 는 **틀렸다**. `work` 는 턴당 한 번만 지어지고
> (`t2_gate_patch.py:6814 work = list(state.messages)` · `unified()` 는 `:6643`) 그 뒤로는
> **덧붙이기만** 한다(★파일 전체 `work = ` 대입 전수 = `:6814 :6868 :6870 :6890 :6892 :6914
> :6916 :6949 :6951 :6967 :6969 :7012 :7014 :10273 :10300 :10302`). 그리고 루프 **뒤**에 `work` 를
> 그대로 모델에 보내는 생성기가 여덟 자리 있다 — `:10412`·`:10515`
> (`la.generate(messages=self._system_messages + _work + …)`) · `:10780`·`:10911`·`:11005`·
> `:11628`·`:11736`·`:11952`(전부 `_gen(self, work + […])`).
> ★실측: t7346 halfA+halfB 의 ASUB 우회 **11건 전부**에서 우회 직후 같은 sim 에
> `[T2_CLAIMPROV] window hit` 또는 `[T2_SELFDECL] declared` 가 나온다. 두 문구는 각각
> `:11736 _gen(self, work + [am, …], "agent_claimprov")` 와 `:11952 _gen(self, work + [am, _dp],
> "agent_selfdecl")` **뒤에서만** 인쇄된다(`:11754`·`:11841`·`:11960`). 중간에 루프-A 마커가 끼지
> 않아 같은 턴이 **확정**되는 것이 **8/11**, 중간에 `[T2_SEARCH_ON_PROCEED]`(`:8943` = 루프 안)이
> 끼어 턴 귀속이 근사인 것이 3/11.
> ⇒ 우회 턴에서 실제로 일어나는 일은 *"배달물이 사라진다"* 가 아니라 **"배달물이 커밋되는 발화의
> 생성기에는 못 가고 비커밋 감사 서브콜(claimprov·selfdecl)에만 간다"** 이다. R3 는 이 사실 위에서
> 다시 정당화했고(§2), R4 의 패치 7(`arrived=False` 낙인)은 **기각**했다(§3·§4).

---

## §0 적용 순서와 이유

### 결론: **R3 → R2 → R4**. 축자 앵커로 적용하고, 매 단계 뒤 `grep -n` 으로 앵커를 다시 뜬다.

| 순서 | 항목 | 성질 | 편집 위치(적용 직전 기준) |
|---|---|---|---|
| 1 | **R3** | 거동 변경(조건부) | `:10274-10304` 삭제 → `:10360-10362` 사이로 이동 |
| 2 | **R2** | 거동 변경(플래그 기본 OFF) | `:4472` 뒤 상수 · `:7052` 뒤 2줄 · `:9736-9743` 가드 몸통 · (R3 이동본 안) 3줄 |
| 3 | **R4** | 계기(모델 가시 바이트 불변) | `:4472` 뒤 헬퍼 2개 · `:4489` 1줄 · `:4556` 앞 5줄 · R3 이동본 안 2자리 · `t2_lever_beat.py` 1함수 |

### 왜 이 순서인가

**⑴ R3 가 R4 의 전제다(자리 이동).** R4 의 계기 훅 두 개(`ctx_skip` 종결 · `attached` 종결)는
소비 블록 **안**에 들어간다. R3 는 그 블록을 통째로 `_am_sub` 분기 뒤로 옮긴다. R4 를 먼저 넣으면
R4 가 방금 심은 코드를 R3 가 다시 옮기게 되어 같은 줄을 두 번 편집하고, R4 명세의 OLD 블록이
현행 파일과 안 맞게 된다. **R4 의 OLD 블록은 R3 적용 뒤 파일에서 다시 떠야 한다** — 이 문서 §3 의
R4 OLD 블록은 그렇게 적었다.

**⑵ R2 는 R3 뒤가 낫다(검증 가능성).** R2 는 `:9736-9743` 가드만 만지고 소비 블록은 안 만진다 —
순서상 앞에 놔도 충돌은 없다. 그러나 R3 의 핵심 주장(*"비-ASUB 회차에서 `_gen` 이 받는 `work` 는
바이트 동일"*)은 **루프 제어가 구판 그대로일 때** 가장 깨끗하게 검정된다. R2 를 먼저 넣으면 회차 수
자체가 달라져 R3 의 동일성 검정이 두 변경의 합성 위에서 돌아간다. 또 R2 는 **R3 이동본 안에
세 줄**(`cp2_attached += 1` 과 주석)을 심어야 하므로 R3 가 이미 자리를 잡고 있어야 한다.

**⑶ R4 가 마지막인 이유(불변식 검정).** R4 의 회귀 검정은 *"`self._t2_cp2_pending` 대입 자리가
정확히 3곳"*, *"소비 지점이 `_gen` 호출과 인접"* 같은 **최종 배치**에 대한 단언이다. 최종 배치가
확정된 뒤에 걸어야 의미가 있다. 그리고 R4 는 유일하게 모델 가시 바이트를 **안** 바꾸므로, 앞선 두
거동 변경이 이미 안착한 위에서 계기를 얹으면 계기 자체가 회귀 원인이 될 여지가 없다.

### 줄번호 밀림

- R4-ⓑ(헬퍼 2개, `:4472` 뒤)는 **약 +90줄**을 파일 앞쪽에 넣는다 ⇒ 그 아래 **전부**가 밀린다.
  R4 를 마지막에 두면 다른 패치의 앵커가 그 밀림을 겪지 않는다.
- R2 는 가드 몸통에 **약 +40줄**을 넣어 `:9743` 아래(=R3 가 만진 영역)를 밀지만, R3 는 그 시점에
  이미 적용돼 있고 R4 의 앵커는 축자 문자열이므로 무해하다.
- R3 는 블록을 **아래로 이동**시키므로 순 증감 ≈ +20줄(주석)뿐이다.
- ⚠**줄번호로 적용하지 마라.** 세 명세가 인용한 줄번호 중 감사서 계열(`:9695` `:10243` `:10257`
  `:10259` `:10320` `:9061`)은 커밋 `844fa7a2` 이후 **+41 밀려 있다**(★검산: `10284−10243 =
  10298−10257 = 10361−10320 = 41`). 이 문서의 OLD 블록은 전부 **오늘 HEAD 축자**다.

### 적용 전 메인 세션이 **먼저 정해야 하는 것 두 가지**

1. **R2 를 켤 것인가.** R2 는 `T2_CP2_HOLD` 기본 **0** 으로 들어간다(=바이트 동일). 켜면 t7346 에서
   달라지는 턴 10개가 **전부 그 sim 의 마지막 assistant 메시지**이고 **그중 6개가 이미
   `reward=1.0`** 이다(★§1.4 에서 독립 재현). 상방은 거의 없고 하방은 통과 중인 6 sim 이다.
2. **R3 를 R2 없이 낼 것인가.** R3 단독은 우회 턴의 claimprov·selfdecl 서브콜에서 재료를 **뺀다**
   (오늘 11/11 이 받고 있다). R2 를 켜면 그 재료가 같은 턴 다음 iteration 에서 커밋 생성기로 들어가고
   claimprov·selfdecl 도 다시 받는다 ⇒ **R3 의 판 것은 R2 가 꺼져 있을 때만 발생한다**(§2.5-3).

---

## §1 R2 — 소비 도달성(break 가드)

### 1.1 지금 무엇이 참인가 (축자)

`t2_gate_patch.py:9736-9743`:
```python
            if (not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_fb is None and un_fb is None and dr_fb is None and pc_fb is None
                    and pr_fb is None and hv_fb is None and dd_fb is None and sig_fb is None
                    and proc_fb is None and abs_fb is None and tr_fb is None and wd_fb is None
                    and fs_fb is None and rdd_fb is None
                    and dw_fb is None and en_fb is None):
                break
```
- 배달 슬롯은 **`self` 속성**이다(`:4556 self._t2_cp2_pending = text`). 가드가 열거하는 22종은 전부
  **지역 `*_fb`** 라 슬롯을 못 본다.
- 소비 지점(`:10284`)은 이 break **뒤**다. ⇒ 어떤 `*_fb` 도 안 선 턴은 여기서 끊기고 배달물이
  소비에 못 닿는다. ★루프-레벨 `break` 는 `:9743` 하나뿐이고 `continue` 는 `:7067`(가드 앞).
- 자리별 운명이 갈린다: `VIEW_FB`(`:9094`)는 `_ar.status == "deny"` 블록 안이고 그 블록 끝
  `:9131` 이 `rw_fb` 를 무조건 세운다 ⇒ 구조적으로 안 끊긴다. `SEARCH_ON_PROCEED`(`:8943`)는
  조건이 `not deny`(`:8913`)라 **어떤 fb 도 안 세운다**.
- ★실측(t7346 halfA+halfB 40 sim · 명세·반증·나 셋이 독립 재현해 일치):
  대입 125 = 부착 100 + 클로버 11 + 병합흡수 2 + **미소비 잔존 12**. 잔존 12건(26,698자)은
  **전부 `SEARCH_ON_PROCEED`**. 자리별 부착률 `VIEW_FB` **64/64** · `SEARCH_ON_PROCEED` 36/61.
- 계기가 자기 결함을 통과시킨다: `test_regen_break_guard.py:48` 축자
  `if isinstance(n, ast.If) and len(n.body) == 1 and isinstance(n.body[0], ast.Break):` 이고
  수집 술어는 `ast.Name` 중 `_fb` 로 끝나는 것만 모은다 ⇒ `ast.Attribute` 인 슬롯은 **원리상**
  안 보인다. ★오늘 돌리면 `RESULT: ALL PASS`.

### 1.2 반증이 바꾼 것 (원안 → 채택안)

| 원안 | 반증 | 채택안 |
|---|---|---|
| 플래그 없이 상시 ON. 근거 = *"다섯 배달 env 가 전부 OFF 면 바이트 동일"* | 다섯을 한 번에 끄는 env 가 **없고** 라이브 PIN 은 `T2_SEARCH_ON_PROCEED=1`(`run_t7346_overnight_stage1_20260822.sh:87`)이라 **OFF 상태가 존재하지 않는다** | **`T2_CP2_HOLD` 신설·기본 0.** OFF 에서 바이트·stderr 완전 동일 |
| `cp2_rounds += 1` 을 가드 **앞**에서 무조건 | fb 가 선 iteration(예: 순수-조언 `rw_fb = (None, _sub17)` `:8846`)에서도 예산이 타서 정작 필요한 자리에 hold 가 없다 | **증가를 가드 `if` 안으로** — 다른 22 항이 전부 참이라 *실제로 break 를 막았을 때만* 오른다 |
| `_CP2_HOLD_MAX = 2` | hold 가 만든 회차 **다음** iteration 에서 루프-A 가 다시 돌아 **다른 축의 무라벨 결정문**이 같은 프롬프트에 두 번째로 실릴 수 있다(감사 §4.1-① 1순위 위험 · 수리는 R6 인데 큐 A/B 와 같은 런 금지) | **`_CP2_HOLD_MAX = 1` + `cp2_attached == 0` 이중 봉인.** R2 는 그 턴의 **첫 부착만** 만들 수 있고 두 번째는 구조적으로 못 만든다 |
| `int(os.environ...)` 크래시를 상수 주석의 근거로 인용 | 감사 §2 축자 *"`int()` 크래시(finding 5)는 이 변경의 결함이 아니다 … 같은 파일에 52곳"* — **기각된 소견** | 그 문장 **삭제** |
| *"[[62]] 새 판단 0 — 한 번 더 생성할 뿐"* | hold 는 루프 몸통의 남은 절반(`:9744~:10362`)을 실행시킨다. ⚠단 루프-A(`:7053~:9735`)는 **같은 iteration 에서 이미 돌았다** — 가드는 몸통 **중간**의 조기 탈출이다 | 주석 문구 정정. 새 결정 생성은 **다음 iteration** 의 루프-A 에서 가능하고, 그것을 `cp2_attached == 0` 이 봉인한다 |
| *"반사실 아님 — 10건이 그 턴에 붙는다"* | 앞 회차를 바꾸는 순간 뒤는 전부 새 분기 | 확정 문장은 *"그 iteration 에서 `_gen` 이 cp2 를 포함한 `work` 로 불린다"* 까지 |
| *"클로버 11건도 R2 의 하류"* | 그 11건의 **앞 턴 `am`** 을 안 봤다 | 주장하지 않는다 |
| §5 검정의 `guard_of` 수리 = `n.body[-1] is Break` | 새 몸통이 `if/else` 라 `body[-1]` 도 Break 가 아니다 | **`ast.If` 서브트리 안에 루프-레벨 `Break` 가 있는 것**으로 일반화 |

### 1.3 최종 패치 (축자 OLD / NEW)

#### R2-ⓐ 모듈 상수 — `t2_gate_patch.py:4472` 바로 아래

**OLD**
```python
_CP2_GUARD_MIN = 5000
```
**NEW**
```python
_CP2_GUARD_MIN = 5000

# ★한 턴에서 cp2 슬롯이 재생성 루프의 break 를 막을 수 있는 횟수 상한 (2026-08-23·R2).
# 왜 상한이 필요한가: 이 루프는 `while True` 이고, 지금까지 루프를 붙잡던 이유는 전부
# **매 iteration 다시 계산되는 지역 `*_fb`** 였다. cp2 는 `self` 속성이라 다시 계산되지 않는다 —
# 상한이 없으면 정지가 다섯 개의 다른 카운터가 계속 옳기에 걸린다. 여기 한 줄로 두면 정지는
# 구조가 보증한다.
# ⚠**1 인 이유**(2026-08-23 반증이 잡은 자리): 2 로 두면 hold 가 만든 회차 다음 iteration 에서
#   루프-A(`:7053~:9735`)가 다시 돌아 **다른 축의 결정문**이 새로 배달되고, 그것이 같은 `work`
#   에 두 번째로 실린다. 그 문자열에는 축 라벨이 없어(감사 §4.1-① 1순위 위험) *"정정된 답"* 과
#   *"다른 질문의 답"* 을 문면으로 가릴 수 없다. 원천 수리는 R6 이고 큐 A/B 와 같은 런에 넣을 수
#   없으므로, 여기서는 **애초에 만들지 않는다**.
# ⚠env 로 읽지 않는다 — 상한이 실험 조건이 되면 그 자체가 또 하나의 미측정 자유도다.
_CP2_HOLD_MAX = 1
```

#### R2-ⓑ 턴-로컬 카운터 — `t2_gate_patch.py:7052`

**OLD**
```python
        absent_fired = False  # ★D1′: 부재 표면화는 **턴당 1회**(재생성 루프가 같은 문구를 도배하지 않게)
```
**NEW**
```python
        absent_fired = False  # ★D1′: 부재 표면화는 **턴당 1회**(재생성 루프가 같은 문구를 도배하지 않게)
        cp2_rounds = 0        # ★R2: cp2 슬롯이 이 턴에 **실제로 break 를 막은** 횟수(상한 `_CP2_HOLD_MAX`)
        cp2_attached = 0      # ★R2: 이 턴에 cp2 가 `work` 에 실제로 붙은 횟수(≥1 이면 hold 금지)
```

#### R2-ⓒ 가드 — `t2_gate_patch.py:9736-9743` (앞 주석 `:9729-9735` 는 **그대로 둔다**)

**OLD**
```python
            if (not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_fb is None and un_fb is None and dr_fb is None and pc_fb is None
                    and pr_fb is None and hv_fb is None and dd_fb is None and sig_fb is None
                    and proc_fb is None and abs_fb is None and tr_fb is None and wd_fb is None
                    and fs_fb is None and rdd_fb is None
                    and dw_fb is None and en_fb is None):
                break
```
**NEW**
```python
            # ★R2 — 배달 채널이 **로컬 fb 가 아니라 슬롯 속성**이라 이 가드가 못 본다
            #   (2026-08-23·t7346 40 sim 독립 census 3인 일치). 위 `proc_fb` 사고와 **같은 사고**다:
            #   `_t2_cp2_pending`(`:4556`)은 `self` 속성이고 소비 지점은 이 break **뒤**라, 그 턴에
            #   어떤 `*_fb` 도 안 서면 여기서 끊겨 배달물이 소비에 닿지 못한다. 자리별 실측이
            #   기전을 그대로 찍는다 — 대입 125 중
            #     `VIEW_FB`(`:9094`)           64 대입 / **64 부착** / 유실 0
            #        └ `_ar.status == "deny"` 블록 안이고 그 블록 끝(`:9131`)이 `rw_fb` 를 무조건
            #          세운다 ⇒ 구조적으로 안 끊긴다.
            #     `SEARCH_ON_PROCEED`(`:8943`) 61 대입 / 36 부착 / **25 유실**
            #        └ 조건이 `_ar.status != "deny"`(`:8913`) 라 **어떤 fb 도 안 선다** ⇒ 같은 턴에
            #          다른 레버가 **우연히** 발화했을 때만 실려 갔다. 부착은 실력이 아니라 우연이었다.
            #   미소비 잔존 12건 26,698자가 그 값이고 12건 **전부가 SEARCH_ON_PROCEED** 다.
            #   C502 축자: *"로그 마커 ≠ 도달"* — `[T2_SEARCH_ON_PROCEED] … 재료 254자 배달` 은
            #   조립이 아니라 **인쇄**였다([[55]] 우리 배관 먼저 · [[25]] 우리 계기는 100% 정답 의무).
            # ⚠**기본 OFF 다.** 라이브 PIN 은 `T2_SEARCH_ON_PROCEED=1` 이라 슬롯이 늘 차 있고,
            #   *"배달 자리가 전부 꺼지면 바이트 동일"* 이라는 안전 논거는 **성립하지 않는다**
            #   (2026-08-23 반증). 그래서 게이트를 따로 만든다 — OFF 면 `_cp2_hold` 가 항상 False =
            #   구판과 바이트·stderr 완전 동일.
            # ⚠**호출이 없는 턴에만** 붙잡는다. `:10003 fb = [am]` 아래 조립은 *"무언가 플래그됐다"*
            #   를 전제해서, 아무 fb 도 없이 통과시키면 `am` 의 모든 tool_call 이 `else`(`:10054`)로
            #   떨어져 `_FB_GENERIC`(`:9986` *"resolve the flagged call(s) first"*)을
            #   `ToolMessage(error=True)` 로 받는다. 아무것도 플래그되지 않았는데 그렇게 말하는 것은
            #   **날조**이고([[25]]), 그 문구는 x246 에서 **정체 3/8 ↔ 원본 본문 0/8** 로 해롭다고
            #   측정된 바로 그 문자열이다(C414·[[64]]).
            # ⚠**이 턴에 이미 부착이 있었으면 붙잡지 않는다**(`cp2_attached`). R2 가 만들 수 있는
            #   것은 그 턴의 **첫 부착**뿐이다 — 두 번째 무라벨 결정문을 같은 프롬프트에 넣는 일은
            #   구조적으로 못 한다(R6 와의 분리·감사 §5 R6 ⚠).
            # ⚠[[62]] 새 판단 0 — 고르지도 순위 매기지도 않는다. 이 iteration 의 남은 절반을 마저
            #   돌게 할 뿐이다(가드는 루프 몸통 **중간**의 조기 탈출이다 — 루프-A `:7053~:9735` 는
            #   이 iteration 에서 이미 돌았다).
            _cp2_hold = bool(os.environ.get("T2_CP2_HOLD") == "1"
                             and getattr(self, "_t2_cp2_pending", None)
                             and not (getattr(am, "tool_calls", None) or [])
                             and cp2_attached == 0
                             and cp2_rounds < _CP2_HOLD_MAX)
            if (not do_gate and not do_prov and ep_fb is None and cons_fb is None
                    and ra_fb is None and te_fb is None and wev_fb is None and rw_fb is None
                    and tl_fb is None and un_fb is None and dr_fb is None and pc_fb is None
                    and pr_fb is None and hv_fb is None and dd_fb is None and sig_fb is None
                    and proc_fb is None and abs_fb is None and tr_fb is None and wd_fb is None
                    and fs_fb is None and rdd_fb is None
                    and dw_fb is None and en_fb is None):
                if _cp2_hold:
                    # ★예산은 **여기서만** 오른다 — 다른 22 항이 전부 참이라 이 hold 가 실제로
                    #   break 를 막았을 때만. 가드 밖에서 올리면 fb 가 선 iteration 이 예산을 태워
                    #   정작 필요한 자리에 hold 가 없다(2026-08-23 반증 ⑷).
                    cp2_rounds += 1
                    print("[T2_CP2_HOLD] break 를 1회 유예 (%d자 대기 · am 호출 0 · %d/%d)"
                          % (len(self._t2_cp2_pending), cp2_rounds, _CP2_HOLD_MAX),
                          file=_sys.stderr, flush=True)
                else:
                    if (os.environ.get("T2_CP2_HOLD") == "1"
                            and getattr(self, "_t2_cp2_pending", None)):
                        # [[64]] 무엇이 막았고 무엇이 남는지 **둘 다** 적는다. 이 자리는 손실이
                        # 아니라 **유예**다 — 슬롯이 `self` 라 다음 턴까지 산다. 유예를 손실로
                        # 인쇄하면 다음 census 가 또 틀린다([[25]]).
                        print("[T2_CP2_UNHELD] 이 턴엔 못 실었다 (%d자 · am 호출 %d개 · 부착 %d회 "
                              "· hold %d/%d) — 슬롯에 남긴다"
                              % (len(self._t2_cp2_pending),
                                 len(getattr(am, "tool_calls", None) or []),
                                 cp2_attached, cp2_rounds, _CP2_HOLD_MAX),
                              file=_sys.stderr, flush=True)
                    break
```

#### R2-ⓓ 부착 카운터 — **R3 이동본 안**. R3 적용 뒤 `am = _am_sub or _gen(...)` 다음 줄

**OLD**(R3 적용 뒤 파일 기준)
```python
            am = _am_sub or _gen(self, work, bw(), "agent_response_unified_regen",
                                 tool_choice="required" if force_required else None, pin=_pin_r)
```
**NEW**
```python
            am = _am_sub or _gen(self, work, bw(), "agent_response_unified_regen",
                                 tool_choice="required" if force_required else None, pin=_pin_r)
            if _cp2:
                # ★R2: 이 턴에 cp2 가 실제로 `work` 에 실려 생성기에 넘어갔다. 다음 iteration 의
                #   가드는 이 사실을 보고 **더는 붙잡지 않는다** — 두 번째 무라벨 결정문 금지.
                cp2_attached += 1
```

> ⚠`_cp2` 는 R3 이동본이 매 iteration 대입하므로 이 자리에서 항상 정의돼 있다. `_am_sub` 가 참인
> 회차에는 R3 가 `_cp2 = None` 으로 두므로 이 카운터는 안 오른다.

### 1.4 거동 델타 ([[70]])

**`T2_CP2_HOLD != "1"`(기본): 바이트·stderr 완전 동일.** `_cp2_hold` 가 항상 False 이고 두 `print`
가 전부 `T2_CP2_HOLD == "1"` 안에 있다.

**`T2_CP2_HOLD=1` 일 때 달라지는 정확한 조건**
> `_t2_cp2_pending` 이 비어 있지 않고 **and** `am.tool_calls` 가 비어 있고 **and** 이 턴에 아직 cp2
> 부착이 0회이고 **and** 이 턴의 hold 가 0회이고 **and** 이 iteration 에서 22개 `*_fb` 가 전부 None
> 이며 `do_gate`/`do_prov` 가 False 인 iteration. 그 iteration 은 `break` 하지 않고 소비 지점까지
> 내려가 배달물을 `work` 에 붙이고 `_gen` 을 한 번 더 부른다.

**t7346 40 sim 반사실(★실측)**

| | 건수 | 자수 |
|---|---|---|
| 대입 턴에 그대로 부착(그 턴 `am.tool_calls == []`) | **10** | **19,471** |
| 호출이 있어 그 턴엔 안 싣고 슬롯에 유예(task_050 ×2) | 2 | 7,227 |
| 새로 생기는 `_gen` 호출 | **≤ +10 / 40 sim** | — |

**판 것**
1. **대조군 동일성.** 켜는 순간 ctl·treat **둘 다** 갈린다 ⇒ **t7346 을 큐 A/B 의 reference 로 쓸 수
   없다**(감사 Stage 2 의 "직전 런을 대조군으로 쓰지 않는다"가 여기서도 구속력을 갖는다).
2. **★가장 위험 — 통과 중인 6 sim 의 마지막 발화.** ★내가 `*.results.json.gz` 로 독립 재현했다:

   | sim | 배달 turn | 마지막 assistant | reward | 그 다음 메시지 |
   |---|---|---|---|---|
   | task_003 s626729 / s373753 | 6 / 6 | 6 / 6 | **1.0 / 1.0** | user `apply_for_credit_card` |
   | task_024 s626729 / s373753 | 8 / 6 | 8 / 6 | **1.0 / 1.0** | user `apply_for_credit_card` |
   | task_100 s626729 / s373753 | 22 / 22 | 22 / 22 | **1.0 / 1.0** | user `submit_referral` |
   | task_072·093×2·074 | 75·36·72·52 | 동일 | 0.0 | 손님 종료 |
   | task_050 s626729 / s373753 | 32 / 50 | 36 / 53 | 0.0 / 1.0 | (패치가 **안** 실음) |

   ★`task_003#s626729` 궤적 축자: `msgs[6] assistant tool_calls=[]` → `msgs[7] user
   tool_calls=['apply_for_credit_card']` → `msgs[8] tool` → `reward=1.0`. **에이전트의 종단 산문
   바로 뒤에 손님이 실행하는 write 가 붙는다** ⇒ 그 산문을 재작성하면 손님이 무엇을 신청하는지가
   바뀔 수 있고 `reward` 는 궤적 재실행 DB 해시다([[69]]). 상방은 거의 없고(재료가 이미 늦었다)
   하방은 통과 중인 6 sim 이다.
3. **핀 승차.** `_gen` 호출은 `pin=_pin_r`(`:10361`)이고 `_pin_r` 는 `go_stack.sh:327 export
   T2_PIN_READ_STEPS=1` · `:328 export T2_PROC_PIN_REARM=1` 에서 나온다 ⇒ 산문으로 끝나던 턴이
   **도구 호출 턴으로 뒤집힐 수 있다**. [[57]] 의무: 새 턴의 `[T2_PIN_READ]`·`[T2_READ_ROUTINE]`
   건수와 over-action(gold 없는 write)을 **짝으로** 센다.
4. **2/12 는 그 턴에 안 준다**(task_050 ×2 · 7,227자). `_FB_GENERIC` 날조를 안 하려고 판다.
   ⚠그 둘이 **잔존 12건 중 유일한 비-종단 턴**이다 — 즉 이 절단선은 도움이 될 수 있는 유일한 두
   자리를 뺀다(반증 §5). 그래도 날조보다는 낫다는 판단이고, `_FB_GENERIC` 이 고쳐지면 이 조건은
   재검토 대상이다.
5. **계기 부수 이동**: 새 iteration 에서 `t2_fbsidecar.record_many(fb=[am], …)`(`:10270`) ·
   `_t2_silenced` 드레인 · `t2_stack.audit`(`:10173`)이 전에 안 돌던 턴에 돈다. 사이드카 행 수와
   `[T2_STACK] audit` 줄이 늘어 `x341_docbody_verdict.py` 같은 판정기의 분모가 움직일 수 있다.
6. **`_ctx_fits` 새 전손 경로**: 8,735자 잔존 2건이 이제 `_CP2_GUARD_MIN` 검사를 받는다.
   `hist+len > 85,596` 이면 `[T2_DOC_DELIVERY] skipped` 와 함께 슬롯이 비워진다 — 오늘도 잃던
   것이라 더 나빠지진 않지만 침묵 잔존이 **기록된 폐기**로 바뀐다.

**판 것이 아닌 것**: 새 선택·순위 0([[62]]) · 도메인 어휘 0([[05]]) · gold 무접촉([[23]]) ·
`state.messages` 불변(C298 replay 불변식).

### 1.5 검정

**파일: `scripts/distill/tau2/test_regen_break_guard.py` 확장**(새 파일 금지 — [[67]]).

- **선행 필수 — `guard_of()` 일반화.** 현재 `:48` 축자
  `if isinstance(n, ast.If) and len(n.body) == 1 and isinstance(n.body[0], ast.Break):` 는 새 몸통
  `[If(hold) / else[…, Break]]` 을 못 찾아 §① 이 `found=0` 으로 **FAIL** 한다(★반증자가 패치본에서
  실행 확인 — *"조용히 건너뛴다"* 는 원안 서술은 틀렸고 실제로는 FAIL 이다. 어느 쪽이든 22-fb 대조가
  **무검정**이 되므로 수리는 필수). 술어를 *"루프 몸통 최상위 `ast.If` 중 서브트리에 루프-레벨
  `Break` 를 가진 것"* 으로 바꾼다. 가드 `test` 는 **바이트 그대로**이므로 §① 의 22-fb 대조는
  그대로 통과해야 한다.
- **① 수집 술어 확장 — 채널 ≠ `*_fb`.** 슬롯 채널 = ⓐ 모듈 어딘가에서 `self.A` 가 대입 타깃이고
  ⓑ 가드 lineno 초과 ~ 루프 끝 구간에서 `A` 를 읽으며 ⓒ 그 읽기가 든 statement 서브트리에 `work`
  또는 `fb` 가 함께 나오는 속성명. 모든 슬롯 채널은 가드 `test` 안에서 이름이 불리거나
  `_KNOWN_UNGUARDED_SLOTS`(사유 + 소유 수리 id 동봉)에 등재돼야 한다. 오늘 코드에 돌리면
  `_t2_cp2_pending` 이 잡혀야 한다. 독스트링에 남길 문장: *"`proc_fb` 는 가드가 채널을 전부
  열거해야 한다고 가르쳤고, 2026-08-23 census 는 **채널이 지역 변수와 같은 것이 아니라고**
  가르쳤다 — 그래서 옛 술어는 원리상 눈이 먼 채 ALL PASS 를 찍었다."*
- **② 가드 식을 소스에서 뽑아 실행**(문자열 대조 금지):
  `compile(ast.Expression(body=guard.test), "<guard>", "eval")` 후 all-None → **True**, 각 fb 하나씩
  non-None(22회) → **False**. ⚠`_cp2_hold` 는 이제 가드 `test` 밖이므로 이 표에 안 들어간다.
- **③ `_cp2_hold` 식을 소스에서 뽑아 실행** — 진리표(stub `self`/`am`/env):

  | `T2_CP2_HOLD` | `_t2_cp2_pending` | `am.tool_calls` | `cp2_attached` | `cp2_rounds` | 기대 | 근거 |
  |---|---|---|---|---|---|---|
  | unset | `"x"*254` | `[]` | 0 | 0 | **False** | 기본 OFF 동일성 |
  | `"1"` | `None` | `[]` | 0 | 0 | **False** | 배달물 없음 |
  | `"1"` | `""` | `[]` | 0 | 0 | **False** | 빈 배달물은 배달이 아니다 |
  | `"1"` | `"x"*254` | `[]` | 0 | 0 | **True** | t7346 실측 10건이 이 칸 |
  | `"1"` | `"x"*254` | `[tc]` | 0 | 0 | **False** | `_FB_GENERIC` 날조 금지 |
  | `"1"` | `"x"*254` | `[]` | **1** | 0 | **False** | 두 번째 결정문 봉인 |
  | `"1"` | `"x"*254` | `[]` | 0 | **1** | **False** | 정지 보증 |
- **④ 정지 보증(행동)** — 슬롯을 매 iteration 다시 채우는 stub 위에서 `_cp2_hold` 식을 돌려
  `_CP2_HOLD_MAX + 1` 회 안에 False 가 되는지 단언. 상한이 없으면 이 검정은 **영원히 안 끝난다** —
  그것이 이 검정의 존재 이유다.
- **⑤ 회계 위치(AST)** — `cp2_rounds += 1` 이 가드 `If` **몸통 안**에 있고 그 밖에는 없음.
- **⑥ 소비 지점이 가드 뒤(AST)** — `guard.lineno` < (`_t2_cp2_pending` 을 `_cp2` 로 읽는 `Assign`
  의 lineno) < 루프 끝. 누군가 소비를 가드 앞으로 올리면 이 검정이 **의도적으로** 갱신되게 만든다.
- **⑦ `_CP2_HOLD_MAX` 계약** — 모듈 레벨 `int` 이고 `== 1` 이며 소스에 `os.environ` 이 없다.
- **⑧ 회귀 배터리** — 기존 §①②③ + `test_route_trace.py` 22/22 + `test_cp2_queue_behavior.py`.

### 1.6 이 패치가 틀릴 수 있는 두 가지

**⑴ 새 재생성은 중립이 아니다 — 핀이 같이 탄다.** §1.4-3. Stage 1 스모크에서 새 턴의
`[T2_PIN_READ] pinned`·`[T2_READ_ROUTINE]` 건수와 over-action 을 세고, 0 이 아니면 메인 세션이
ⓐ그대로 두고 보고 / ⓑcp2-only hold 에서 `_pin_r = None` 중 **명시적으로** 고른다. 나는 ⓑ를
스펙에 넣지 않았다 — 넣으면 cp2 만 22채널 중 유일하게 다른 재생성 경로를 갖고, 그 비대칭이 다음
표류의 씨앗이다.

**⑵ `not am.tool_calls` 가 잘못된 절단선일 수 있다.** 진짜 결함은 호출의 존재가 아니라
`_FB_GENERIC` 날조다. 그것이 고쳐지면 이 조건은 12건 중 2건을 계속 못 싣는 불필요한 제약으로
남는다. 그리고 *"그 둘도 다음 산문 턴에 간다"* 는 **OFF 팔 궤적 위의 반사실**이다 — 확정된 것은
*"10건이 대입 턴에 붙는다"* 뿐이다([[55]]).

---

## §2 R3 — ASUB 우회

### 2.1 지금 무엇이 참인가 (축자)

소비 지점 `t2_gate_patch.py:10284-10304` · 분기 지점 `:10356-10362`:
```python
10356            _am_sub = None
10357            if (os.environ.get("T2_ACTION_SUB") == "1" and rw_fb
10358                    and not force_required and not _pin_r
10359                    and getattr(self, "_t2_asub", None)):
10360                _am_sub = _gen_action_sub(self, state, self._t2_asub)
10361            am = _am_sub or _gen(self, work, bw(), "agent_response_unified_regen",
10362                                 tool_choice="required" if force_required else None, pin=_pin_r)
```
- `_gen_action_sub`(`:6583-6641`)는 `work` 를 인자로 받지 않고 `state.messages` 로 자기 문맥을
  새로 짓는다(`:6601-6602` `:6626` `:6632`). cp2 는 **비커밋**이라 거기 없다 — `:10280-10281`
  자기 주석 축자: *"⚠여전히 **비커밋**이다: `work` 는 생성-시점 버퍼이고 `state.messages` 가
  아니다"*.
- 단락 평가로 `_am_sub` 가 참인 회차에는 `_gen` 이 **불리지 않는다**. 그런데 `:10298` 은 이미
  슬롯을 비웠고 `:10303` 은 "부착" 을 인쇄했다.
- ★실측 t7346 halfA+halfB: 부착 100 중 **11건**이 부착 직후 `[T2_ACTION_SUB]`. 축자
  (`bank_t7346_halfB_20260822.log.gz`): `:12985 [sim=task_074#s373753] [T2_DECISION_CARRY] 이 턴
  재생성 버퍼에 부착 (19718자)` → `:12988 [sim=task_074#s373753] [T2_ACTION_SUB] 발화를 격리에서
  지음 (손님 발화 4건 · 값 431자 · 표기 O)`.
- `_t2_asub` 는 `:8664` 이후 **해제되지 않는다**(★전수 grep = `8664`·`8667`·`10359`·`10360`).
- `T2_ACTION_SUB=1` 은 PIN 에 박혀 있다(`run_t7346_overnight_stage1_20260822.sh:84`).

### 2.2 ★이번에 뒤집힌 전제 — "버려진다"가 아니다

문서 머리에 적은 대로다. `work` 는 턴당 한 번 지어지고(`:6814`) 루프 뒤 여덟 자리가 그것을
그대로 모델에 보낸다. ★t7346 우회 11건 전부에서 직후에 `[T2_CLAIMPROV] window hit`(그 문구는
`:11736 _gen(self, work + [am, …], "agent_claimprov")` **뒤**의 `:11754`/`:11841` 에서만 인쇄된다)
또는 `[T2_SELFDECL] declared`(`:11952 _gen(self, work + [am, _dp], "agent_selfdecl")` 뒤의
`:11960`)가 나온다. 중간에 루프-A 마커가 안 낀 **확정 8/11**, `[T2_SEARCH_ON_PROCEED]` 가 낀
근사 3/11(그 마커도 루프 안이라 같은 턴의 다음 iteration 일 가능성이 높지만 **확정 안 됨**).
`T2_SELF_DECLARATION=1` 은 `go_stack.sh:393` 에 export 돼 있다.

⇒ **정정된 결함 진술**: *ASUB 우회 턴에서 배달물은 커밋되는 발화의 생성기에 못 가고, 비커밋
감사 서브콜(claimprov·selfdecl)에만 간다. 그리고 슬롯은 소비됐다고 표시된다.*
⇒ 감사 §1-A6 의 *"≈32,940자 폐기"* 는 **자수 기준으로는 과장**이고, R4 의 `arrived=False` 낙인은
**사실과 반대**다(§3). 결함 자체는 남는다 — 재료가 **행동을 짓는 자리**에 못 갔다.

### 2.3 반증이 바꾼 것 (원안 → 채택안)

| 원안(HUNK 1~3: `_cp2_in_work` + 소비 표시 지연) | 반증 | 채택안 |
|---|---|---|
| `_cp2_in_work` 정확일치로 중복 부착 방지 | 큐 ON(`T2_CP2_QUEUE=1`)의 병합은 `_prev + "\n\n" + text`(`:4544`/`:4547`) = **초문자열**이라 가드를 통과 ⇒ 같은 문단이 한 프롬프트에 **두 번**(반증자가 실행 재현) | **폐기.** 부착 자체를 옮긴다 |
| 소비 표시만 `_gen` 직전으로 | 슬롯이 살아남으면 다음 회차 `_ctx_fits(work, _cp2)`(`:10291`)가 **이미 `work` 에 든 같은 배달물을 hist 로 다시 센다** ⇒ 거짓 `[T2_DOC_DELIVERY] skipped` + 살려 둔 배달물 파괴(반증자 실행 재현) | **폐기.** 이동하면 이 경로가 소멸한다 |
| *"손실 11건 → 0건"* | 우회 사건 11건 중 **10건은 그 sim 에서 다시는 깨끗한 부착이 없다**(반증자 실측) ⇒ 회수 상한 1/11 | *"확정 손실 11 → 미확정"*. 단 **R2 를 켜면** 같은 턴 회수 창이 생긴다(§2.5-3) |
| *"`work` 는 루프 뒤 생성기도 쓴다(:10389/:10412/:10515/:10651/:10780)"* | `:10651` 에 생성기 없음 | 실제 자리 정정: `:10389 :10412 :10515 :10780 :10911 :11005 :11628 :11736 :11952` |
| §5-[E] *"용량 초과 경로는 ASUB 여부와 무관하게 동일"* | 위 이중계수로 **거짓** | 이동본에서는 참이 된다. 검정 [F] 로 못 박는다 |
| *"모델에 간 적 없는 배달물"* | §2.2 (11/11 반례) | 결함 진술 자체를 정정 |

### 2.4 최종 패치 (축자 OLD / NEW)

#### R3-① 소비 블록을 **잘라낸다** — `t2_gate_patch.py:10274-10304`

**OLD**
```python
            # ★CP2 를 **이 턴의 재생성 버퍼**에 붙인다 (2026-08-12·C443 교정).
            #   초판은 비커밋 뷰 큐(`_t2_view_fb`)에 넣었는데 그 큐는 **다음 턴** `unified()`
            #   시작에서 소비된다 — 결정점도 write 도 **이 턴**이라 한 턴 늦었다. 계측이
            #   그것을 그대로 찍었다: `agent=decision_carry · arrived=False`(070·071 둘 다),
            #   그리고 행동도 같은 말을 했다 — 서브가 `Sky Blue` 를 냈는데 제출은 `Hunter
            #   Green`(값 없이 후보 명단만 받으면 메뉴가 된다·C440 동형).
            #   ⚠여전히 **비커밋**이다: `work` 는 생성-시점 버퍼이고 `state.messages` 가 아니다
            #     (C298 replay 불변식 유지).
            #   ⚠배타 체인 밖은 그대로다 — `fb` 뒤에 **따로** 붙지, 어느 tool_call 도 차지하지
            #     않는다(억제·경쟁 무관).
            _cp2 = getattr(self, "_t2_cp2_pending", None)
            # ★컨텍스트 가드 — **소비 지점 하나**에 둔다(2026-08-16·t7304·심사 권고: 대입 자리
            #   5곳을 한 가드로 덮으려면 여기여야 한다). 대용량(≥5k자)만 검사·보수 추정(자수/3)·
            #   초과면 **건너뛰고 기록**(축약·선별 0 — 엔진이 줄이면 [[62]]③). 소형 배달물은
            #   종전 그대로(ctl 바이트 불변). skip 수는 ⓔ 부작용 표에 계상된다.
            if _cp2 and len(_cp2) >= _CP2_GUARD_MIN:
                # 산식·보정 근거는 `_ctx_fits` 독스트링(2026-08-22 함수로 올림·거동 동일).
                _fit2, _hist = _ctx_fits(work, _cp2)
                if not _fit2:
                    print("[T2_DOC_DELIVERY] skipped: est %d+%d chars > cap"
                          % (_hist, len(_cp2)), file=_sys.stderr, flush=True)
                    self._t2_cp2_pending = None
                    _cp2 = None
            if _cp2:
                self._t2_cp2_pending = None
                try:
                    work = work + [UserMessage(role="user", content=_cp2)]
                except TypeError:
                    work = work + [UserMessage(content=_cp2)]
                print("[T2_DECISION_CARRY] 이 턴 재생성 버퍼에 부착 (%d자)" % len(_cp2),
                      file=_sys.stderr, flush=True)
```
**NEW**
```python
            # ★CP2 부착은 더 이상 이 자리에 없다 — `_am_sub` 분기 **뒤**로 내렸다(2026-08-23·R3).
            #   이유·실측은 옮긴 자리의 주석에 적었다. 이 사이 코드(`_pin_r` 블록)는 `work` 도
            #   슬롯도 읽지 않는다(★전수 확인: `:10305-10360` 에 `work`·`_t2_cp2` 참조 0건).
```

#### R3-② `_am_sub` 분기 뒤에 **붙여넣는다** — `t2_gate_patch.py:10360-10362`

**OLD**
```python
                _am_sub = _gen_action_sub(self, state, self._t2_asub)
            am = _am_sub or _gen(self, work, bw(), "agent_response_unified_regen",
                                 tool_choice="required" if force_required else None, pin=_pin_r)
```
**NEW**
```python
                _am_sub = _gen_action_sub(self, state, self._t2_asub)
            # ★R3 (2026-08-23·CP2 큐 감사 §5 R3 · t7346 실측 11건). 부착·소비를 **`_am_sub` 계산
            #   뒤로** 통째로 내렸다. `_am_sub` 가 서면 아래 `or` 의 단락 평가로 `_gen` 이 아예
            #   안 불리고, `_gen_action_sub`(`:6583`)는 `state.messages` 로 자기 문맥을 새로
            #   짓는다(`:6626`) — cp2 는 비커밋이라 거기 없다(`:10280` 자기 주석). 구판은 그런
            #   회차에도 슬롯을 비우고 `부착` 을 인쇄했다: `bank_t7346_halfB_20260822.log.gz`
            #   `task_074#s373753` 19,718자 부착 → 3줄 뒤 `[T2_ACTION_SUB] … 값 431자`.
            #   ⚠**"모델에 안 갔다"는 아니다**(2026-08-23 재검증이 이 전제를 뒤집었다). `work` 는
            #     턴당 한 번 지어지고(`:6814`) 루프 뒤 여덟 자리가 그대로 모델에 보낸다
            #     (`:10412 :10515 :10780 :10911 :11005 :11628 :11736 :11952`). 실제로 우회 11건
            #     **전부**에서 직후에 `[T2_CLAIMPROV] window hit`(=`:11736` 뒤) 또는
            #     `[T2_SELFDECL] declared`(=`:11952` 뒤)가 찍힌다(같은 턴 확정 8/11). 잃은 것은
            #     *배달* 이 아니라 **행동을 짓는 생성기에의 배달**이다. 그래서 여기서 고치는 것은
            #     하나 — 그 회차에 슬롯을 **소비 처리하지 않는다**. 손실이 아니라 유예다
            #     ([[55]] 로그 마크 ≠ 도달).
            #   ⚠`_cp2_in_work` 식 중복 가드는 쓰지 않는다: 큐 ON 의 병합은 `_prev + "\n\n" + text`
            #     (`:4544`)라 **초문자열**이 되어 정확일치 가드를 통과하고, 같은 문단이 한
            #     프롬프트에 두 번 실린다(2026-08-23 반증이 실행으로 재현). 부착 자체를 옮기면
            #     그 경로도, 살아남은 슬롯을 `_ctx_fits` 가 hist 로 **두 번 세는** 경로도 함께
            #     사라진다.
            #   ⚠stderr 순서가 바뀐다: `[T2_DECISION_CARRY] … 부착` 이 이제 `[T2_READ_ROUTINE]`·
            #     `[T2_ACTION_SUB]` **뒤**에 인쇄된다. "부착 직후 ASUB" 인접으로 우회를 세던
            #     census 는 이 커밋 이후 로그에서 0 을 낸다 — 그것이 수리의 증거다.
            _cp2 = None if _am_sub else getattr(self, "_t2_cp2_pending", None)
            # ★CP2 를 **이 턴의 재생성 버퍼**에 붙인다 (2026-08-12·C443 교정).
            #   초판은 비커밋 뷰 큐(`_t2_view_fb`)에 넣었는데 그 큐는 **다음 턴** `unified()`
            #   시작에서 소비된다 — 결정점도 write 도 **이 턴**이라 한 턴 늦었다. 계측이
            #   그것을 그대로 찍었다: `agent=decision_carry · arrived=False`(070·071 둘 다),
            #   그리고 행동도 같은 말을 했다 — 서브가 `Sky Blue` 를 냈는데 제출은 `Hunter
            #   Green`(값 없이 후보 명단만 받으면 메뉴가 된다·C440 동형).
            #   ⚠여전히 **비커밋**이다: `work` 는 생성-시점 버퍼이고 `state.messages` 가 아니다
            #     (C298 replay 불변식 유지).
            #   ⚠배타 체인 밖은 그대로다 — `fb` 뒤에 **따로** 붙지, 어느 tool_call 도 차지하지
            #     않는다(억제·경쟁 무관).
            # ★컨텍스트 가드 — **소비 지점 하나**에 둔다(2026-08-16·t7304·심사 권고: 대입 자리
            #   5곳을 한 가드로 덮으려면 여기여야 한다). 대용량(≥5k자)만 검사·보수 추정(자수/3)·
            #   초과면 **건너뛰고 기록**(축약·선별 0 — 엔진이 줄이면 [[62]]③). 소형 배달물은
            #   종전 그대로(ctl 바이트 불변). skip 수는 ⓔ 부작용 표에 계상된다.
            if _cp2 and len(_cp2) >= _CP2_GUARD_MIN:
                # 산식·보정 근거는 `_ctx_fits` 독스트링(2026-08-22 함수로 올림·거동 동일).
                _fit2, _hist = _ctx_fits(work, _cp2)
                if not _fit2:
                    print("[T2_DOC_DELIVERY] skipped: est %d+%d chars > cap"
                          % (_hist, len(_cp2)), file=_sys.stderr, flush=True)
                    self._t2_cp2_pending = None
                    _cp2 = None
            if _cp2:
                self._t2_cp2_pending = None
                try:
                    work = work + [UserMessage(role="user", content=_cp2)]
                except TypeError:
                    work = work + [UserMessage(content=_cp2)]
                print("[T2_DECISION_CARRY] 이 턴 재생성 버퍼에 부착 (%d자)" % len(_cp2),
                      file=_sys.stderr, flush=True)
            am = _am_sub or _gen(self, work, bw(), "agent_response_unified_regen",
                                 tool_choice="required" if force_required else None, pin=_pin_r)
```

> **왜 `_gen` 앞이고 뒤가 아닌가**: `_gen` 이 예외를 던지면 소비 표시가 안 돼 중복 배달이 생긴다.
> 앞뒤 어느 쪽이든 `work` 는 이미 완성돼 있어 모델 입력은 동일하다.
> **`:10295` 의 `None` 은 건드리지 않는다** — 의도된 폐기고 `_cp2 = None` 을 함께 세운다.

### 2.5 거동 델타 ([[70]])

**달라지지 않는 것 (조건 명시)**: `_am_sub` 가 서지 않는 **모든 회차** = `T2_ACTION_SUB != "1"`
이거나 `rw_fb is None` 이거나 `force_required`·`_pin_r` 이 서거나 `self._t2_asub` 가 없는 회차에서
`_gen` 이 받는 `work` 는 **바이트 동일**. 근거: `:10305-10360`(`_pin_r` 블록 + ASUB 분기)에
`work` 도 `_t2_cp2_pending` 도 참조가 **0건**(★awk 전수). 슬롯 참조는 리포 전체에서
`t2_gate_patch.py:4489 4556 10284 10295 10298` 뿐.

**달라지는 것**
1. **`_am_sub` 가 서는 회차**(t7346 40 sim 에서 11회): 슬롯이 **안 비워진다**. 그리고 `work` 에
   **안 붙는다** ⇒ 루프 뒤 claimprov·selfdecl 이 그 재료를 **못 본다**(오늘은 11/11 이 본다).
2. **모든 부착 턴에서 stderr 줄 순서**가 바뀐다(`[T2_DECISION_CARRY] … 부착` 이
   `[T2_READ_ROUTINE]`·`[T2_ACTION_SUB]` 뒤로). 모델 입력은 무관.
3. **R2 와의 합성(★중요)**: R2 가 ON 이면 우회 회차 다음 iteration 에서 루프-A 가 다시 돌아
   `rw_fb` 가 대개 None 이 되고(`_am_sub` 조건이 `rw_fb` 를 요구한다) `am` 은 ASUB 산문이라
   `tool_calls == []` ⇒ **가드가 hold** → 같은 턴에서 `_gen` 이 cp2 를 실은 `work` 로 불린다.
   즉 **R3+R2(ON) 은 우회 구멍을 같은 턴에 닫고, 그 턴의 claimprov·selfdecl 도 재료를 되찾는다.**
   R2 가 OFF 면 R3 는 회수 창을 못 만든다(반증 실측: 11건 중 뒤에 깨끗한 부착이 있는 것 1건).
   ⚠이 합성 논증은 **코드 형세**에서 나온 것이고 라이브로 안 쟀다(**확인 안 됨**).

**판 것**
1. **우회 턴의 비커밋 감사 서브콜 재료** — 오늘 11/11 이 받는다. claimprov 산출(JSON claims)은
   `_unbacked` 판정을 거쳐 되먹임을 만들 수 있으므로 이것은 로그 순서 문제가 아니라 **거동 경로**다.
   크기는 **미측정**(확인 안 됨).
2. **늦은 배달.** 재료가 결정점보다 한 회차/한 턴 늦게 도착한다. C578(순서·매몰) 효과가 이 깊이에서
   어떤지는 미측정이므로 성적 주장으로 쓰지 마라.
3. **슬롯 점유 시간** — 살아남은 배달물이 다음 대입에 덮일 창이 생긴다. 오늘 조용히 사라지던 것이
   `[T2_CP2_CLOBBER]` 로 **보이게** 사라질 수 있다(순손실 증가 아님 · 가시성 상승). ⚠R1(큐)과 같은
   런에서 켜면 클로버 계수의 귀속이 섞인다.
4. **다음 턴 프롬프트 부피** — t7346 기준 최대 ≈33KB(2 sim 이 각각 19.7KB·11.2KB 단발). 지연 ·
   `ContextWindowExceededError` · `[T2_DOC_DELIVERY] skipped` 를 계상해야 한다.

**⛔ 하지 않는 것**: `_gen_action_sub` 의 `_work` 에 cp2 를 넣지 않는다(감사 지시대로 별건).
선택·순위·요약 0([[62]]).

### 2.6 검정

**새 파일 `scripts/distill/tau2/test_cp2_asub_bypass.py`** — 소스 문자열이 아니라 **함수를 실행**한다
(`test_cp2_queue_behavior.py` 계열). 실소스에서 `^ {12}_am_sub = None$` ~ `am = _am_sub or _gen(...)`
구간을 정규식으로 잘라 `textwrap.dedent` 후 `exec`. 스텁 = `_gen`(호출 기록) ·
`_gen_action_sub`(반환값 주입) · `_ctx_fits` · `UserMessage` · 가짜 `self`. `T2_ACTION_SUB=1`.

- **[A] 비-ASUB = 종전** — `work` 에 배달물 정확히 1개 · 슬롯 `None` · `_gen` 이 받은 messages 에
  그 문자열 포함 · `[T2_DECISION_CARRY]` 1줄.
- **[B] ASUB 발화** — `_gen` 0회 · `_gen_action_sub` 1회 · **슬롯이 원문 그대로 살아 있다** ·
  **`work` 에 배달물이 없다** · `[T2_DECISION_CARRY]` **0줄**. ← 구판이 FAIL 하는 줄.
- **[C] 같은 턴 다음 회차(비-ASUB)** — [B] 의 `work` 를 물려 1회 더 실행 → `work` 안 배달물
  **정확히 1개** · 슬롯 비움. **`_cp2_assign` 을 함께 돌려 큐 ON(병합 초문자열) 조합에서도 1개**
  여야 한다 ← 원안이 FAIL 하던 줄.
- **[D] pending 없음** — `work` 불변 · `_gen` 정상 1회.
- **[E] 용량 초과** — `_ctx_fits` 가 `(False, …)` → `[T2_DOC_DELIVERY] skipped` + 슬롯 None +
  부착 0. **ASUB 회차에서는 이 검사 자체가 안 돈다**(`_cp2 is None`)를 함께 단언.
- **[F] `_ctx_fits` 이중계수 부재** — 히스토리 62,000자 + 19,718자 배달물로 회차 A(ASUB) →
  회차 B(비-ASUB)를 돌려 B 의 `_ctx_fits` hist 가 **A 의 부착분을 포함하지 않음**을 단언. 스텁을
  실제 합산식(`sum(len(content) for m in work)`)으로 둔다 — 원안 하네스는 `(True, 0)` 스텁이라
  원리상 이것을 못 봤다.
- **[G] 술어 동형(AST)** — `am = _am_sub or _gen(...)` 직전의 `_cp2 = …` 가 `_am_sub` 를 술어로
  쓰는지, `self._t2_cp2_pending = None` 이 파일 전체에 **2곳**(ctx 폐기 1 + 부착 1)뿐인지.

**기존 회귀(★내가 HEAD 에서 초록 확인 — 패치 뒤 전부 재실행)**: `test_route_trace.py` **22/22** ·
`test_regen_break_guard.py` ALL PASS · `test_cp2_queue_behavior.py` PASS · `test_cp2_clobber.py` PASS ·
`test_decision_carry.py` **24/24** · `test_proceed_docbody.py` PASS.

### 2.7 이 패치가 틀릴 수 있는 두 가지

**⑴ 판 것이 산 것보다 클 수 있다.** R2 가 OFF 인 채로 R3 만 내면, 오늘 11/11 이 받던 claimprov·
selfdecl 재료가 사라지고 회수는 1/11 이다. 즉 **R3 단독의 순효과가 음일 수 있다.** R3 는 R2 를 켠
상태에서 재는 것이 맞다 — 그 결정은 §0 의 2번 항목이다.

**⑵ `_t2_asub` 영구 래치.** `:8664` 이후 해제가 없어 sim 후반이 통째로 ASUB 턴이 되면 슬롯이 계속
이월돼 클로버·부피 압력이 예상보다 클 수 있다. 그건 R3 가 아니라 별건 결함이고 **미측정**이다.

---

## §3 R4 — 팔-대칭 배달 계기 (모델 가시 바이트 불변)

### 3.1 지금 무엇이 참인가 (축자)

- `arrived` 판정은 `_gen` 안 `t2_gate_patch.py:6455-6473` 에만 있다. cp2 를 그 큐에 등재하는 코드는
  **`:9101-9107`(VIEW_FB) 하나뿐**. 나머지 네 배달 자리(`:6735 :8123 :8906 :8943`)는 등재 0.
- `record_many(fb, …)` 는 `:10270`, cp2 부착은 `:10300` ⇒ **cp2 본문은 어느 사이드카 행에도 없다**.
- ★보관 사이드카 전수(`t7307`~`t7328`, 14파일)에서 `agent="decision_carry"` 행의 `arrived` 가
  **100% True**(총 303행 · False 0). 그리고 그 행 수는 도달 수가 아니라 **`VIEW_FB` 대입 수와
  1:1** 이다(t7326 halfA 35=35 · halfB 34=34 · t7328 halfB 32=32). ⇒ 지금 계기는 **한 배달 자리의
  발화 횟수를 되찍고 있다** — C502 가 무효 판정한 *"처치 배정의 재인쇄"* 와 같은 형태이고, 값이
  항상 True 라 **두 팔을 구분할 힘이 0**이다.
- `t7336`·`t7346` 의 `.fb.jsonl.gz` 는 `sim_results/` 에 **없다**(★디렉터리 실사 — 마지막은
  t7328). 러너 `run_t7346_overnight_stage1_20260822.sh:208-209` 가 results/log 만 회수한다.
- `t2_liveness.py:115-140` 의 `delivery()` 는 `k = str(o.get("agent") or o.get("mark") or "?")` 로
  버킷한다 ⇒ `agent` 없는 새 행은 전부 `"?"` 채널로 간다.

### 3.2 반증이 바꾼 것 (원안 → 채택안) — **원안 8패치 중 3개 폐기 · 4개 수정**

| 원안 | 반증/재검증 | 채택안 |
|---|---|---|
| **P4** `_gen` 훅을 `_route_drain` 으로 교체 | `test_route_trace.py:85-87` 축자가 `re.search(r"def _gen\(.*?_rec\[\"arrived\"\] = bool\(_txt and _txt in _hay\)", SRC, re.S)` 를 요구한다. 리터럴이 `_gen` 밖으로 나가면 **FAIL** 하고 러너는 VERIFY 실패 시 `exit 1` 한다 ⇒ **런 거절** | **폐기.** `_gen` 훅은 손대지 않는다 |
| **P5** VIEW_FB 등재 제거 | 제거하면 route 채널의 `decision_carry` 행이 사라져 `t2_liveness.delivery()` 의 과거 비교가 끊긴다. 그리고 P4 를 폐기하면 옮길 곳이 없다 | **폐기.** `:9094-9107` 은 **바이트 그대로 둔다** |
| **P7** ASUB 자리에서 `_route_drain(self, None, …)` = `arrived=False` 낙인 | ⓐ `work` 는 버려지지 않는다(루프 뒤 여덟 자리) ⓑ★t7346 우회 11건 **전부**에서 직후 claimprov/selfdecl 이 그 `work` 를 모델에 보낸다 ⇒ 낙인은 **사실과 반대** ⓒ `_t2_route_pending` 은 **공유 큐**다(`:10129`·`:10155` 배타 체인 계측도 들어온다) — ASUB 조건이 `rw_fb` 참을 요구하므로 그 턴엔 `resolve_write` 행이 반드시 있고 그것들까지 오염된다 ⓓ R3 적용 뒤에는 우회 회차에 부착 자체가 없어 낙인할 대상도 없다 | **폐기** |
| **P0** `current_turn()` 신설 근거 = *"읽는 문이 없어서"* | `t2_lever_beat.py:130` 이 이미 `getattr(_LOCAL, "turn", None)` 을 읽는다 | 근거 수정 + `:130` 을 새 접근자로 리팩터(사본 금지·[[67]]) |
| **P3** `if _incoming:` 무조건 개방 | 자기 주석(*"`_prev == text` 재대입은 새 배달이 아니다"*)과 **모순**. `PRECOMMIT`(`:6735`)에는 `_t2_cp2_said` 가드가 없어 도달 경로도 있다(지금은 `T2_DELIVER_PRECOMMIT=0` 이라 死) | `if _incoming and _incoming != _prev:` |
| **P8** 러너 회수 | `"$LOG/fb_$TAG.jsonl"` 의 `$TAG` 를 escape 안 해 로컬에서 빈 문자열로 전개 ⇒ **매번 회수 0**. 그리고 *"없으면 종점 계산 불가"* 는 과장(총계는 감사·명세·반증·나 넷이 로그만으로 냈다) | **escape 수정**(`\$TAG`) + "런 거절" → **경고**로 강등 |
| 1차 종점 = `arrived` | R3 이후 부착 지점은 `_gen` 호출 **바로 앞**이라 `arrived` 는 **동어반복**(부착 = 도달). 그러면 "부착 인쇄의 재인쇄" 가 된다 | **cp2 행에 `arrived` 를 쓰지 않는다.** 종점은 `assign` ↔ `close(attached/clobbered/ctx_skip)` 의 **닫힌 분할** |
| 채널 | `kind="cp2"` 행은 `agent`·`mark` 가 없어 `t2_liveness` 의 `"?"` 버킷에 쌓인다 | 행에 `agent="cp2"` 를 넣어 기존 `decision_carry` 채널과 **분리** |
| `slot_bytes` 키 `(cp2_attach_turn, cp2_slot_n)` | `current_turn()` 은 턴당 1회만 심어져 한 턴의 여러 부착이 같은 값 + 247/254자 반복으로 충돌 실재 | 키를 **`cp2_id`** 로 |

### 3.3 최종 패치 (축자 OLD / NEW) — **R3·R2 적용 뒤 파일 기준**

#### R4-ⓐ `scripts/distill/tau2/t2_lever_beat.py:33-35` 뒤

**OLD**
```python
def current_sim():
    """이 스레드가 지금 돌리고 있는 sim id(모르면 None). 태그를 붙이는 모든 자리의 단일 출처."""
    return getattr(_LOCAL, "sim", None)
```
**NEW**
```python
def current_sim():
    """이 스레드가 지금 돌리고 있는 sim id(모르면 None). 태그를 붙이는 모든 자리의 단일 출처."""
    return getattr(_LOCAL, "sim", None)


# ★턴 읽기 (2026-08-23·R4). `set_turn` 이 심는 값을 읽는 자리는 이미 있었다 — 아래 `_trace` 가
#   `getattr(_LOCAL, "turn", None)` 을 인라인으로 읽는다. 그런데 **공개 접근자가 없어서** 다른
#   계기(cp2 생애 원장)가 같은 축에 올라오려면 사본을 하나 더 만들어야 했다. 사본은 조용히
#   갈라진다([[67]]) — 읽는 자리를 하나로 모은다. `_trace` 도 이 함수를 쓰도록 함께 고친다.
def current_turn():
    """이 스레드가 지금 짓고 있는 턴(모르면 None). `set_turn` 이 심은 값의 단일 출처."""
    return getattr(_LOCAL, "turn", None)
```
그리고 `t2_lever_beat.py:130`:
**OLD** `                rows.append(json.dumps({"sim": sim, "turn": getattr(_LOCAL, "turn", None),`
**NEW** `                rows.append(json.dumps({"sim": sim, "turn": current_turn(),`

#### R4-ⓑ 헬퍼 2개 — `t2_gate_patch.py` `_CP2_HOLD_MAX` 정의 뒤 · `def _cp2_assign` 앞

**OLD**
```python
def _cp2_assign(self, text, tag):
```
**NEW**
```python
# ─── CP2 배달 생애 원장 (2026-08-23 · R4 · 원장 C502) ──────────────────────────
# ⛔**순환 종점 재발 금지.** t7303 A/B 가 무효가 된 이유는 결손이 아니라 *계기*였다 — 1차 종점이
#   `[T2_CP2_APPEND] … (queue)` 였는데 그 줄은 **플래그가 꺼진 팔에서는 존재할 수 없다**. 그래서
#   "0/8 → 8/8" 은 측정이 아니라 **처치 배정의 재인쇄**였다(C502 축자). ⇒ 아래 세 이벤트·세
#   outcome 어디에서도 `T2_CP2_QUEUE` 를 읽지 않는다. 검정 §F 가 소스로 그것을 강제한다.
# ⛔**계기가 이미 한 번 순환이었다**(2026-08-23 실측). 보관 사이드카 14파일 전수에서
#   `agent=decision_carry` 의 `arrived` 가 **100% True**(303행·False 0)인데, 그 행 수는 도달 수가
#   아니라 **`VIEW_FB` 대입 수와 정확히 1:1**이다(t7326 35=35·34=34 · t7328 32=32). 등재가
#   `VIEW_FB` 한 자리에만 있어서(t7346 대입 125 중 61 = SEARCH_ON_PROCEED 가 무계측), 결국
#   *한 배달 자리가 몇 번 발화했나*를 도달률이라고 불러 온 것이다([[25]] 우리 계기는 100% 정답
#   의무). ⇒ 등재를 **다섯 자리 공통 입구**(`_cp2_assign`)로 옮긴다.
# ⚠**`arrived` 를 여기서 쓰지 않는다.** R3 이후 부착 지점은 `_gen` 호출 바로 앞이라 "부착됐다"와
#   "모델 입력에 있다"가 같은 말이 된다 — 그런 값을 `arrived` 라 부르면 이번엔 **부착 인쇄의
#   재인쇄**가 된다. 대신 배달물 하나의 생애를 **닫힌 분할**로 적는다:
#       assign → close(attached) | close(clobbered) | close(ctx_skip) | (미종결 = sim 종료 시 잔존)
#   그 분할이 닫혀 있어야 `대입 = 도달 + 손실` 검산식이 서고, 검산식이 서야 감사 스크립트가
#   *"이 팔에서 몇 건이 어디서 죽었나"* 를 **양 팔 같은 규칙으로** 낼 수 있다.
# ⚠거동 불변 계약: 이 블록은 `_t2_cp2_track` · `_t2_cp2_seq` 두 속성과 사이드카 파일 **밖으로
#   나가지 않는다**. `work`·`fb`·`state.messages`·`_t2_cp2_pending`·`_t2_cp2_said` 에 대입하는
#   문장이 하나도 없고, 정상 경로에서 stderr 한 줄도 늘지 않는다(검정 §E 가 AST 로 대조).
# ⚠[[62]]: 고르는 것이 0 이다 — 순위·최댓값·지목 없이 *무엇이 어디까지 갔나*만 센다.
def _cp2_open(self, text, tag, disp):
    """배달물 1건을 **미결(open)** 로 열고 `assign` 행을 즉시 남긴다.

    ⚠`assign` 행이 **분모**다. 이 행이 없으면 끝내 미소비로 죽은 배달물(t7346 12건 26,698자)이
      사이드카에 흔적 0 이 되고, 그러면 도달률의 분모를 stderr grep 으로 세게 된다 — 감사서가
      기록한 grep 함정(`부착 (N자)` 만으로 세면 `T2_REQUIRE_DOC_DELIVER` 채널이 섞여 125가
      116으로 부푼다)이 바로 그 자리다.
    ⚠도달 판정은 여기서 **하지 않는다** — 여기서 채우면 `_cp2_assign` 호출을 배달로 위조한다.
    """
    try:
        _n = getattr(self, "_t2_cp2_seq", 0) + 1
        self._t2_cp2_seq = _n
        try:
            import t2_lever_beat as _lb0
            _sim, _turn = (_lb0.current_sim() or "nosim"), _lb0.current_turn()
        except Exception:
            _sim, _turn = "nosim", None
        _rec = {"agent": "cp2", "cp2_id": "%s#%d" % (_sim, _n), "cp2_tag": str(tag),
                "cp2_n": len(text or ""), "cp2_disp": str(disp), "turn": _turn}
        _tr = list(getattr(self, "_t2_cp2_track", None) or [])
        _tr.append(dict(_rec, _text=text or ""))
        self._t2_cp2_track = _tr
        import t2_fbsidecar as _fbo
        _fbo.record("cp2", text or "", None, ev="assign", **_rec)
    except Exception as _eo:
        print("[T2_CP2_TRACK] open 실패(무시): %r" % (_eo,), file=sys.stderr, flush=True)


def _cp2_close(self, outcome, slot_n=None):
    """슬롯에 열려 있던 배달물 **전부**를 `outcome` 으로 종결한다.

    슬롯이 병합본이면 그 안의 조각이 여럿이므로 전부 닫는다 — 하나만 닫으면 나머지가 영원히
    미결로 남아 검산식이 깨진다. 그리고 **배달물 단위**로 닫는다(부착 단위가 아니라): 병합본
    1회 부착에 조각이 둘이면 행도 둘이다. 부착 단위로 세면 큐 ON 이 2건을 1건으로 접어 도달률이
    구조적으로 낮게 나오고, 그 순간 두 팔의 분모 정의가 달라져 A/B 가 다시 무효가 된다.
    """
    try:
        _tr = getattr(self, "_t2_cp2_track", None) or []
        self._t2_cp2_track = []
        if not _tr:
            return
        try:
            import t2_lever_beat as _lb1
            _turn = _lb1.current_turn()
        except Exception:
            _turn = None
        import t2_fbsidecar as _fbc
        for _r in _tr:
            _t = _r.pop("_text", "") or ""
            _r["ev"] = "close"
            _r["outcome"] = str(outcome)
            _r["cp2_close_turn"] = _turn
            if slot_n is not None:
                _r["cp2_slot_n"] = int(slot_n)
            _fbc.record("cp2", _t, None, **_r)
    except Exception as _ec:
        print("[T2_CP2_TRACK] close(%s) 실패(무시): %r" % (outcome, _ec),
              file=sys.stderr, flush=True)


def _cp2_assign(self, text, tag):
```

> `sys` 가 맞다 — 이 함수들은 **모듈 레벨**이라 `_sys`(함수 안 정의)를 쓰면 NameError 다.
> `_cp2_assign` 이 같은 이유로 `sys.stderr` 를 쓴다(`:4541` 등 · 그 파일 `:4529-4530` 자기 주석
> 축자: *"⛔`sys` 다(2026-08-18·C538). 이 함수도 **모듈 레벨**"*).

#### R4-ⓒ 대입 시점 원본 포착 — `t2_gate_patch.py:4489`

**OLD**
```python
    _prev = getattr(self, "_t2_cp2_pending", None)
```
**NEW**
```python
    _prev = getattr(self, "_t2_cp2_pending", None)
    # ★병합 분기가 `text` 를 `_prev + "\n\n" + text` 로 **덮어쓰기 때문에**, 계기가 기록할
    #   *이번 배달물* 원본을 여기서 잡아 둔다. 병합 후 값을 기록하면 조각 하나가 두 번 세어진다.
    _incoming = text
```

#### R4-ⓓ 생애 등록 — `t2_gate_patch.py:4556`

**OLD**
```python
    self._t2_cp2_pending = text
```
**NEW**
```python
    # ★생애 등록 (R4). 배달물 하나는 `attached · clobbered · ctx_skip` 중 정확히 하나로 끝나거나
    #   sim 종료까지 미결로 남는다(=잔존). 그 분할이 닫혀 있어야 검산식이 선다. 세 라벨 어느
    #   것도 `_queue` 를 보지 않는다 — 그것이 팔-대칭의 전부다(C502 가 무너진 자리).
    # ⚠빈 배달물은 배달이 아니다(열지 않는다). `_prev == _incoming` 재대입도 새 배달이 아니다 —
    #   슬롯 내용이 그대로라 이미 열린 건이 계속 유효하다. (원안은 `if _incoming:` 이라 같은
    #   바이트를 두 번 열어 분모·분자를 동시에 부풀렸다 — 2026-08-23 반증이 잡았다.)
    if _prev and _prev != _incoming and not (_big or _qok):
        _cp2_close(self, "clobbered")          # 앞 건은 여기서 죽는다(양 팔 같은 규칙)
    if _incoming and _incoming != _prev:
        _cp2_open(self, _incoming, tag,
                  "append" if (_big or _qok) else ("clobber" if _prev else "fresh"))
    self._t2_cp2_pending = text
```

#### R4-ⓔ `ctx_skip` 종결 — **R3 이동본 안**

**OLD**(R3 적용 뒤)
```python
                    print("[T2_DOC_DELIVERY] skipped: est %d+%d chars > cap"
                          % (_hist, len(_cp2)), file=_sys.stderr, flush=True)
                    self._t2_cp2_pending = None
                    _cp2 = None
```
**NEW**
```python
                    print("[T2_DOC_DELIVERY] skipped: est %d+%d chars > cap"
                          % (_hist, len(_cp2)), file=_sys.stderr, flush=True)
                    self._t2_cp2_pending = None
                    # ★R4: 창 초과로 여기서 죽는다 — 이 사실이 사이드카에 안 남으면 그 배달물은
                    #   *대입은 됐는데 아무 데도 없는* 유령이 되고, 검산식이 그 자리에서 깨진다.
                    _cp2_close(self, "ctx_skip")
                    _cp2 = None
```

#### R4-ⓕ `attached` 종결 — **`_gen` 이 실제로 돌아온 뒤에만**

**OLD**(R3 + R2 적용 뒤)
```python
            am = _am_sub or _gen(self, work, bw(), "agent_response_unified_regen",
                                 tool_choice="required" if force_required else None, pin=_pin_r)
            if _cp2:
                # ★R2: 이 턴에 cp2 가 실제로 `work` 에 실려 생성기에 넘어갔다. 다음 iteration 의
                #   가드는 이 사실을 보고 **더는 붙잡지 않는다** — 두 번째 무라벨 결정문 금지.
                cp2_attached += 1
```
**NEW**
```python
            am = _am_sub or _gen(self, work, bw(), "agent_response_unified_regen",
                                 tool_choice="required" if force_required else None, pin=_pin_r)
            if _cp2:
                # ★R2: 이 턴에 cp2 가 실제로 `work` 에 실려 생성기에 넘어갔다. 다음 iteration 의
                #   가드는 이 사실을 보고 **더는 붙잡지 않는다** — 두 번째 무라벨 결정문 금지.
                cp2_attached += 1
                # ★R4: 종결은 **`_gen` 이 돌아온 뒤**에만 찍는다. 부착 인쇄(`부착 (N자)`)를 세면
                #   생성기가 예외로 죽은 회차까지 도달로 위조한다 — `proc_fb` 死배선이 deny 11회를
                #   인쇄로 만든 것과 같은 종류의 사고다([[55]] 로그 마크 ≠ 전달).
                _cp2_close(self, "attached", slot_n=len(_cp2))
```

#### R4-ⓖ 러너 — 사이드카 회수 (A/B 러너 신규 파일. 양 팔 · smoke 각각)

`run_t7346_overnight_stage1_20260822.sh:208-209` 와 같은 자리:

**OLD**
```sh
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
```
**NEW**
```sh
    gzip -c '$SIMS/'\$TAG'/results.json' > reports/facet_rft_2026/sim_results/\$TAG.results.json.gz
    gzip -c $LOG/\$TAG.log > reports/facet_rft_2026/sim_results/\$TAG.log.gz
    # ★R4 (2026-08-23). go_stack.sh:222 가 사이드카를 항상 켜는데 **회수하는 문장이 없었다** —
    #   t7336·t7346 의 .fb.jsonl.gz 가 sim_results 에 없는 이유다(마지막은 t7328). 조각 단위
    #   생애 원장은 여기에만 있다(건수·자수 총계는 로그로도 나온다 — 그래서 경고이지 거절은 아니다).
    #   ⚠`\$TAG` 는 **원격에서** 전개돼야 한다(`$LOG` 는 로컬 전개) — 이 escape 를 빠뜨리면
    #     로컬에서 빈 문자열이 되어 `fb_.jsonl` 을 찾고 매번 조용히 회수 0 이 된다(원안 결함).
    [ -s "$LOG/fb_\$TAG.jsonl" ] \
      && gzip -c "$LOG/fb_\$TAG.jsonl" > reports/facet_rft_2026/sim_results/\$TAG.fb.jsonl.gz \
      || echo "[R4] WARNING: 사이드카 없음 — 조각 단위 검산 불가: $LOG/fb_\$TAG.jsonl" >&2
```

### 3.4 **모델 가시 바이트 불변 — 어떻게 보증하는가** (요구 항목)

1. **대입 대상이 두 속성뿐.** 신규 코드가 대입하는 이름은 `self._t2_cp2_track` ·
   `self._t2_cp2_seq` · 지역 `_incoming`/`_rec`/`_tr`/`_n`/`_sim`/`_turn`/`_t`/`_r` 뿐이다.
   `work`·`fb`·`kw`·`am`·`state.messages`·`self._t2_cp2_pending`·`self._t2_cp2_said`·`text` 에
   대입하는 문장이 **0개**다(R4-ⓓ 는 기존 `self._t2_cp2_pending = text` 줄을 **그대로 둔다**).
   검정 §E 가 AST 로 대조한다.
2. **`_t2_route_pending` 을 건드리지 않는다.** P4·P5·P7 을 전부 폐기했으므로 route 채널 · `_gen`
   훅 · VIEW_FB 등재는 **바이트 그대로**다 ⇒ `test_route_trace.py` 22/22 유지, `t2_liveness` 의
   `decision_carry` 채널 과거 비교 유지, 공유 큐(`:10129`·`:10155` 배타 체인 계측) 오염 0.
3. **`t2_fbsidecar` 는 궤적을 만지지 않는다** — 모듈 독스트링 설계 제약 ① 축자 *"이 모듈은
   messages를 절대 만지지 않는다(replay 위생 유지)"*. 그리고 `T2_FB_SIDECAR` 미설정이면
   `record()` 가 즉시 return = 완전 no-op(★`t2_fbsidecar.py:60` `path =
   os.environ.get("T2_FB_SIDECAR")` / `if not path: return`).
4. **`record(..., messages=None, turn=…)` 이 깨지지 않는다.** `record()` 는 `row["turn"] =
   len(messages or [])` 를 **먼저** 넣고 그 뒤 `for k, v in meta.items(): row[k] = v` 로 덮는다 ⇒
   중복 kwarg 가 아니라 정상 덮어쓰기다. `_sim_key(None)` 은 `"nouser"` 이고 조인은 `simtag`
   (`current_sim()`)로 한다. ★코드 축자 확인.
5. **stderr 가 모델 입력이 아니다.** 신규 `print` 는 전부 `except` 안이며 정상 경로에서는 **한 줄도
   늘지 않는다** — 기존 로그 grep(census · 스모크 게이트 문구)도 그대로 산다.
6. **예외가 새 경로를 만들지 않는다.** 두 헬퍼 전부 몸통이 `try/except Exception` 이고 예외를
   다시 올리지 않는다(검정 §E).
7. **행동 검산**: 검정 §A 가 `T2_FB_SIDECAR` 유/무 두 조건에서 같은 시퀀스를 실행해
   `_t2_cp2_pending` 바이트와 stderr 전문이 **byte-identical** 임을 실행으로 확인한다.

### 3.5 거동 델타 ([[70]])

**달라지지 않는 것**: 모델 입력 바이트 0 변화 · 정상 경로 stderr 0줄 변화 · route 채널 0 변화 ·
`reward`·궤적·replay 무관(cp2 는 여전히 비커밋).

**달라지는 것**

| # | 조건 | 구판 | 신판 |
|---|---|---|---|
| ① | `T2_FB_SIDECAR` 가 설정된 모든 런 | `kind="cp2"` 행 0 | 비지 않은 **새** 배달물마다 `ev="assign"` 행 1개(t7346 규모 = 40 sim 당 약 125행) |
| ② | 클로버·ctx_skip·부착 | 행 0 | `ev="close"` 행 1개/조각(약 110행) |
| ③ | `t2_liveness.delivery()` 표 | 채널 없음 | **새 채널 `cp2`**. 기존 `decision_carry` 는 불변 |
| ④ | `t2_lever_beat._trace` | 인라인 `getattr` | `current_turn()` 호출(값 동일) |
| ⑤ | 회수 산출물 | `<tag>.results.json.gz`·`<tag>.log.gz` | + `<tag>.fb.jsonl.gz` |

**판 것**
1. **사이드카 부피·쓰기 횟수** — 배달물당 2행 ⇒ 40 sim 당 약 +235행. `GO_CONCURRENCY=1` 이고
   halfA/halfB 는 별 프로세스라 `_LOCK` 경합 0. 지연 기여는 **미측정** — Stage 1 에서 duration 을
   계상한다.
2. **`assign` 행의 `text` 는 4,000자 프리픽스다**(`t2_fbsidecar.record` `s[:4000]`). `len`·`sha` 는
   전문 기준이라 ⓐ건수 ⓑ자수 ⓒ도달률 세 종점은 무손실이고, 손실은 *문면 정독*에만 있다(50k 문서
   본문 배달의 경우). `record()` 를 고치면 **모든 채널**의 행 크기가 함께 바뀌므로 감수한다.
3. **새로 예외를 낼 수 있는 코드 경로 2개.** 조용히 실패하면 계기가 부분적으로 비고 검산식이
   깨진다 — 그래서 감사 스크립트가 검산 위반을 경고가 아니라 **exit 2** 로 다룬다([[25]]).
4. **`t2_liveness` 표에 새 줄이 하나 는다.** 사람이 읽는 배선-생존 표의 모양이 바뀐다(오염은 아님).

### 3.6 검정

**새 파일 `scripts/distill/tau2/test_cp2_arrival_instrument.py`**(실행형 · 오프라인 · 모델 0).
가짜 `self`, 가짜 `UserMessage`, 임시 `T2_FB_SIDECAR` 파일. **소스 문자열 대조가 아니라 함수를
돌려서** 잰다(`test_cp2_queue_behavior.py` 방식).

- **§A 바이트 불변(행동)** — `(prev, text)` 격자 × `T2_CP2_QUEUE ∈ {unset,0,1}` ×
  `T2_FB_SIDECAR ∈ {unset, tmpfile}` 조합에서 `_cp2_assign` 실행 후 `_t2_cp2_pending` 바이트와
  stderr 전문이 **사이드카 유/무 간 완전 동일**. 미설정에서 파일 생성 0 · 예외 0 · 신규 stderr 0.
- **§B 생애 분할이 닫혀 있다(핵심 · 팔-대칭 주장 그 자체)**
  - QUEUE=0: `assign(A,247)` → `assign(B,254)` ⇒ 행 = `assign(A)`, `close(A,clobbered)`,
    `assign(B)`. `A.cp2_id` 는 `attached` 로 절대 안 닫힌다.
  - QUEUE=1: 같은 시퀀스 ⇒ `assign(A)`, `assign(B)`(close 없음). 이어서
    `_cp2_close("attached", slot_n)` ⇒ **A·B 둘 다 close(attached)**.
  - ⇒ 단일 규칙으로 OFF = 1/2, ON = 2/2. **분모·분자가 두 팔 모두에 존재**한다(구판은 OFF 에서
    분모가 정의 불가).
  - 검산식 `assign == attached + clobbered + ctx_skip + 미종결` 을 각 시나리오에서 assert.
- **§C 재대입은 새 배달이 아니다** — `assign(A)` → `_cp2_assign(same A)` ⇒ 새 `assign` 0 ·
  `close` 0. (원안 `if _incoming:` 이 여기서 FAIL 한다.)
- **§D 다섯 자리 전부가 같은 필드로 온다** — 5개 태그로 각각 호출 → `assign` 행의 키 집합이
  **완전히 동일**. `SEARCH_ON_PROCEED` 로 넣은 건이 종결까지 나온다(구판이면 0행).
- **§E 거동 불변을 코드 형세로 강제(AST)** — 두 헬퍼의 AST 에 `work`·`fb`·`kw`·`am`·`state`·
  `_t2_cp2_pending`·`_t2_cp2_said` 로의 대입 **0건** · 몸통 최상위가 `Try` 이고 `raise` 없음 ·
  `t2_gate_patch.py` 전체에서 `self._t2_cp2_pending` 에 대입하는 자리가 정확히 **3곳**이며 각각에
  대응하는 계기 호출이 인접(네 번째가 생기면 원장과 슬롯이 조용히 갈린다).
- **§F 팔-대칭 형식 보증(반-C502)** — 두 헬퍼 소스에 `T2_CP2_QUEUE`·`_queue`·`"(queue)"` 문자열
  **0건**. 종점이 한쪽 팔에서만 존재할 수 있는 술어에 걸리면 FAIL.
- **§G 채널 분리** — `assign`/`close` 행의 `agent` 가 `"cp2"` 이고 `"decision_carry"` 가 **아니다**
  (`t2_liveness` 의 기존 채널을 오염시키지 않는다) · `agent` 키가 항상 있다(`"?"` 버킷 금지).
- **§H `arrived` 부재** — cp2 행에 `arrived` 키가 **없다**. 있으면 부착 인쇄의 재인쇄가 된다.

**배터리 등재**: A/B 러너의 VERIFY 목록(`run_t7346_…:55-66` 형식)에
`test_cp2_arrival_instrument.py` 와 `test_cp2_asub_bypass.py` 추가. 실패 시 런 거절.

### 3.7 이 패치가 틀릴 수 있는 두 가지

**⑴ 같은 본문이 슬롯 안에 두 번 열리면 분할이 부풀고, 그 부풀기가 두 팔에서 다르다.** 서로 다른
자리가 **같은 문자열**을 배달하면 큐 ON 은 병합본에 조각 2개(둘 다 attached), OFF 는
`_prev == _incoming` 분기로 아예 안 열려 분모가 1이다. 완화: `assign` 행이 `sha` 를 지니므로 감사
스크립트가 **"배달물 수"와 "고유 본문 수"를 둘 다** 인쇄하고 갈리면 경고한다. 실측상 위험은
낮다(각 자리가 `_t2_cp2_said` 로 직전 문자열을 막는다) — 그러나 그 가드는 자리마다 따로라
**원리적으로 닫혀 있지 않고**, `PRECOMMIT`(`:6735`)에는 아예 없다.

**⑵ `_t2_cp2_track` 이 `_t2_cp2_pending` 과 조용히 어긋난다.** 나중에 네 번째 슬롯 대입이 생기면
track 은 죽은 본문을 들고 있게 되고 종결이 엉뚱한 조각에 붙는다 — 그러면 "우리 계기는 100%
정답"([[25]])이 깨진 채로 유일한 근거원이 오염된다. 완화 = 검정 §E 가 대입 자리 수를 AST 로 못
박는다. 또한 `_cp2_close` 는 예외가 나면 track 을 비운 채로 남길 수 있어
(`self._t2_cp2_track = []` 가 `for` 앞에 있다) 그 배달물들이 **미종결**로 빠진다 — 감사 스크립트가
이것을 조용히 흡수하지 않고 `unresolved` 칸에 드러내므로 오판이 아니라 미지로 남는다.
`unresolved` 가 크면 결론을 내지 말고 원인을 먼저 찾아야 한다.

---

## §4 반증에서 살아남지 못한 것

| # | 어느 명세의 무엇 | 왜 기각/수정 | 이 계획의 처리 |
|---|---|---|---|
| 1 | **세 명세 공통 전제** *"ASUB 우회 턴의 `work` 는 버려진다 = 모델에 간 적 없다"*(감사 §1-A6 · R3 §1.3 · R4 §1.3) | `work` 는 턴당 한 번 지어지고(`:6814`) 덧붙이기만 하며 루프 뒤 여덟 자리가 그대로 모델에 보낸다. ★t7346 우회 11건 전부에서 직후 `[T2_CLAIMPROV]`(`:11736` 뒤) / `[T2_SELFDECL]`(`:11952` 뒤) — 같은 턴 확정 8/11 | 결함 진술을 *"커밋 발화의 생성기에 못 간다"* 로 정정(§2.2). R4 P7 기각 |
| 2 | R2 *"다섯 배달 env 가 전부 OFF 면 바이트 동일 ⇒ 새 플래그 불필요"* | 다섯을 한 번에 끄는 env 가 없고 PIN 은 `T2_SEARCH_ON_PROCEED=1` | `T2_CP2_HOLD` 신설 · 기본 0 |
| 3 | R2 `cp2_rounds += 1` 을 가드 앞에서 무조건 | fb 가 선 iteration(`:8846` 순수-조언 `rw_fb`)이 예산을 태운다 | 증가를 가드 `if` **안**으로 |
| 4 | R2 `_CP2_HOLD_MAX = 2` | 두 번째 무라벨 결정문이 같은 프롬프트에(R6 영역·큐 A/B 와 같은 런 금지) | `_CP2_HOLD_MAX = 1` + `cp2_attached == 0` 이중 봉인 |
| 5 | R2 주석의 `int()` 크래시 인용 | 감사 §2 가 **기각한 소견**(같은 파일에 52곳) | 문장 삭제 |
| 6 | R2 *"[[62]] 한 번 더 생성할 뿐"* | 가드는 루프 몸통 **중간**의 조기 탈출 — hold 는 나머지 절반을 실행한다 | 주석 문구 정정 |
| 7 | R2 검정의 `guard_of` 수리(`n.body[-1] is Break`) | 새 몸통이 `if/else` 라 그것도 못 찾는다 | *"서브트리에 루프-레벨 Break 를 가진 최상위 `ast.If`"* 로 일반화 |
| 8 | R2 *"클로버 11건도 R2 의 하류"* | 그 11건의 **앞 턴 `am`** 을 안 봤다 | 삭제(주장 안 함) |
| 9 | R2 *"가드 술어 미수리 시 검사를 **조용히** 건너뛴다"* | 실제로는 **FAIL** 한다(반증자 실행) | 서술 정정(수리 필요성은 유지 — 22-fb 대조가 무검정이 된다) |
| 10 | R3 `_cp2_in_work` 정확일치 가드 | 큐 병합 초문자열이 통과 ⇒ 같은 문단 2회(실행 재현) | 폐기 — 부착 자체를 이동 |
| 11 | R3 소비 표시만 지연 | 살아남은 슬롯을 `_ctx_fits` 가 hist 로 이중계수 ⇒ 거짓 `skipped` + 신규 파괴 경로(실행 재현) | 폐기 — 이동으로 소멸 |
| 12 | R3 *"손실 11 → 0"* | 우회 뒤 깨끗한 부착이 있는 것은 1/11 | *"확정 손실 11 → 미확정"*. R2 ON 이면 같은 턴 회수 창(§2.5-3, 코드 형세·**미측정**) |
| 13 | R3 §5-[E] *"용량 초과 경로 불변"* · `:10651` 생성기 | 각각 거짓 / 줄번호 오기 | 이동본에서 참이 되고(검정 [F]), 줄번호 정정 |
| 14 | R4 P4 `_route_drain` | `test_route_trace.py:85-87` 정규식이 죽어 **런 거절** | 폐기 |
| 15 | R4 P5 VIEW_FB 등재 제거 | route 채널 · `t2_liveness` 과거 비교가 끊긴다 | 폐기(바이트 그대로) |
| 16 | R4 P7 ASUB `arrived=False` | 사실과 반대(#1) + 공유 큐(`:10129`·`:10155`) 오염 | 폐기 |
| 17 | R4 P0 *"읽는 문이 없어서"* | `t2_lever_beat.py:130` 이 이미 읽는다 | 근거 수정 + `:130` 리팩터 |
| 18 | R4 P3 `if _incoming:` | 자기 주석과 모순(재대입 이중 개방) | `and _incoming != _prev` |
| 19 | R4 P8 escaping | `$TAG` 미escape ⇒ 매번 회수 0 | `\$TAG` 로 수정 + "거절" → "경고" |
| 20 | R4 1차 종점 `arrived` | R3 이후 부착 = 도달, 동어반복(부착 인쇄의 재인쇄) | cp2 행에 `arrived` 를 안 쓴다. 닫힌 분할이 종점 |
| 21 | R4 `slot_bytes` 키 `(cp2_attach_turn, cp2_slot_n)` | `current_turn()` 은 턴당 1회만 심어져 충돌 실재 | 키를 `cp2_id` 로 |
| 22 | R4 *"사이드카 없으면 종점 계산 불가 → t7303 순환"* | 총계는 로그로 이미 넷이 냈다 | 조각 단위 검산만 불가 — 경고로 강등 |
| 23 | 감사 §5 R4 *"등재를 다섯 자리 전부로"* | 현행 PIN 에서 발화 가능한 자리는 **둘**(`T2_ACT_DEMAND=0`·`T2_DELIVER_PRECOMMIT=0`·`T2_MATERIAL_BYPASS` 미export) | 코드는 다섯 자리 공통 입구에 두되, 실효 증분은 `SEARCH_ON_PROCEED` 61건이라고 적는다 |
| 24 | R4 *"부착 후 ASUB 폐기 11건"* 을 [M] 로 | 추정자 의존(6줄 창 = 11 / 다른 규칙 = 17) | **[?]** 로 표기. 이 문서의 11 은 "부착 다음 이벤트가 ASUB 이고 6줄 이내" 규칙 |

---

## §5 적용 후 검증 절차

### 5.0 각 패치 직후(3회 반복)

```
python -m py_compile scripts/distill/tau2/t2_gate_patch.py
```
그리고 **다음 패치의 OLD 블록을 `grep -n` 으로 다시 뜬다**(줄번호가 밀렸다).

### 5.1 회귀 배터리 (전부 오프라인 · 모델 0). ★HEAD 초록 확인 완료 — 패치 뒤 같은 결과여야 한다

| 검정 | 무엇을 보증하나 | HEAD 기준값 |
|---|---|---|
| `test_route_trace.py` | route 계기가 `_gen` 안에 있고 content/work 를 안 건드린다 · `_SRC8` 순서 == 배타 체인 순서 | **22/22** |
| `test_regen_break_guard.py` | break 가드가 **모든 채널**(fb 22종 + 슬롯 속성)을 본다 · 카운터가 가드 뒤 | ALL PASS |
| `test_cp2_queue_behavior.py` | `_cp2_assign` 의 OFF 바이트 동일성(구판 replica 차분 매트릭스) | PASS |
| `test_cp2_clobber.py` | 슬롯이 하나 · 대용량 이어붙임 · 소비 지점 하나 | PASS |
| `test_decision_carry.py` | CP2 배달 계약(기본값·문구) | **24/24** |
| `test_proceed_docbody.py` | argmax 없음 · "정답은 X" 없음([[62]]) | PASS |
| **`test_cp2_asub_bypass.py`**(신규) | §2.6 [A]~[G] — 비-ASUB 동일성 · ASUB 회차 슬롯 생존 · 중복 부착 0 · `_ctx_fits` 이중계수 0 | — |
| **`test_cp2_arrival_instrument.py`**(신규) | §3.6 §A~§H — 바이트 불변 · 닫힌 분할 · 팔-대칭 형식 보증 · `arrived` 부재 | — |

⚠`test_regen_break_guard.py` 는 R2 와 **같은 커밋**에서 술어를 고쳐야 한다. 안 고치면 `found=0`
으로 FAIL 하고(★반증자 실행 확인) 러너가 `exit 1` 한다.
⚠`test_route_trace.py` 는 **R4 원안이라면 죽는다**. 채택안(P4·P5 폐기)에서는 22/22 가 유지되어야
하고, 유지되지 않으면 R4 를 잘못 적용한 것이다.

### 5.2 정적 감사(스크립트 없이 grep 으로)

1. `grep -n "self._t2_cp2_pending = " t2_gate_patch.py` → **정확히 3곳**(`_cp2_assign` 말미 ·
   ctx_skip · 부착).
2. R4 헬퍼 범위에서 `grep -n "T2_CP2_QUEUE\|_queue\|(queue)"` → **0건**(팔-대칭).
3. `awk` 로 R3 이동본 **앞** 구간(`_pin_r` 블록)에 `work`·`_t2_cp2` 참조 **0건** 재확인.
4. `grep -n "_route_drain\|_t2_route_pending" t2_gate_patch.py` → route 관련 줄이 **HEAD 와 동일**
   (`:6456 :6458 :9102 :9107 :10129 :10139 :10155 :10161`).
5. `git diff --stat` — `t2_gate_patch.py` · `t2_lever_beat.py` · 신규 검정 2파일 · A/B 러너.
   다른 파일이 섞였으면 중단.

### 5.3 로그 재생 대조 (런 0 · 기존 gz 로)

`bank_t7346_half{A,B}_20260822.log.gz` census 로 **패치 전 기준값**을 고정해 둔다:
`{assign 125, attach 100, clobber 11, append 2, residual 12}` · ASUB 우회 11 · 자리별
`VIEW_FB 64/64 · SEARCH_ON_PROCEED 36/61` · 잔존 12건 크기 `254×4 · 243×3 · 8735×2 · 6973 · 247 ·
263`. Stage 1 스모크 로그를 **같은 스크립트**로 돌려 검산식이 닫히는지 확인한다.

### 5.4 Stage 1 스모크 게이트 (8 sim: 2 태스크 × nt=2 × 2팔)

감사 §5 Stage 1 에 다음을 **추가**한다. 하나라도 실패하면 Stage 2 금지.

- ⓐ `T2_CP2_HOLD=1` 팔에서 `[T2_CP2_HOLD]` ≥1 · `[T2_CP2_UNHELD]` 계상.
- ⓑ **핀 승차 계상**([[57]] 의무): hold 로 새로 생긴 턴의 `[T2_PIN_READ] pinned` ·
  `[T2_READ_ROUTINE]` 건수와 **over-action(gold 없는 write)** 을 짝으로 센다. 0 이 아니면
  §1.6-⑴ 의 선택지를 메인 세션이 명시적으로 고른다.
- ⓒ **종단 턴 계상**: `last_assistant == 배달 turn` 비율. 지금 지표로는 R2 의 상방이 0 일 가능성이
  가장 크다(§1.4-2).
- ⓓ `[T2_DECISION_CARRY] … 부착` 이 **`[T2_ACTION_SUB]` 뒤**에 오는지(R3 가 먹었다는 증거) ·
  "부착 직후 ASUB" 인접 건수 = **0**.
- ⓔ 사이드카 `.fb.jsonl.gz` 가 실제로 회수됐는지(R4-ⓖ escaping 검증) · `kind="cp2"` 행에
  `assign`/`close` 가 둘 다 있는지 · `agent="cp2"` 인지.
- ⓕ 검산식 `assign == attached + clobbered + ctx_skip + 미종결` 이 **양 팔에서** 닫히는지.
- ⓖ duration · `usage.prompt_tokens` 증분 · `ContextWindowExceededError` 건수 계상.

---

## §6 남은 미측정 위험 ([[70]] 판 것 포함)

| # | 위험 | 등급 | 왜 못 쟀나 |
|---|---|---|---|
| 1 | **R2 ON 이 통과 중인 6 sim 의 마지막 발화를 갈아 끼운다.** 그 발화 바로 뒤에 손님이 실행하는 write 가 붙는다(★`task_003#s626729` msgs[6]→msgs[7] `apply_for_credit_card` → reward 1.0) | **[S]**(형세) / **[?]**(결과) | 라이브 A/B 없이는 부호를 알 수 없다. Stage 1 ⓒ 로만 근사 |
| 2 | **핀 승차** — `_gen(..., pin=_pin_r)` 이 산문 턴을 도구 호출 턴으로 뒤집을 수 있다(`go_stack.sh:327,328`) | **[?]** | 스모크 전까지 미측정 |
| 3 | **R3 가 파는 것 — 우회 턴의 claimprov·selfdecl 재료(오늘 11/11).** claimprov 산출은 `_unbacked` 판정을 거쳐 되먹임을 만들 수 있어 순수 계기 문제가 아니다 | **[?]** | 그 서브콜 산출의 차이를 오프라인으로 잴 방법이 없다 |
| 4 | **R3 의 회수율.** 우회 11건 중 뒤에 깨끗한 부착이 있는 것은 1건(OFF 팔 궤적 위의 반사실). R2 ON 이면 같은 턴 회수 창이 생긴다는 것은 **코드 형세 논증**이지 측정이 아니다 | **[?] 확인 안 됨** | 반사실 |
| 5 | **무라벨 다중 답의 밀도.** 이미 t7346 37 sim 중 17 sim(46%)이 235~300자 무라벨 결정문을 2회 이상 받는다. R2 는 `cp2_attached` 로 **자기 몫**을 봉인했지만 기존 밀도는 그대로다. 원천 수리는 **R6**(축 라벨링)이고 큐 A/B 와 같은 런 금지 | **[M]** | 감사 §1-B4 |
| 6 | **`arrived` 는 이제 동어반복이다.** R4 가 새로 주는 정보는 `clobbered`/`ctx_skip`/미종결의 **조각 단위 귀속**과 `sha` 뿐 — 총계는 이미 로그로 나온다 | **[S]** | — |
| 7 | **같은 본문 이중 개방**(§3.7-⑴) — 팔마다 분모가 달라질 수 있다 | **[?]** | `PRECOMMIT` 에 `_t2_cp2_said` 가드가 없다(지금은 死) |
| 8 | **`_t2_asub` 영구 래치**(`:8664` 이후 해제 0) — sim 후반이 통째로 ASUB 턴이 되면 슬롯 이월 · 부피 압력이 예상보다 클 수 있다 | **[?]** | 별건 결함 |
| 9 | **`_cap`(90,000) > `_ctx_fits` 절대 상한(85,596) 사수 대역** — 이 계획의 범위 밖(**R1**). R1 미적용이면 큐 A/B 자체가 무효 | **[S]** | 감사 §1-A3 |
| 10 | **C578(순서·매몰)이 이 깊이(≤8,735자)에서도 같은 크기인지** | **[?] 확인 안 됨** | C578 자신이 태스크 일반화를 미측정으로 못박았다 |
| 11 | **계기 부수 이동** — R2 hold 로 `record_many(fb=[am])`·`t2_stack.audit`·`_t2_silenced` 드레인이 전에 안 돌던 턴에 돈다. `x341_docbody_verdict.py` 같은 판정기의 분모가 이유 없이 움직일 수 있다 | **[?]** | 판정기 수리는 **R5** |
| 12 | **stderr 줄 순서 변경**(R3) — 로그 순서에 의존하는 기존 포렌식 스크립트가 있는지 | **[?] 확인 안 됨** | 전수 감사 안 함 |
| 13 | **사이드카 지연 기여**(+235행/40 sim) | **[?]** | Stage 1 ⓖ 로 계상 |
| 14 | **ASUB 우회 건수 11 자체가 추정자 의존** | **[?]** | 6줄 창 = 11 / 다른 규칙 = 17(반증자) |

### 마지막 한 줄

> 이 세 수리 중 **모델 가시 바이트를 안 바꾸는 것은 R4 하나**다. R2·R3 는 레버이고, 그 성적
> 효과는 **어느 층화에서도 실증되지 않았다**(감사 §1-B3·§6-21). 그래서 R2 는 플래그 뒤에 두고
> R3 는 R2 와 함께 재도록 설계했다 — *pass 를 팔지 않고 계기를 고치는 것*이 이 Stage 0 의 목적이다.
