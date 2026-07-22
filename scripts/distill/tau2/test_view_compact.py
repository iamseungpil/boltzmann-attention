# -*- coding: utf-8 -*-
"""T2_VIEW_COMPACT(생성-뷰 압축·2026-07-21 §2bi·097 컨텍스트 레버) 오프라인 테스트.
검정: ①문턱 미만=무개입(거동보존) ②오래된 벌크 tool 출력만 다이제스트·최근 K개/유저·에이전트/에러/
짧은 출력 보존 ③커밋 원본 불변(뷰만 복사) ④다이제스트에 head+tail 보존 ⑤결정론.
⚠️단위통과≠라이브발화([[30]])."""
import sys, os, types
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception: pass
def mkmod(n):
    m = types.ModuleType(n); sys.modules[n] = m; return m
mkmod("tau2"); mkmod("tau2.agent"); la = mkmod("tau2.agent.llm_agent")
mkmod("tau2.data_model"); msgmod = mkmod("tau2.data_model.message")
mkmod("tau2.orchestrator"); oo = mkmod("tau2.orchestrator.orchestrator")
msgmod.ToolMessage = type("ToolMessage", (), {})
msgmod.UserMessage = type("UserMessage", (), {})
msgmod.MultiToolMessage = type("MultiToolMessage", (), {})
la.LLMAgent = type("LLMAgent", (), {})
oo.BaseOrchestrator = type("BaseOrchestrator", (), {"__init__": lambda s, **k: None})
import t2_gate_patch as G  # noqa: E402

class M:
    def __init__(self, role, content, id=None, error=False):
        self.role, self.content, self.id, self.error = role, content, id, error

ok = True
def chk(c, msg):
    global ok
    ok &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + msg)

BIG = "HEAD-" + ("x" * 3000) + "-TAIL"
hist = [M("user", "hi")]
for i in range(10):
    hist.append(M("assistant", "calling"))
    hist.append(M("tool", BIG, id="t%d" % i))
hist.append(M("tool", "short output", id="tshort"))
hist.append(M("tool", "E" * 2000, id="terr", error=True))

print("① 문턱 미만 = 무개입:")
v, dg = G._compact_view(hist, keep_recent=2, min_len=800, min_total=10**9)
chk(v[2].content == BIG and not dg, "min_total 미만 → 원문 그대로·digested 0")

print("② 선택적 다이제스트:")
v, dg = G._compact_view(hist, keep_recent=2, min_len=800, min_total=1000)
tools = [m for m in v if m.role == "tool"]
chk("view digest" in tools[0].content, "가장 오래된 벌크 출력 = 다이제스트")
chk(tools[-3].content == BIG or "view digest" in tools[-3].content, "(구성 확인용)")
# 최근 keep_recent=2개 tool = tshort·terr → 원문 유지; 그 앞 t9 등은 다이제스트
chk(v[-2].content == "short output", "짧은 출력 보존")
chk(v[-1].content == "E" * 2000, "에러 출력 보존(다이제스트 제외)")
chk(all(m.content == "calling" for m in v if m.role == "assistant"), "assistant 무개입")
chk(v[0].content == "hi", "user 무개입")
chk("t0" in dg and "tshort" not in dg and "terr" not in dg, "digested id 집합 정확")

print("③ 원본 불변·head/tail 보존:")
chk(hist[2].content == BIG, "커밋 히스토리 원본 불변(뷰만 복사)")
dmsg = [m for m in v if m.role == "tool"][0].content
chk(dmsg.startswith("HEAD-") and dmsg.endswith("-TAIL"), "다이제스트가 head/tail 원문 보존")
chk(("%d chars" % len(BIG)) in dmsg, "원문 길이 안내 포함")

print("④ 결정론:")
v2, dg2 = G._compact_view(hist, keep_recent=2, min_len=800, min_total=1000)
chk([m.content for m in v] == [m.content for m in v2] and dg == dg2, "동일 입력→동일 뷰")

# ── §2bi _pairfix 검정 (같은 파일에 부속·별도 하니스 불요) ──
print("⑤ _pairfix: 스왑 블록 교정:")
class TC2:
    def __init__(self, id): self.id = id
class AMsg:
    def __init__(self, tcs): self.role, self.tool_calls = "assistant", tcs
class TMsg:
    def __init__(self, id): self.role, self.id, self.tool_calls = "tool", id, None
hist2 = [AMsg([TC2("a"), TC2("b"), TC2("c")]), TMsg("a"), TMsg("c"), TMsg("b"),
         AMsg([TC2("d")]), TMsg("d")]
n = G._pairfix(hist2)
chk(n == 1, "스왑 1블록 교정 카운트")
chk([m.id for m in hist2[1:4]] == ["a", "b", "c"], "결과 순서 = 호출 순서로 복원")
chk(hist2[5].id == "d", "정상 블록 무개입")
n2 = G._pairfix(hist2)
chk(n2 == 0, "재실행 시 무개입(멱등)")
hist3 = [AMsg([TC2("x"), TC2("y")]), TMsg("x"), TMsg("z")]   # 집합 불일치 → 교정 안 함
n3 = G._pairfix(hist3)
chk(n3 == 0 and hist3[2].id == "z", "id 집합 불일치 = 무개입(과잉교정 방지)")

print("\n%s" % ("PASS(+pairfix)" if ok else "FAIL"))
sys.exit(0 if ok else 1)
