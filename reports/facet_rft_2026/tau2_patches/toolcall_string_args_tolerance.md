# tau2 patch: ToolCall JSON-string arguments tolerance (2026-07-20)

**대상**: `tau2-bench/src/tau2/data_model/message.py` · `class ToolCall`
**계기**: 093·031(FORCE_ACTION) infra_error — 32B가 `arguments`를 dict 아닌 **JSON-문자열**로 방출
(특히 tool_choice=required 하에서 빈발) → pydantic `arguments: dict` 검증 거부 → 크래시.
**근거**: OpenAI API는 tool-call arguments를 **canonically JSON 문자열**로 반환한다. tau2의 `arguments: dict`
strict가 오히려 비표준. 문자열→dict coerce = LLM API 계약에 tau2를 정렬하는 것(채점 로직 무변경·ingestion 관용만).

## 패치 (scratch clone·gitignored라 여기 박제·재적용용)
`arguments: dict = Field(...)` 뒤에 삽입:
```python
    @field_validator("arguments", mode="before")
    @classmethod
    def coerce_arguments(cls, v):
        """Tolerate JSON-string arguments (OpenAI/vLLM canonically emit a JSON string,
        esp. under tool_choice=required). Parse to dict; leave dicts untouched.
        v2 (2026-07-20, e2e9 054 crash): a MALFORMED string (broken escapes/truncation)
        used to fall through to pydantic dict_type -> sim death. Degrade to {} instead:
        the env rejects the empty/wrong call with an error ToolMessage the agent can
        recover from (bad emission = recoverable turn, never a dead sim)."""
        if isinstance(v, str):
            import json as _json
            try:
                parsed = _json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                try:
                    parsed, _ = _json.JSONDecoder().raw_decode(v.strip())
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            import sys as _sys
            print("[TAU2_PATCH] unparseable tool-call arguments -> {} (recoverable): %r" % v[:120],
                  file=_sys.stderr, flush=True)
            return {}
        return v
```
검증: 문자열 args→dict coerce·dict 보존·비정형 문자열→{} 강등(sim 생존) 확인. FORCE_ACTION·093·054 크래시 해소.
⚠️리모트 tau2 clone에 적용됨(runner 재시작 시 자동 반영·**실행 중 프로세스엔 미반영**). 재클론 시 재적용 필요.
