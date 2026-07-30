# -*- coding: utf-8 -*-
"""X14 — hermes 파서 강등이 "실행 안 함"으로 오계상되는지 전수 조사 (2026-07-31·무료).

★동기(사용자 지적): vLLM hermes 파서 소스(`hermes_tool_parser.py::extract_tool_calls`)에 경로가
**둘** 있다.
  ① 정상: `content = model_output[:model_output.find("<tool_call>")]`
     ⇒ **호출 뒤 텍스트는 버려진다**(선언을 뒤에 쓰면 없는 것으로 계상)
  ② 예외: `except Exception: tools_called=False, tool_calls=[], content=model_output`
     ⇒ **JSON 파싱이 실패하면 호출 자체가 텍스트로 강등**된다

②가 우리 실패 분류의 지배 조각과 직접 충돌한다: C245의 `NAME_ABSENT` 45.5%(외부 도구를 아예
안 부름)·C244의 EXEC_GAP 20.6%·T2_FORCE_ACTION이 겨냥한 "say-don't-do"는 전부 **호출 부재**로
관측된 것들이다. 그중 일부가 **모델이 호출을 냈는데 파서가 강등한 것**이라면, 그 조각은 능력·부하
문제가 아니라 **서빙 층 아티팩트**다.

판정 술어(전부 닫힘·궤적 텍스트만 본다):
  · `DEMOTED_TAG`   : assistant 메시지에 tool_calls가 **없는데** 본문에 `<tool_call>` 태그가 있다
  · `DEMOTED_JSONish`: tool_calls 없고 본문이 `{"name": ..., "arguments": ...}` 형태를 담고 있다
  · `POST_CALL_TEXT` : tool_calls가 있고 content도 비어있지 않다(=호출 앞 텍스트가 살아남은 경우)
  · `EMPTY_WITH_CALL`: tool_calls가 있고 content가 비었다(뒤에 썼다면 잘렸을 수 있는 후보)

⚠이 도구는 **강등 후보**를 세는 것이지 강등을 증명하지 않는다. `<tool_call>` 태그가 본문에 있으면
파서가 잡지 못한 것이 확실하지만(①·② 어느 경로든 태그는 content에 안 남는다), JSONish는
모델이 그냥 JSON을 서술한 경우와 구별되지 않는다 — 그래서 둘을 **따로** 센다([[08]]).

용법: py -3 x14_parser_demotion_census.py
"""
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
GLOB = "bank_day*front[AB]_*.results.json.gz"

_TAG = re.compile(r"<tool_call>|</tool_call>|<function_call>|<tools_call>")
_JSONISH = re.compile(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:', re.S)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    files = sorted(glob.glob(os.path.join(_SIM, GLOB)))
    if not files:
        sys.exit("궤적 0 — 경로 확인")

    cnt = Counter()
    examples = {"DEMOTED_TAG": [], "DEMOTED_JSONish": []}
    per_file = {}
    for path in files:
        d = json.load(gzip.open(path, "rt", encoding="utf-8"))
        local = Counter()
        for sim in d.get("simulations") or []:
            for m in sim.get("messages") or []:
                if m.get("role") != "assistant":
                    continue
                tcs = m.get("tool_calls") or []
                c = m.get("content")
                c = c if isinstance(c, str) else ""
                local["assistant_msgs"] += 1
                if tcs:
                    local["with_calls"] += 1
                    local["EMPTY_WITH_CALL" if not c.strip() else "POST_CALL_TEXT"] += 1
                else:
                    local["no_calls"] += 1
                    if _TAG.search(c):
                        local["DEMOTED_TAG"] += 1
                        if len(examples["DEMOTED_TAG"]) < 6:
                            examples["DEMOTED_TAG"].append(
                                (os.path.basename(path), sim.get("task_id"), c[:220]))
                    elif _JSONISH.search(c):
                        local["DEMOTED_JSONish"] += 1
                        if len(examples["DEMOTED_JSONish"]) < 6:
                            examples["DEMOTED_JSONish"].append(
                                (os.path.basename(path), sim.get("task_id"), c[:220]))
        per_file[os.path.basename(path)] = local
        cnt.update(local)

    print("궤적 파일 %d · assistant 메시지 %d" % (len(files), cnt["assistant_msgs"]))
    print("  도구 호출 있음 %d  (그중 content 있음 %d · content 빔 %d)"
          % (cnt["with_calls"], cnt["POST_CALL_TEXT"], cnt["EMPTY_WITH_CALL"]))
    print("  도구 호출 없음 %d" % cnt["no_calls"])
    print()
    print("=== ★강등 후보 (호출 없음인데 본문이 호출처럼 생김) ===")
    print("  DEMOTED_TAG    (본문에 <tool_call> 태그) : **%d**" % cnt["DEMOTED_TAG"])
    print("  DEMOTED_JSONish(본문이 name/arguments JSON): %d" % cnt["DEMOTED_JSONish"])
    tot = cnt["DEMOTED_TAG"] + cnt["DEMOTED_JSONish"]
    if cnt["no_calls"]:
        print("  ⇒ 호출-없음 메시지의 %.1f%% (%d/%d)"
              % (100.0 * tot / cnt["no_calls"], tot, cnt["no_calls"]))
    print()
    for k, ex in examples.items():
        if not ex:
            print("  (%s 사례 0)" % k)
            continue
        print("  --- %s 사례 %d건(최대 6) ---" % (k, len(ex)))
        for f, t, c in ex:
            print("    %s / %s" % (f[:34], t))
            print("      %s" % c.replace("\n", " ")[:200])
    print()
    print("파일별 강등 후보:")
    for f, l in per_file.items():
        print("  %-44s TAG %2d · JSONish %2d · 호출없음 %3d"
              % (f[:44], l["DEMOTED_TAG"], l["DEMOTED_JSONish"], l["no_calls"]))
    print()
    print("⚠[[08]]: 이 수는 **후보**다. TAG는 파서가 못 잡은 것이 확실하나, JSONish는 모델이 JSON을")
    print("  서술한 경우와 구별되지 않는다. 사례 정독 후에만 귀속할 것.")


if __name__ == "__main__":
    main()
