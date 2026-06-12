#!/usr/bin/env python
"""P-D0 diffusion 제안기 스모크 (TB_DIFFUSION_PROPOSER_DESIGN §3).

Dream-7B(diffusion_generate, vllm 불가)로 MM 프롬프트 N개 × K샘플 생성.
프롬프트 = inference.py resource-분기와 동일 문자열(공정 비교), 출력 = inference.py
호환 jsonl({**record, "result": parsed}) per-k 파일 → tb_kgate_heldout.py로 풀 분석.

1차 관문 = 형식 준수율(JSON 파싱율·valid_frac). K샘플 temp=0.8(AR 풀과 동일).

Run (GPU 1장, day 배치 종료 후):
  CUDA_VISIBLE_DEVICES=0 /home/woori/venvs/seka_env/bin/python tb_diffusion_sample.py \
    --data_dir /home/woori/scratch/JARVIS_tb/taskbench/data_multimedia \
    --out_prefix /home/woori/scratch/tb_dream/pd0 --n 50 --k 4
"""
import argparse, json, os, re


def build_prompt(tool_string, user_request):
    # inference.py:186-187,193,197,199 resource 분기 재현 (demos 없음)
    prompt = """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}]} """
    prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; \n4. the tool arguments should be align with the input-type field of # TASK LIST #;"""
    prompt += "\n"
    prompt += """\n\n# USER REQUEST #: {{user_request}}\nnow please generate your result in a strict JSON format:\n# RESULT #:"""
    return tool_string + prompt.replace("{{user_request}}", user_request)


def extract_json(text):
    """가벼운 parse-fix: 첫 '{'부터 brace-매칭 영역 추출 후 json.loads."""
    s = text.find("{")
    if s < 0:
        return None
    depth = 0
    for i in range(s, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[s:i + 1])
                except ValueError:
                    return None
    # 미완결 brace — 흔한 truncation: 닫는 괄호 보충 1회 시도
    frag = text[s:]
    fix = frag + "]}" * 3
    m = re.search(r"^(.*\})", fix, re.S)
    if m:
        try:
            return json.loads(frag + "}" * (frag.count("{") - frag.count("}")))
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    tool_list = json.load(open(f"{a.data_dir}/tool_desc.json"))["nodes"]
    tool_string = "# TASK LIST #:\n"
    for tool in tool_list:
        tool_string += json.dumps(tool) + "\n"
    valid = {t["id"] for t in tool_list}

    records = [json.loads(l) for l in open(f"{a.data_dir}/data.json")]
    records = sorted(records, key=lambda d: d["id"])[: a.n]

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).cuda().eval()

    os.makedirs(os.path.dirname(a.out_prefix), exist_ok=True)
    n_parse = n_total = 0
    vfracs, plans_by_id = [], {}
    writers = [open(f"{a.out_prefix}_k{k}.json", "w") for k in range(a.k)]
    for ri, rec in enumerate(records):
        msgs = [{"role": "user", "content": build_prompt(tool_string, rec["user_request"])}]
        inputs = tok.apply_chat_template(msgs, return_tensors="pt", return_dict=True,
                                         add_generation_prompt=True)
        ids = inputs.input_ids.cuda()
        am = inputs.attention_mask.cuda()
        for k in range(a.k):
            torch.manual_seed(1000 * k + ri)
            out = model.diffusion_generate(
                ids, attention_mask=am, max_new_tokens=a.max_new_tokens,
                output_history=False, return_dict_in_generate=True,
                steps=a.steps, temperature=a.temperature, top_p=a.top_p,
                alg="entropy", alg_temp=0.0)
            text = tok.decode(out.sequences[0][ids.shape[1]:].tolist(),
                              skip_special_tokens=True)
            parsed = extract_json(text)
            n_total += 1
            res = parsed if isinstance(parsed, dict) else {"raw": text[:2000]}
            if isinstance(parsed, dict) and isinstance(parsed.get("task_nodes"), list):
                n_parse += 1
                names = [x.get("task") for x in parsed["task_nodes"] if isinstance(x, dict)]
                vfracs.append(sum(x in valid for x in names) / max(len(names), 1))
                plans_by_id.setdefault(rec["id"], []).append(tuple(sorted(
                    x for x in names if isinstance(x, str))))
            writers[k].write(json.dumps({**rec, "result": res}) + "\n")
            writers[k].flush()
        if (ri + 1) % 10 == 0:
            print(f"[pd0] {ri + 1}/{len(records)} parse={n_parse}/{n_total}", flush=True)
    for w in writers:
        w.close()

    # 다양성: distinct-plan율 + pairwise 노드셋 Jaccard
    dis, jacc, npairs = 0.0, 0.0, 0
    for plans in plans_by_id.values():
        dis += len(set(plans)) / len(plans)
        for i in range(len(plans)):
            for j in range(i + 1, len(plans)):
                s1, s2 = set(plans[i]), set(plans[j])
                jacc += len(s1 & s2) / max(len(s1 | s2), 1)
                npairs += 1
    nid = max(len(plans_by_id), 1)
    print(f"[pd0 VERDICT] parse_rate={n_parse}/{n_total}={n_parse / max(n_total, 1):.3f} "
          f"valid_frac={sum(vfracs) / max(len(vfracs), 1):.3f} "
          f"distinct_plan={dis / nid:.3f} mean_jaccard={jacc / max(npairs, 1):.3f} "
          f"(관문: parse>=0.5)")


if __name__ == "__main__":
    main()
