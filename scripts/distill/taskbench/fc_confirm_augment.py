#!/usr/bin/env python
"""fc_confirm_augment.py — P6 confirm-gate 합성 (SYNTHESIS_IMPL_SPEC §A, 2026-06-15).

SOPBench native-FC 궤적에 비가역 write 전 confirm-then-write 주입:
  pos: [A confirm "...Shall I proceed?"][U "Yes"][원래 write]   (every-write)
  neg: [A confirm][U "no"][A "won't proceed"] + write 제거·절단 (반사실=게이트화)

★write/read 분류 = 반환시그니처(spec §write/read·prefix 폐기):
  궤적 내 tool 출력 shape — bare bool("True"/"False") = write / tuple("(bool,val)") = getter.
  (get_loan→"True"=write 정확·internal_check→"(T,F)"=getter 정확.) + auth/meta carveout.

출력 = {tools, messages, _meta} native-FC. 삽입 assistant 턴 = _supervise:true.
값은 sop_rand2(이미 val-random)서 그대로 echo = copy 강제·날조0.

Usage: fc_confirm_augment.py --in sop_rand2.jsonl --out sop_confirm.jsonl \
         --frac_pos 0.65 --frac_neg 0.35 --sample 2000 --seed 42
"""
import argparse, json, random, re

# 반환시그니처 carveout: bool 반환이나 write 아님 (auth) · meta
AUTH_CARVEOUT = {"login_user", "logout_user", "authenticate_admin_password",
                 "authenticate_password", "verify_identity"}
META = {"exit_conversation", "transfer_to_human_agents", "transfer_to_human"}
# ★read-술어 prefix (bare bool 반환해도 write 아님 — QC 반증: internal_is_loyalty_member→False=read).
#   get_ 는 *제외 안 함*(get_loan=write는 output-shape로 잡음). 반환시그니처+이 배제 = 이중안전.
READ_PREFIX = ("internal_", "is_", "has_", "check_", "verify_", "show_",
               "list_", "view_", "search_", "find_", "query_", "lookup_")

CONFIRM_TMpl = [
    "To confirm: I'll {act} with {args}. Shall I proceed?",
    "Just to confirm before I proceed — I'm about to {act} with {args}. Should I go ahead?",
    "Before I {act} ({args}), can you confirm you'd like me to proceed?",
    "I'd like to confirm: this will {act} with {args}. Proceed?",
]
YES_TMpl = ["Yes, please proceed.", "Yes, go ahead.", "Confirmed, please proceed.",
            "Yes, that's correct — go ahead."]
NO_TMpl = ["Actually, no — please don't.", "On second thought, please don't proceed.",
           "No, hold off — don't do that.", "Actually, cancel that — don't proceed."]
ACK_TMpl = ["Understood, I won't proceed. Is there anything else I can help with?",
            "No problem, I won't proceed with that. Anything else?",
            "Okay, I'll hold off on that. Is there anything else?"]


def is_bare_bool(s):
    return str(s).strip() in ("True", "False", "true", "false")


def humanize(name):
    return name.replace("_", " ")


def fmt_args(arguments):
    try:
        d = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except Exception:
        d = {}
    if not isinstance(d, dict) or not d:
        return "the given details"
    parts = []
    for k, v in d.items():
        sv = str(v)
        if len(sv) > 24:
            sv = sv[:22] + "…"
        parts.append("%s=%s" % (k, sv))
    return ", ".join(parts)


def write_indices(msgs):
    """write tool_call 을 가진 assistant 메시지 인덱스 (반환시그니처=bare bool)."""
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        tcs = m["tool_calls"]
        # v1: 단일 tool_call 메시지 가정(SOPBench 궤적 구조). 다중이면 첫 write 기준.
        for tc in tcs:
            nm = tc.get("function", {}).get("name", "")
            if nm in AUTH_CARVEOUT or nm in META or nm.startswith(READ_PREFIX):
                continue
            nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
            if nxt.get("role") == "tool" and is_bare_bool(nxt.get("content")):
                out.append((i, tc))
                break
    return out


def confirm_pair(tc, rng, branch):
    nm = tc.get("function", {}).get("name", "")
    args = tc.get("function", {}).get("arguments", "{}")
    q = rng.choice(CONFIRM_TMpl).format(act=humanize(nm), args=fmt_args(args))
    conf = {"role": "assistant", "_supervise": True, "content": q}
    if branch == "pos":
        return [conf, {"role": "user", "content": rng.choice(YES_TMpl)}]
    return [conf, {"role": "user", "content": rng.choice(NO_TMpl)},
            {"role": "assistant", "_supervise": True, "content": rng.choice(ACK_TMpl)}]


def augment(d, rng, frac_neg):
    msgs = d["messages"]
    wr = write_indices(msgs)
    if not wr:
        return None
    branch = "neg" if rng.random() < frac_neg else "pos"
    write_name = wr[0][1].get("function", {}).get("name")

    if branch == "pos":
        # every-write: 각 write assistant 메시지 앞에 confirm+yes 삽입
        wset = {i for i, _ in wr}
        new = []
        for i, m in enumerate(msgs):
            if i in wset:
                tc = next(t for j, t in wr if j == i)
                new.extend(confirm_pair(tc, rng, "pos"))
            new.append(m)
        out_msgs = new
    else:
        # neg: 첫 write 앞에 confirm+no+ack, 그 write(+tool 응답)·이후 전부 절단
        fi, ftc = wr[0]
        out_msgs = list(msgs[:fi])
        out_msgs.extend(confirm_pair(ftc, rng, "neg"))
        # exit_conversation 으로 깔끔 종료(포맷 일관) — 원 궤적에 있으면 그 형식 재사용
        out_msgs.append({"role": "assistant", "_supervise": True,
                         "tool_calls": [{"id": "call_p6neg", "type": "function",
                                         "function": {"name": "exit_conversation", "arguments": "{}"}}]})

    meta = dict(d.get("_meta", {}))
    meta.update({"p6_confirm": True, "branch": branch, "write_name": write_name})
    return {"tools": d.get("tools"), "messages": out_msgs, "_meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac_neg", type=float, default=0.35)
    ap.add_argument("--sample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # write 포함 후보 수집 (should_succeed 우선=confirm 의미)
    cands = []
    for line in open(a.inp, encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if write_indices(d.get("messages", [])):
            cands.append(d)
    rng.shuffle(cands)
    cands = cands[:a.sample]

    npos = nneg = nskip = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for d in cands:
            r = augment(d, rng, a.frac_neg)
            if r is None:
                nskip += 1
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["_meta"]["branch"] == "pos":
                npos += 1
            else:
                nneg += 1
    print("[fc_confirm] candidates=%d written=%d (pos=%d neg=%d) skip=%d -> %s"
          % (len(cands), npos + nneg, npos, nneg, nskip, a.out))


if __name__ == "__main__":
    main()
