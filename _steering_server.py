"""Phase 2a — Steering-Enabled OpenAI-Compatible Inference Server.

HuggingFace transformers + FastAPI. Loads agent model with forward pre-hooks
on residual streams; injects alpha * v_relation at specified layers.

Exposes /v1/chat/completions compatible with tau2-bench / LiteLLM clients.

Per-request steering control via 'extra_body' OR special chat message:
  {
    "model": "qwen7b-steer",
    "messages": [...],
    "tools": [...],
    "extra_body": {
        "steering": {"relation": "validates", "alpha": 0.5, "layers": [12,13,14]}
    }
  }

If steering is None or alpha == 0, baseline (no-op) inference runs.

Tool-call parsing borrowed from vLLM hermes / llama3_json parsers.

Usage:
  python _steering_server.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --steering-vectors steering_vectors_qwen7b.pt \
    --port 8200 --device cuda:0 \
    --tool-parser hermes
"""
import argparse
import json
import re
import time
import uuid
import threading
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# -------- request / response schemas (OpenAI subset) --------

class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class SteeringSpec(BaseModel):
    relation: str
    alpha: float = 0.5
    layers: List[int] = Field(default_factory=lambda: [12, 13, 14])


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    stream: bool = False
    stop: Optional[List[str]] = None
    extra_body: Optional[Dict[str, Any]] = None
    # also accept top-level "steering" for clients that don't support extra_body
    steering: Optional[SteeringSpec] = None


# -------- tool-call parsers --------

def parse_hermes_tool_calls(text: str) -> (str, List[Dict[str, Any]]):
    """Parse <tool_call>{json}</tool_call> blocks. Returns (clean_content, tool_calls)."""
    pattern = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    calls = []
    for m in pattern.finditer(text):
        try:
            j = json.loads(m.group(1))
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": j.get("name", ""),
                    "arguments": json.dumps(j.get("arguments", {})),
                },
            })
        except Exception:
            continue
    clean = pattern.sub("", text).strip()
    return clean, calls


def parse_llama3_json_tool_calls(text: str) -> (str, List[Dict[str, Any]]):
    """Llama-3.1 tool-call format: <|python_tag|>{json}<|eom_id|> or raw JSON."""
    calls = []
    s = text
    # python_tag-bracketed form
    py_pat = re.compile(r"<\|python_tag\|>\s*(\{.*?\})\s*(?:<\|eom_id\|>|$)", re.DOTALL)
    for m in py_pat.finditer(s):
        try:
            j = json.loads(m.group(1))
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": j.get("name", ""),
                    "arguments": json.dumps(j.get("parameters", j.get("arguments", {}))),
                },
            })
        except Exception:
            pass
    clean = py_pat.sub("", s).strip()
    if calls:
        return clean, calls
    # fallback: bare JSON object that looks like a tool call
    if clean.startswith("{") and '"name"' in clean:
        try:
            j = json.loads(clean)
            if "name" in j:
                calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": j["name"],
                        "arguments": json.dumps(j.get("parameters", j.get("arguments", {}))),
                    },
                })
                clean = ""
        except Exception:
            pass
    return clean, calls


# -------- steering hook manager --------

class SteeringHookManager:
    def __init__(self, model: torch.nn.Module, vectors: Dict[str, Dict[int, torch.Tensor]],
                 device: str, dtype: torch.dtype):
        self.model = model
        self.vectors = vectors
        self.device = device
        self.dtype = dtype
        self._hooks = []
        # cache vectors on device in correct dtype
        self._cache: Dict[(str, int), torch.Tensor] = {}

    def _get(self, relation: str, layer: int) -> Optional[torch.Tensor]:
        key = (relation, layer)
        if key in self._cache:
            return self._cache[key]
        rel_vecs = self.vectors.get(relation)
        if rel_vecs is None:
            return None
        v = rel_vecs.get(layer)
        if v is None:
            return None
        v = v.to(self.device, dtype=self.dtype)
        # normalize then keep as unit direction
        nrm = v.norm() + 1e-8
        v = v / nrm
        self._cache[key] = v
        return v

    def _layer_modules(self) -> List[torch.nn.Module]:
        # works for LlamaForCausalLM / Qwen2ForCausalLM
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        raise RuntimeError("Unknown model architecture; cannot locate transformer layers.")

    @contextmanager
    def apply(self, spec: Optional[SteeringSpec]):
        if spec is None or spec.alpha == 0.0:
            yield
            return
        layers = self._layer_modules()
        handles = []
        try:
            for L in spec.layers:
                v = self._get(spec.relation, L + 1)  # layer index l+1 maps to "after block L"
                if v is None:
                    continue
                alpha = float(spec.alpha)

                def make_hook(vec=v, a=alpha):
                    def hook(_module, _inputs, output):
                        # output is (hidden, ...) tuple OR just hidden
                        if isinstance(output, tuple):
                            h = output[0]
                            h = h + a * vec
                            return (h,) + output[1:]
                        return output + a * vec
                    return hook

                handles.append(layers[L].register_forward_hook(make_hook()))
            yield
        finally:
            for h in handles:
                h.remove()


# -------- generation --------

class Engine:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        print(f"[steer-srv] Loading model {args.model}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=self.dtype, device_map=args.device,
            attn_implementation="eager",
        )
        self.model.eval()
        print(f"[steer-srv] loaded in {time.time()-t0:.1f}s")

        # load steering vectors
        if args.steering_vectors:
            print(f"[steer-srv] Loading steering vectors {args.steering_vectors}...")
            blob = torch.load(args.steering_vectors, map_location="cpu", weights_only=False)
            vectors = blob["vectors"]
            self.hook_mgr = SteeringHookManager(self.model, vectors, args.device, self.dtype)
            print(f"[steer-srv] steering vectors loaded: {len(vectors)} relations")
        else:
            self.hook_mgr = None
            print(f"[steer-srv] NO steering vectors — baseline mode only")

        # parser
        if args.tool_parser == "hermes":
            self.parse_tools = parse_hermes_tool_calls
        elif args.tool_parser == "llama3_json":
            self.parse_tools = parse_llama3_json_tool_calls
        else:
            self.parse_tools = lambda t: (t, [])

    def _render_prompt(self, req: ChatCompletionRequest) -> str:
        msgs = [m.dict(exclude_none=True) for m in req.messages]
        tools = req.tools
        try:
            prompt = self.tokenizer.apply_chat_template(
                msgs, tools=tools, add_generation_prompt=True, tokenize=False,
            )
        except Exception:
            # fallback: no tools support in template
            prompt = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
            )
        return prompt

    @staticmethod
    def _parse_model_steering(model_name: str) -> Optional[SteeringSpec]:
        # forms:   base                       -> None
        #          base:relation@alpha        -> default layers
        #          base:relation@alpha/L1,L2  -> custom layers
        if ":" not in model_name:
            return None
        rest = model_name.split(":", 1)[1]
        if "@" not in rest:
            return None
        relation, tail = rest.split("@", 1)
        if "/" in tail:
            alpha_s, layers_s = tail.split("/", 1)
            layers = [int(x) for x in layers_s.split(",") if x.strip() != ""]
        else:
            alpha_s, layers = tail, [12, 13, 14]
        try:
            alpha = float(alpha_s)
        except ValueError:
            return None
        return SteeringSpec(relation=relation, alpha=alpha, layers=layers)

    @torch.no_grad()
    def generate(self, req: ChatCompletionRequest) -> Dict[str, Any]:
        prompt = self._render_prompt(req)
        spec = req.steering
        if spec is None and req.extra_body:
            s = req.extra_body.get("steering")
            if s:
                spec = SteeringSpec(**s)
        if spec is None:
            spec = self._parse_model_steering(req.model)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                              max_length=8192).to(self.args.device)
        gen_kwargs = dict(
            max_new_tokens=req.max_tokens,
            do_sample=(req.temperature > 0),
            temperature=max(req.temperature, 1e-5),
            top_p=req.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with self.lock:
            ctx = self.hook_mgr.apply(spec) if (self.hook_mgr and spec) else _nullctx()
            with ctx:
                out = self.model.generate(**enc, **gen_kwargs)

        new_tokens = out[0, enc["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        # strip trailing eos / pad
        text = text.replace(self.tokenizer.eos_token or "", "").strip()

        clean, tool_calls = self.parse_tools(text)

        # tau2 requires AT LEAST ONE of (content, tool_calls). If both missing
        # (e.g., model emitted malformed <tool_call> tags that we stripped),
        # fall back to the raw text so tau2 can decide.
        if not tool_calls and not clean:
            clean = text.strip() or " "  # never return totally empty
        msg = {"role": "assistant", "content": clean if clean else None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
            finish = "tool_calls"
            # content must be string or null (not both null and tool_calls-empty)
            if msg["content"] is None:
                msg["content"] = ""
        else:
            finish = "stop"
            if msg["content"] is None:
                msg["content"] = ""

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": msg,
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "completion_tokens": int(new_tokens.shape[0]),
                "total_tokens": int(enc["input_ids"].shape[1] + new_tokens.shape[0]),
            },
            "_steering": spec.dict() if spec else None,
        }


@contextmanager
def _nullctx():
    yield


# -------- FastAPI app --------

def build_app(engine: Engine) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": engine.args.served_model_name,
                                              "object": "model"}]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat(req: ChatCompletionRequest):
        try:
            return engine.generate(req)
        except Exception as e:
            import traceback; traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steering-vectors", default=None, help="path to .pt; omit for baseline")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tool-parser", default="hermes", choices=["hermes", "llama3_json", "none"])
    ap.add_argument("--served-model-name", default="steer-model")
    args = ap.parse_args()

    engine = Engine(args)
    app = build_app(engine)
    print(f"[steer-srv] listening on {args.host}:{args.port} model={args.served_model_name}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
