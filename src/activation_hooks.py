"""Forward-hook activation capture for HF transformer models on MPS.

Captures the residual-stream output of each decoder layer at the final token.
Returns a (n_layers, hidden_size) fp16 tensor.

Designed for raw HF transformers - no TransformerLens. MPS-friendly.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _find_decoder_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Locate the list of transformer decoder blocks across common architectures."""
    candidates = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("language_model", "model", "layers"),
    ]
    for path in candidates:
        obj = model
        ok = True
        for attr in path:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, torch.nn.ModuleList):
            return list(obj)
    raise RuntimeError(
        "Could not locate decoder layers. Available top-level attrs: "
        + str([n for n, _ in model.named_children()])
    )


class ActivationCapture:
    """Hook-based capture of per-layer residual streams.

    Use as a context manager. After running a forward pass while inside
    the context, call `.last(seq_index)` to fetch the activations at a given
    token position across all layers as a (n_layers, hidden) tensor.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layers = _find_decoder_layers(model)
        self._buffers: list[torch.Tensor | None] = [None] * len(self.layers)
        self._handles: list = []

    def _make_hook(self, idx: int):
        def hook(_module, _inputs, output):
            # Decoder block output is either a tensor or a tuple whose first
            # element is the hidden state.
            hs = output[0] if isinstance(output, tuple) else output
            # Detach and keep on device; we'll cast to fp16 when extracting.
            self._buffers[idx] = hs.detach()
        return hook

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def last(self, seq_index: int = -1) -> torch.Tensor:
        """Return (n_layers, hidden) fp16 tensor at seq_index of the final pass."""
        out = []
        for buf in self._buffers:
            if buf is None:
                raise RuntimeError("No activations captured. Did forward pass run?")
            # buf shape: (batch, seq, hidden). Take batch 0, seq=seq_index.
            vec = buf[0, seq_index, :].to(torch.float16).cpu()
            out.append(vec)
        return torch.stack(out, dim=0)


def load_model(model_id_or_path: str, device: str = "mps", dtype: torch.dtype = torch.bfloat16, load_in_4bit: bool = False):
    """Load tokenizer and causal LM, ready for hooked inference.

    On CUDA, optionally quantize to 4-bit for big models.
    """
    tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # transformers 4.x uses torch_dtype, 5.x uses dtype. Try both for portability.
    import transformers as _tx
    major = int(_tx.__version__.split(".")[0])
    dtype_kw = "dtype" if major >= 5 else "torch_dtype"
    kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    kwargs[dtype_kw] = dtype
    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = device
        # Remove dtype kw since quantization handles dtype
        kwargs.pop(dtype_kw, None)
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
        model = model.to(device)
    model.eval()
    return tok, model


@contextmanager
def timer(label: str) -> Iterator[dict]:
    info: dict = {}
    t0 = time.perf_counter()
    try:
        yield info
    finally:
        info["elapsed_s"] = time.perf_counter() - t0
        info["label"] = label


def probe_and_capture(
    tok,
    model,
    chat_messages: list,
    device: str = "mps",
    max_new_tokens: int = 80,
    temperature: float = 0.9,
) -> tuple[torch.Tensor, str]:
    """Apply chat template, run forward, capture activations at final input token,
    then generate the model's response continuing from there.

    Returns (activations [n_layers, hidden] fp16, response_text).
    """
    prompt = tok.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(device)
    input_len = enc["input_ids"].shape[1]

    with ActivationCapture(model) as cap:
        with torch.no_grad():
            _ = model(**enc)
        acts = cap.last(seq_index=-1)

    # Now generate a continuation for the probe answer.
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
    response_text = tok.decode(gen[0, input_len:], skip_special_tokens=True).strip()
    return acts, response_text


def generate_response(tok, model, chat_messages: list, device: str = "mps", max_new_tokens: int = 200) -> str:
    """Generate without activation capture (for reaction turns)."""
    prompt = tok.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(device)
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(gen[0, input_len:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    # Smoke test: try loading the local Qwen3.5-0.8B and capturing one activation.
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
    print(f"Loading {model_path} ...", flush=True)
    with timer("load") as t:
        tok, model = load_model(model_path)
    print(f"  loaded in {t['elapsed_s']:.1f}s", flush=True)

    messages = [
        {"role": "system", "content": "You are a thoughtful undecided professional in your 30s."},
        {"role": "user", "content": "What is your current view on remote work? Two sentences."},
    ]

    with timer("probe") as t:
        acts, text = probe_and_capture(tok, model, messages)
    print(f"  probe ran in {t['elapsed_s']:.2f}s", flush=True)
    print(f"  acts shape: {tuple(acts.shape)} dtype: {acts.dtype}", flush=True)
    print(f"  response: {text[:160]!r}", flush=True)
