"""
Convert the fused Llama-3.2-3B-Instruct model (models/fused-hf-model) into a
stateful Core ML .mlpackage with a slice-updated KV-cache, then quantize to INT4.

Adapted from HuggingFace's swift-transformers Mistral7B example
(https://github.com/huggingface/swift-transformers/blob/preview/Examples/Mistral7B/export.py),
but rewritten against the current `transformers` attention-interface API
(no more per-model `*_ATTENTION_CLASSES` dict / separate Sdpa attention class —
`LlamaAttention.forward()` now takes a precomputed `position_embeddings` tuple
and dispatches through `ALL_ATTENTION_FUNCTIONS`). We bypass that dispatch and
call `scaled_dot_product_attention` directly, same approach as the reference
example, just matched to the current method signatures.

Requires: iOS18 / macOS15 minimum deployment target (stateful models are not
available on iOS17).
"""
import logging
import os
import warnings
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

warnings.filterwarnings("ignore")
logging.getLogger("coremltools").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = "models/fused-hf-model"
OUT_DIR = "models"
METADATA_TOKENIZER = "co.huggingface.exporters.name"


class SliceUpdateKeyValueCache:
    """Plain (non-HF-Cache-subclass) KV cache using in-place slice updates,
    which the Core ML GPU compiler can optimize into a stateful buffer update
    instead of a full copy. We don't subclass transformers.Cache here — the
    current Cache base class is a much heavier CacheLayerMixin-based design
    meant for the generic generate() loop; we only need the narrow `.update()`
    contract that our own custom attention forward (below) calls into."""

    def __init__(self, shape: Tuple[int, ...], dtype=torch.float32):
        # shape: (#layers, batch_size, #kv_heads, max_context_size, head_dim)
        self.past_seen_tokens: int = 0
        self.k_cache = torch.zeros(shape, dtype=dtype)
        self.v_cache = torch.zeros(shape, dtype=dtype)

    def update(self, k_state, v_state, layer_idx, slice_indices):
        begin, end = slice_indices
        self.k_cache[layer_idx, :, :, begin:end, :] = k_state
        self.v_cache[layer_idx, :, :, begin:end, :] = v_state
        return self.k_cache[layer_idx, :, :, :end, :], self.v_cache[layer_idx, :, :, :end, :]


def slice_update_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """Drop-in replacement for LlamaAttention.forward bound onto each layer's
    self_attn module. Matches current signature (position_embeddings passed
    in, not computed here) but uses our slice-update cache + direct SDPA call
    instead of the ALL_ATTENTION_FUNCTIONS dispatch."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    end_step = attention_mask.shape[-1]
    q_len = query_states.shape[2]
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, slice_indices=(end_step - q_len, end_step)
    )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_output = F.scaled_dot_product_attention(
        query_states, key_states, value_states, attn_mask=attention_mask, scale=self.scaling
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    return attn_output, None


class StatefulLlamaForCausalLM(torch.nn.Module):
    def __init__(self, model_path: str, max_context_size: int = 2048, batch_size: int = 1):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(model_path, dtype=torch.float32)
        self.model.eval()

        # Bind our custom forward onto every decoder layer's self-attention module.
        import types

        for layer in self.model.model.layers:
            layer.self_attn.forward = types.MethodType(slice_update_attention_forward, layer.self_attn)

        config = self.model.config
        self.kv_cache_shape: Tuple[int, ...] = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context_size,
            config.hidden_size // config.num_attention_heads,
        )
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)

    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor, causal_mask: torch.Tensor) -> torch.Tensor:
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        seq_len = input_ids.shape[-1]
        position_ids = torch.arange(
            self.kv_cache.past_seen_tokens, self.kv_cache.past_seen_tokens + seq_len
        ).unsqueeze(0)

        llama_model = self.model.model
        hidden_states = llama_model.embed_tokens(input_ids)
        position_embeddings = llama_model.rotary_emb(hidden_states, position_ids)

        for decoder_layer in llama_model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=self.kv_cache,
                position_embeddings=position_embeddings,
            )

        hidden_states = llama_model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits


def build_causal_mask(query_length: int, end_step: int) -> torch.Tensor:
    """(1, 1, query_length, end_step) float mask, -inf above the diagonal
    (offset so cached positions are always attendable)."""
    mask = torch.full((query_length, end_step), float("-inf"), dtype=torch.float32)
    past = end_step - query_length
    for i in range(query_length):
        mask[i, : past + i + 1] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


def sanity_check_against_reference(max_context_size: int):
    """Run a few tokens through both the original HF model (normal cache) and
    our stateful wrapper (slice-update cache) and compare logits, before
    spending time on the full trace + convert + quantize pipeline."""
    print("Sanity check: stateful wrapper vs reference HF model...")
    tokenizer_model = LlamaForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    tokenizer_model.eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    messages = [{"role": "user", "content": "i feel so tired and empty lately"}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    with torch.no_grad():
        ref_out = tokenizer_model(input_ids).logits
    ref_next_token = ref_out[0, -1].argmax().item()
    print(f"  Reference next-token id: {ref_next_token} ({tokenizer.decode([ref_next_token])!r})")

    wrapper = StatefulLlamaForCausalLM(MODEL_PATH, max_context_size=max_context_size)
    wrapper.eval()
    seq_len = input_ids.shape[-1]
    causal_mask = build_causal_mask(seq_len, seq_len)
    with torch.no_grad():
        wrapped_out = wrapper(input_ids.to(torch.int32), causal_mask)
    wrapped_next_token = wrapped_out[0, -1].argmax().item()
    print(f"  Wrapper next-token id:   {wrapped_next_token} ({tokenizer.decode([wrapped_next_token])!r})")

    max_diff = (ref_out[0, -1] - wrapped_out[0, -1]).abs().max().item()
    print(f"  Max logit diff on last position: {max_diff:.6f}")
    assert ref_next_token == wrapped_next_token, "Next-token prediction MISMATCH — wrapper is not equivalent!"
    print("  MATCH — wrapper produces the same next-token prediction as the reference model.")
    return wrapper


def export(max_context_size: int = 2048) -> None:
    print(f"Building StatefulLlamaForCausalLM (max_context_size={max_context_size})...")
    torch_model = StatefulLlamaForCausalLM(MODEL_PATH, max_context_size=max_context_size)
    torch_model.eval()

    print("Tracing with torch.jit.trace...")
    input_ids = torch.zeros((1, 2), dtype=torch.int32)
    causal_mask = torch.zeros((1, 1, 2, 5), dtype=torch.float32)
    traced_model = torch.jit.trace(torch_model, [input_ids, causal_mask])
    kv_cache_shape = torch_model.kv_cache_shape
    del torch_model

    print("Converting traced model to Core ML (stateful, iOS18+)...")
    query_length = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=max_context_size, default=1)
    inputs: List[ct.TensorType] = [
        ct.TensorType(shape=(1, query_length), dtype=np.int32, name="inputIds"),
        ct.TensorType(
            shape=(1, 1, query_length, end_step_dim), dtype=np.float16, name="causalMask"
        ),
    ]
    outputs: List[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
    states: List[ct.StateType] = [
        ct.StateType(
            wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16), name="keyCache"
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16), name="valueCache"
        ),
    ]

    mlmodel_fp16 = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        skip_model_load=True,
    )
    del traced_model

    fp16_path = os.path.join(OUT_DIR, "MentalHealthCompanion.mlpackage")
    print(f"Saving FP16 Core ML model to {fp16_path} ...")
    mlmodel_fp16.save(fp16_path)

    print("Quantizing weights to INT4 (per-block, block_size=32)...")
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int4", granularity="per_block", block_size=32
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel_fp16, config=config)
    mlmodel_int4._spec.description.metadata.userDefined.update({METADATA_TOKENIZER: MODEL_PATH})
    del mlmodel_fp16

    int4_path = os.path.join(OUT_DIR, "MentalHealthCompanionInt4.mlpackage")
    print(f"Saving INT4 Core ML model to {int4_path} ...")
    mlmodel_int4.save(int4_path)
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--max-context-size", type=int, default=2048)
    args = parser.parse_args()

    if args.sanity_check_only:
        sanity_check_against_reference(max_context_size=512)
    else:
        export(max_context_size=args.max_context_size)
