# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JAX-native implementation of Upstage Solar-open MoE for tpu-inference.

Architecture summary (HF reference: transformers/models/solar_open):
- Standard pre-norm decoder.
- GQA self-attention with rotary embeddings; no QK normalization, no
  attention bias.
- Every layer is MoE (config.first_k_dense_replace == 0).
- SolarOpenMoE = TopKRouter (sigmoid + e_score_correction_bias + group-wise
  top-k, identical to DeepSeek-V3) + packed routed experts + a single shared
  expert (SolarOpenMLP with intermediate = moe_intermediate_size *
  n_shared_experts).
- The published Solar-open checkpoint stores experts in the standard HF
  per-expert format (``mlp.experts.<i>.{gate,up,down}_proj.weight``).
  :class:`SolarOpenSharedFusedMoe` overrides ``_load_weights`` to place
  the per-device E shard directly on its TPU device (avoiding the
  full-tensor staging that would otherwise blow HBM at 100B scale).
  Layout matches the standard JaxMoE loader's "names lie" convention
  (gate/up actually ``(E, F, D)``, down actually ``(E, D, F)``) so the
  fused GMM_EP path's ``process_moe_weights`` gets what it expects.

Known limitations of this first cut (intentional, to keep the diff focused):
- BF16 only; no quant_method paths.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference import envs, utils
from tpu_inference.distributed.jax_parallel_state import get_pp_group
from tpu_inference.kernels.experimental.batched_rpa import (
    configs as solar_brpa_configs)
from tpu_inference.kernels.experimental.batched_rpa import (
    wrapper as solar_batched_rpa)
from tpu_inference.layers.common.attention_interface import attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.moe.utils import get_expert_parallelism
from tpu_inference.utils import get_mesh_shape_product
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.norm import JaxRmsNorm
from tpu_inference.layers.jax.pp_utils import PPMissingLayer, make_layers
from tpu_inference.layers.jax.rope_interface import apply_rope
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.logger import init_logger
from tpu_inference.layers.common.utils import cpu_mesh_context
from tpu_inference.models.jax.deepseek_v3 import (DeepseekV3MLP,
                                                  DeepSeekV3Router,
                                                  SharedFusedMoe)
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator, jax_array_from_reshaped_torch)

logger = init_logger(__name__)
init_fn = nnx.initializers.uniform()


# ---------------------------------------------------------------------------
# YaRN RoPE helpers (math from arxiv:2309.00071, see DeepseekScalingRotary
# in tpu_inference.layers.jax.rope for the interleaved-layout variant).
# Solar-open uses HF's half-split rotate convention so we cannot reuse that
# class directly; we just borrow its inv-freq math.
# ---------------------------------------------------------------------------
def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float,
                              max_position_embeddings: int) -> float:
    return (dim *
            math.log(max_position_embeddings /
                     (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(low_rot: float, high_rot: float, dim: int,
                           base: float,
                           max_position_embeddings: int) -> Tuple[int, int]:
    low = int(
        math.floor(
            _yarn_find_correction_dim(low_rot, dim, base,
                                      max_position_embeddings)))
    high = int(
        math.ceil(
            _yarn_find_correction_dim(high_rot, dim, base,
                                      max_position_embeddings)))
    return max(low, 0), min(high, dim - 1)


def _yarn_inv_freq(rotary_dim: int, rope_theta: float, scaling_factor: float,
                   beta_fast: float, beta_slow: float,
                   original_max_position_embeddings: int) -> np.ndarray:
    """YaRN-corrected inverse frequencies for the half-rotation dims.

    Returns a NumPy array. Computed from Python scalars only, so when called
    inside a jit'd forward pass it gets constant-folded by XLA.
    """
    fractions = np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim
    inv_freq_extrapolation = 1.0 / (rope_theta**fractions)
    inv_freq_interpolation = 1.0 / (scaling_factor * rope_theta**fractions)
    low, high = _yarn_correction_range(beta_fast, beta_slow, rotary_dim,
                                       rope_theta,
                                       original_max_position_embeddings)
    if low == high:
        high += 1  # avoid singularity
    ramp = (np.arange(rotary_dim // 2, dtype=np.float32) - low) / (high - low)
    mask = 1.0 - np.clip(ramp, 0.0, 1.0)
    return inv_freq_interpolation * (1 - mask) + inv_freq_extrapolation * mask


def _yarn_mscale(scaling_factor: float, mscale: float = 1.0) -> float:
    """Temperature scaling applied to sin/cos. Identity when scale <= 1."""
    if scaling_factor <= 1.0:
        return 1.0
    return float(0.1 * mscale * math.log(scaling_factor) + 1.0)


def _apply_yarn_rope(x_TNH: jax.Array, positions: jax.Array, head_dim: int,
                     inv_freq: np.ndarray, mscale: float) -> jax.Array:
    """Half-split rotary position embedding with YaRN scaling.

    Matches HF ``rotate_half``:
        out = concat([first*cos - second*sin, second*cos + first*sin])
    where ``first`` and ``second`` are the two halves of the last dim.
    """
    # Take the rotary (= head_dim) prefix; drop any padding the caller added.
    rotary = x_TNH[..., :head_dim]
    pad = x_TNH.shape[-1] - head_dim

    # freqs: (T, head_dim/2)
    freqs = jnp.einsum("T,H->TH",
                       positions.astype(jnp.float32),
                       inv_freq,
                       precision=jax.lax.Precision.HIGHEST)
    cos = jnp.cos(freqs) * mscale
    sin = jnp.sin(freqs) * mscale
    # (T, 1, head_dim/2) for broadcasting over the heads dim.
    cos_T1H = cos[:, None, :]
    sin_T1H = sin[:, None, :]

    first, second = jnp.split(rotary, 2, axis=-1)
    rotated = jnp.concatenate(
        [first * cos_T1H - second * sin_T1H,
         second * cos_T1H + first * sin_T1H],
        axis=-1)
    if pad > 0:
        rotated = jnp.pad(rotated, ((0, 0), (0, 0), (0, pad)))
    return rotated.astype(x_TNH.dtype)


# ---------------------------------------------------------------------------
# Solar-specific attention dispatch (Option A first step: explicit block
# sizes for the experimental batched-RPA kernel).
#
# History: enabling the experimental ``kernels/experimental/batched_rpa``
# kernel cut the synthetic batched-decode probe latency 22.67s -> 19.57s
# (~14%) but the kernel crashed in the real serving path at concurrency
# >= 2 with
#   jax.errors.JaxRuntimeError: INTERNAL: E0200
#   HLO: RPAm-p256-b1-q1-k256.1
# The HLO suffix encodes the block-size config the kernel ran with:
#   p256 = page_size,  b1 = batch_size (internal block batch),
#   q1 = bq_sz,       k256 = bkv_sz.
# Notably ``b1`` (= 1) means the wrapper's ``get_default_block_sizes``
# fell through to its catch-all branch (it has hand-tuned entries only
# for Qwen32b and Qwen-coder; Solar's q=64/kv=8/bf16 shape isn't covered
# and so picks ``BlockSizes(bq_sz=1, bkv_sz=page_size, batch_size=1,
# n_buffer=2)`` — i.e. defeats the kernel's whole reason for existing
# (batching across sequences) and apparently hits a Mosaic bug for
# mixed prefill+decode shapes.
#
# This dispatch passes explicit Solar-tuned BlockSizes so:
#  (a) we exercise the real "batched" path (batch_size > 1),
#  (b) we avoid the crashing default config,
#  (c) we sit on the empirical sweet spot found by the grid search below.
#
# Decode block sweep (256 concurrent, 4k/1k workload, throughput numbers
# are cumulative vs the bs=8 / bkv=512 / nb=2 starting point that already
# achieves ~2x over GMM_EP-without-batched-rpa):
#
#   bs   bkv   nb   decode_lat   throughput      verdict
#   ---  ----  --   ----------   -----------     -------
#    8   512    2     19.43s     +0%   (base)    starting point
#   16   512    2     19.48s    +10%             better
#   32   512    2     19.45s    +13%   WINNER    final config
#   32  1024    2     19.7s     +14%             latency regression, drop
#   32   256    2     19.99s     -5%             worse both, drop
#   32   512    3     19.50s    ~+13% (≈ winner) recovers the bkv=256 dip
#                                                but doesn't beat winner;
#                                                wastes VMEM/SMEM budget
#   64   512    3     CRASH (SMEM OOM, see below)
#
# Final winner: bs=32, bkv=512, nb=2 (~13% above starting point on
# throughput, latency neutral; total ~2.3x over GMM_EP-without-batched-rpa).
#
# Why bs=64 dies: SMEM, not VMEM. The kernel keeps per-sequence metadata
# (page indices, kv_lens, distribution) in SMEM as a "prefetched SMEM
# operand" sized ~batch_size * num_seqs. At bs=64 + 256 sequences this
# operand alone is 512KB and total SMEM hits 1.04M against the 1.00M
# limit (XLA error "Used 1.04M of 1.00M smem"). Don't bother retrying
# bs=64+ with smaller bkv/nb — the operand size depends on bs alone.
#
# Why we stick with nb=2: the nb=3 measurement (~+5% throughput vs the
# bkv=256 low point) is exactly the recovery from the bkv=256 dip — it
# returns to the bs=32 / bkv=512 / nb=2 winner level, not above it. The
# extra buffer just spends VMEM/SMEM with no measurable payoff at this
# bkv. The qwen-coder default uses nb=3 for short-context decode where
# KV-load latency dominates per-block compute; Solar's longer-context
# / larger-block regime saturates the pipe with double-buffering.
#
# Prefill block sweep (single 4k-token request, --batch-size 1
# --output-len 1 isolating one prefill step; latency is the wall-clock
# time of one such step):
#
#   bq    bkv    nb   prefill_lat   verdict
#   ----  -----  --   -----------   -------
#    128   512    2     0.1735s     starting point (Option A inherited)
#    256   512    2     0.1646s     -5% — Phase A WINNER
#    512   512    2     0.1723s     near-baseline regression
#    256  1024    2     0.1640s     -0.4% vs Phase A — Phase B WINNER
#    256  2048    2     0.1820s     worse than baseline (vmem pressure)
#    256  1024    3     0.1650s     ≈ Phase B winner; tps marginally up
#                                   but latency slightly higher and the
#                                   extra buffer just spends VMEM
#
# Final winner: bq=256, bkv=1024, bs=2, nb=2 (~5.5% over starting point).
#
# Why bq=256 wins over 128 and 512: 4k prefill / bq=128 → 32 inner
# iterations vs 16 at bq=256, and the per-block MXU utilisation is the
# same — fewer launches with the same compute is a clean win. bq=512
# doubles the per-block (q × num_q_heads × head_dim) activation; with
# Solar's wide hidden_size and the YaRN RoPE constants riding along,
# vmem/register pressure outweighs the extra amortisation.
#
# Why bkv=1024 helps marginally and bkv=2048 regresses: bkv=1024 reads
# 4 pages per block (page_size=256), still within vmem; bkv=2048 reads
# 8 and either spills or pushes the kernel onto a slower codepath.
#
# Note: kernel batch_size=2 was NOT exercised by the bs=1 profile (only
# 1 prefill seq present per step), so the value is inherited unchanged
# from Option A. If the engine packs concurrent prefill chunks under
# real load, batch_size could matter — re-tune via TTFT benchmark
# rather than this single-seq profile if it becomes a hotspot.
# ---------------------------------------------------------------------------
# Decode (steady-state batched serving): each request contributes 1 query
# token. The kernel processes ``batch_size`` sequences per invocation.
_SOLAR_DECODE_BLOCKS = solar_brpa_configs.BlockSizes(
    bq_sz=1,
    bkv_sz=512,
    batch_size=32,
    n_buffer=2,
)
# Prefill / chunked prefill: queries can be hundreds of tokens long.
_SOLAR_PREFILL_BLOCKS = solar_brpa_configs.BlockSizes(
    bq_sz=256,
    bkv_sz=1024,
    batch_size=2,
    n_buffer=2,
)


def _solar_sharded_batched_rpa(
    mesh: Mesh,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    sm_scale: float,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    """Sharded wrapper around ``batched_rpa.ragged_paged_attention`` with
    Solar-tuned block sizes (default config crashes on mixed batches)."""
    tp_size = get_mesh_shape_product(mesh, ShardingAxisName.ATTN_HEAD)
    if tp_size > 1:
        num_kv_heads = k.shape[1]
        if num_kv_heads < tp_size:
            if tp_size % num_kv_heads != 0:
                raise ValueError(
                    f"For GQA/MQA, tp_size {tp_size} must be divisible by "
                    f"num_kv_heads {num_kv_heads}")
            factor = tp_size // num_kv_heads
            k = jnp.repeat(k, factor, axis=1)
            v = jnp.repeat(v, factor, axis=1)

    qkv_spec = P(ShardingAxisName.ATTN_DATA, ShardingAxisName.ATTN_HEAD, None)
    kv_cache_spec = P(ShardingAxisName.ATTN_DATA, None,
                      ShardingAxisName.ATTN_HEAD, None, None)
    in_specs = (
        qkv_spec,
        qkv_spec,
        qkv_spec,
        kv_cache_spec,
        P(ShardingAxisName.ATTN_DATA),
        P(ShardingAxisName.ATTN_DATA),
        P(ShardingAxisName.ATTN_DATA),
        P(ShardingAxisName.ATTN_DATA),
    )
    out_specs = (qkv_spec, kv_cache_spec)
    args = (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)

    def _call(*args):
        return solar_batched_rpa.ragged_paged_attention(
            *args,
            sm_scale=sm_scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            decode_block_sizes=_SOLAR_DECODE_BLOCKS,
            prefill_block_sizes=_SOLAR_PREFILL_BLOCKS,
        )

    return jax.shard_map(
        _call,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(*args)


def _solar_attention(
    kv_cache: jax.Array,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    attention_metadata: AttentionMetadata,
    mesh: Mesh,
    head_dim_original: int | None = None,
    sm_scale: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """Drop-in replacement for ``attention_interface.attention`` that
    routes Solar through the batched-RPA kernel with explicit Solar
    block sizes."""
    if head_dim_original is None:
        head_dim_original = q.shape[-1]
    if sm_scale is None:
        sm_scale = head_dim_original**-0.5

    md = attention_metadata
    output, kv_cache = _solar_sharded_batched_rpa(
        mesh,
        q,
        k,
        v,
        kv_cache,
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return kv_cache, output


# ---------------------------------------------------------------------------
# Attention: GQA + RoPE, no QK norm, no projection bias.
# ---------------------------------------------------------------------------
class SolarOpenAttention(JaxModule):
    """Multi-query (GQA) attention for Solar-open.

    Closely mirrors :class:`Qwen3Attention` but drops QK normalization (Solar
    does not use it) and assumes ``attention_bias=False``.
    """

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 kv_cache_dtype: str, quant_config: VllmQuantConfig,
                 prefix: str = ""):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.rope_theta = getattr(config, "rope_theta", 10000.0)

        self.head_dim_original = getattr(config, "head_dim",
                                         self.hidden_size // self.num_heads)
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)

        # --- RoPE setup (YaRN if requested by config, otherwise plain) ---
        # We deliberately do NOT cache the inv_freq table on ``self`` — nnx
        # would either reject the numpy array (static slot) or wrap it as
        # data, in which case ``nnx.eval_shape`` abstracts it to a
        # ShapeDtypeStruct that survives into forward and breaks einsum.
        # Instead we keep only Python scalars (which nnx treats as static)
        # and recompute the (small, dim/2-length) inv_freq array inside
        # ``__call__``. XLA folds it into a compile-time constant.
        rope_scaling: Optional[Dict[str, Any]] = getattr(
            config, "rope_scaling", None)
        rope_type = (rope_scaling or {}).get("type") or (rope_scaling
                                                         or {}).get("rope_type")
        self._yarn_enabled: bool = False
        self._yarn_mscale_value: float = 1.0
        self._yarn_scaling_factor: float = 1.0
        self._yarn_original_max: int = 0
        self._yarn_beta_fast: float = 32.0
        self._yarn_beta_slow: float = 1.0
        if rope_scaling and rope_type == "yarn":
            self._yarn_enabled = True
            self._yarn_scaling_factor = float(rope_scaling.get("factor", 1.0))
            self._yarn_original_max = int(
                rope_scaling.get(
                    "original_max_position_embeddings",
                    getattr(config, "max_position_embeddings", 2048)))
            self._yarn_beta_fast = float(rope_scaling.get("beta_fast", 32))
            self._yarn_beta_slow = float(rope_scaling.get("beta_slow", 1))
            mscale_param = float(rope_scaling.get("mscale", 1.0))
            self._yarn_mscale_value = _yarn_mscale(self._yarn_scaling_factor,
                                                   mscale_param)
            # Keep ``self.rope_scaling`` None so the standard ``apply_rope``
            # in __call__ doesn't try to apply *another* (incompatible) llama-
            # style scaling on top of YaRN.
            self.rope_scaling = None
        else:
            self.rope_scaling = rope_scaling

        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.num_kv_heads = utils.get_padded_num_heads(self.num_kv_heads,
                                                       sharding_size)
        self.mesh = mesh

        if envs.LAYOUT_Q_PROJ_AS_NDH:
            rhs_str = "NDH"
            q_proj_sharding = ("model", None, None)
            q_kernel_shape = (self.num_heads, self.hidden_size, self.head_dim)
        else:
            rhs_str = "DNH"
            q_proj_sharding = (None, "model", None)
            q_kernel_shape = (self.hidden_size, self.num_heads, self.head_dim)

        self.q_proj = JaxEinsum(
            f"TD,{rhs_str}->TNH",
            q_kernel_shape,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, q_proj_sharding),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".q_proj",
        )
        self.k_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".k_proj",
        )
        self.v_proj = JaxEinsum(
            "TD,DKH->TKH",
            (self.hidden_size, self.num_kv_heads, self.head_dim),
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".v_proj",
        )
        self.o_proj = JaxEinsum(
            "TNH,NHD->TD",
            (self.num_heads, self.head_dim, self.hidden_size),
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".o_proj",
        )

        self._q_scale = 1.0
        self._k_scale = 1.0
        self._v_scale = 1.0
        self.kv_cache_quantized_dtype = None
        if kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                kv_cache_dtype)

    def __call__(
        self,
        kv_cache: Optional[jax.Array],
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        md = attention_metadata
        q = self.q_proj(x)
        k = self.k_proj(x)
        if self._yarn_enabled:
            # Recompute the (small) inv_freq table from Python scalars at
            # trace time; XLA folds it into a compile-time constant. See the
            # comment in ``__init__`` for why we don't cache it on ``self``.
            inv_freq = _yarn_inv_freq(
                rotary_dim=self.head_dim_original,
                rope_theta=self.rope_theta,
                scaling_factor=self._yarn_scaling_factor,
                beta_fast=self._yarn_beta_fast,
                beta_slow=self._yarn_beta_slow,
                original_max_position_embeddings=self._yarn_original_max,
            )
            q = _apply_yarn_rope(q, md.input_positions, self.head_dim_original,
                                 inv_freq, self._yarn_mscale_value)
            k = _apply_yarn_rope(k, md.input_positions, self.head_dim_original,
                                 inv_freq, self._yarn_mscale_value)
        else:
            q = apply_rope(q, md.input_positions, self.head_dim_original,
                           self.rope_theta, self.rope_scaling)
            k = apply_rope(k, md.input_positions, self.head_dim_original,
                           self.rope_theta, self.rope_scaling)
        v = self.v_proj(x)

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            k_scale = self._k_scale
            v_scale = self._v_scale
            k, v = quantize_kv(self.kv_cache_quantized_dtype, k, v, k_scale,
                               v_scale)

        new_kv_cache, outputs = _solar_attention(
            kv_cache,
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            self.head_dim_original,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        o = self.o_proj(outputs)
        return new_kv_cache, o


# ---------------------------------------------------------------------------
# MoE: routed experts + shared expert.
# ---------------------------------------------------------------------------
class SolarOpenSharedFusedMoe(SharedFusedMoe):
    """Routed-expert MoE with a memory-efficient weight loader for 100B MoE.

    Why we override ``_load_weights``:

    The default ``JaxMoE._load_weights`` concatenates all per-expert tensors
    into a full ``(E, *)`` CPU array and then calls ``shard_put``, which
    under the hood goes through ``jax.device_put`` and stages the whole
    tensor on a single TPU device before redistributing. For Solar-open's
    ``(E=128, F=1280, D=4096)`` kernels that staging step requires ~1.34 GB
    on one device on top of every other already-loaded weight, which OOMs
    partway through loading. ``_build_e_sharded_param`` instead assembles
    the param via ``jax.make_array_from_single_device_arrays`` so each TPU
    device only ever materializes its own ~168 MB E shard
    (``E_local = 16`` experts × ``F * D * 2 bytes`` ≈ 168 MB on an 8-way
    EP mesh).

    Layout: we keep the standard JaxMoE "names lie" convention — the param
    names (``kernel_gating_EDF``, ``kernel_down_proj_EFD``) suggest EDF /
    EFD layouts, but the standard loader never permutes either, so by
    convention the actual stored shapes are reversed: gate/up =
    ``(E, F, D)``, down = ``(E, D, F)``. The fused GMM_EP pipeline
    (``process_moe_weights`` + the megablox kernel) is written against
    that convention; if we permuted on load, ``process_moe_weights`` would
    read the wrong dim from ``w2.shape`` and downstream padding goes
    negative.
    """

    def _load_weights(self, weights):  # type: ignore[override]
        cnt = 0
        for param_name, torch_weight in weights:
            cnt += 1
            tail = param_name.split(self.prefix)[-1]
            names = tail.split(".")
            assert len(names) == 3, (
                f"Expected '<expert_id>.<proj>.weight', got {param_name!r}")
            expert_id_str, param_type, _ = names
            expert_id = int(expert_id_str)
            if param_type.endswith("up_proj"):
                jax_param = self.kernel_up_proj_EDF
            elif param_type.endswith("down_proj"):
                jax_param = self.kernel_down_proj_EFD
            elif param_type.endswith("gate_proj"):
                jax_param = self.kernel_gating_EDF
            else:
                raise ValueError(f"Unexpected MoE proj name in {param_name!r}")
            # We deliberately do NOT permute (0, 2, 1) here. The param
            # *names* (kernel_gating_EDF, kernel_down_proj_EFD) suggest the
            # canonical EDF / EFD layouts, but the standard JaxMoE loader
            # never permutes either, so by convention the actual shapes are
            # reversed: gate/up = (E, F, D), down = (E, D, F). The GMM_EP
            # pipeline (`process_moe_weights` + the megablox kernel) is
            # written against that "names-lie" convention. If we permute,
            # `process_moe_weights` reads the wrong dim from `w2.shape` and
            # downstream padding goes negative ("from 4096 to 1280").
            jax_weight = jax_array_from_reshaped_torch(
                torch_weight,
                reshape_dims=(1, ) + tuple(torch_weight.shape))
            jax_param._weights_to_load[expert_id] = jax_weight

        # Use the sharding specs we passed to JaxMoE explicitly. We can't
        # trust ``param.sharding`` because nnx.Param does not preserve the
        # sharding metadata kwarg in a way that round-trips back as a
        # PartitionSpec — it returns P() (replicated), which would silently
        # blow up HBM by replicating each kernel on every device.
        kernel_to_spec = {
            "kernel_gating_EDF": self.edf_sharding,
            "kernel_up_proj_EDF": self.edf_sharding,
            "kernel_down_proj_EFD": self.efd_sharding,
        }
        # Expected concrete shapes after loading. Param names suggest EDF /
        # EFD, but the standard JaxMoE convention stores them swapped — see
        # the class docstring. Asserting here makes future layout drift
        # surface at load time rather than via cryptic kernel padding errors.
        E = self.num_local_experts
        F = self.intermediate_size_moe
        D = self.hidden_size
        kernel_to_expected_shape = {
            "kernel_gating_EDF": (E, F, D),
            "kernel_up_proj_EDF": (E, F, D),
            "kernel_down_proj_EFD": (E, D, F),
        }
        loaded_names: set[str] = set()
        for kernel_name, param in (
            ("kernel_gating_EDF", self.kernel_gating_EDF),
            ("kernel_up_proj_EDF", self.kernel_up_proj_EDF),
            ("kernel_down_proj_EFD", self.kernel_down_proj_EFD),
        ):
            staged = param._weights_to_load
            if any(w is None for w in staged):
                continue

            spec = kernel_to_spec[kernel_name]
            if isinstance(spec, P):
                named = NamedSharding(self.mesh, spec)
            elif isinstance(spec, tuple):
                named = NamedSharding(self.mesh, P(*spec))
            else:
                named = NamedSharding(self.mesh, P())

            param.value = self._build_e_sharded_param(staged, named)
            expected = kernel_to_expected_shape[kernel_name]
            assert param.value.shape == expected, (
                f"{kernel_name} loaded with shape {param.value.shape}, "
                f"expected {expected} under the JaxMoE 'names lie' "
                f"convention. If you changed the per-expert reshape/permute "
                f"in this loader, update the kernel_to_expected_shape table "
                f"and re-verify the GMM_EP path.")
            # NOTE: do NOT null out ``staged`` here. Standard
            # ``UnquantizedFusedMoEMethod.process_weights_after_loading``
            # gates fusion on
            # ``all(w is not None for w in param._weights_to_load)``;
            # nulling them makes that check fail, so the fused
            # ``kernel_gating_upproj_EDF`` is never created and the forward
            # pass crashes with AttributeError. They get freed naturally
            # when fusion runs and ``del layer.kernel_gating_EDF`` happens.
            loaded_names.add(kernel_name)

        # MEGABLX_GMM (sparse_moe path) layout fix:
        # ``megablox_gmm`` (kernels/megablox/gmm.py) requires the rhs
        # contracted axis to be at index 1, i.e. rhs shape (E, K, N)
        # where K is the contracted feature dim. Solar's "names lie"
        # convention stores gate/up as (E, F, D) and down as (E, D, F),
        # which puts the non-contracted axis at index 1 in both cases.
        # GMM_EP / FUSED_MOE handle this via ``process_moe_weights``
        # (swapaxes(1,2)); MEGABLX_GMM goes through
        # ``UnfusedMoEWeights`` directly so we must do the swap here.
        # axes (1, 2) only — the leading E axis (sharded by EXPERT) is
        # untouched, so the existing sharding spec stays valid.
        if (self.moe_backend == MoEBackend.MEGABLX_GMM
                and len(loaded_names) == 3):
            self.kernel_gating_EDF.value = jnp.swapaxes(
                self.kernel_gating_EDF.value, 1, 2)
            self.kernel_up_proj_EDF.value = jnp.swapaxes(
                self.kernel_up_proj_EDF.value, 1, 2)
            self.kernel_down_proj_EFD.value = jnp.swapaxes(
                self.kernel_down_proj_EFD.value, 1, 2)
        return loaded_names

    def _build_e_sharded_param(self, staged: list,
                               named: NamedSharding) -> jax.Array:
        """Build the param's value, sharded according to ``named``, without
        ever materializing the full ``(E, ...)`` tensor on a single device.

        Per-expert torch_weights have already been staged into ``staged`` as
        small CPU jax.Arrays of shape ``(1, ...)``. For each device, we
        concatenate only the experts that map to that device's E shard and
        ``device_put`` directly onto the target device. The final array is
        assembled with ``make_array_from_single_device_arrays``, which adds
        no extra staging.

        For non-E-sharded specs (replicated, or sharded on dims other than
        dim 0) we fall back to ``make_array_from_callback`` on a host-side
        full-array. We only hit this path for params that are not E-sharded,
        which for this model is none of the routed-expert kernels.
        """
        E = sum(0 if w is None else int(w.shape[0]) for w in staged)
        # Trailing dims come from any per-expert tensor (all should match).
        trailing = tuple(int(d) for d in staged[0].shape[1:])
        full_shape = (E, ) + trailing

        spec = named.spec
        mesh = named.mesh

        # E-sharded fast path: spec is `(<axis_or_axes>, None, None, ...)`.
        e_axis = spec[0] if len(spec) >= 1 else None
        rest_replicated = all(a is None for a in spec[1:])
        if e_axis is not None and rest_replicated:
            axes = (e_axis, ) if isinstance(e_axis, str) else tuple(e_axis)
            n_shards = 1
            for a in axes:
                n_shards *= mesh.shape[a]
            assert E % n_shards == 0, (
                f"Cannot evenly E-shard {E} experts across {n_shards} shards")
            device_to_index = named.devices_indices_map(full_shape)
            per_device_arrays = []
            for device, idx in device_to_index.items():
                start = idx[0].start or 0
                stop = idx[0].stop if idx[0].stop is not None else E
                with cpu_mesh_context():
                    chunk = jnp.concatenate(staged[start:stop], axis=0)
                per_device_arrays.append(jax.device_put(chunk, device))
            return jax.make_array_from_single_device_arrays(
                full_shape, named, per_device_arrays)

        # Generic fallback: replicated or other shardings. Build the full
        # CPU array once (we have no choice) and rely on JAX to slice it.
        with cpu_mesh_context():
            full_cpu = jnp.concatenate(staged, axis=0)
        return jax.make_array_from_callback(
            full_shape, named, lambda idx, _a=full_cpu: _a[idx])


class SolarOpenSparseMoeBlock(JaxModule):
    """``SolarOpenMoE`` block: router + routed experts + 1 shared expert."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        dtype = vllm_config.model_config.dtype
        quant_config = vllm_config.quant_config

        hidden_size = config.hidden_size
        moe_intermediate_size = config.moe_intermediate_size
        n_routed_experts = config.n_routed_experts
        num_experts_per_tok = config.num_experts_per_tok
        n_shared_experts = config.n_shared_experts
        norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0))
        # Solar-open's released config does not set n_group/topk_group; the
        # router code path falls back to flat top-k when n_groups <= 1.
        n_group = int(getattr(config, "n_group", 1) or 1)
        topk_group = int(getattr(config, "topk_group", 1) or 1)
        hidden_act = getattr(config, "hidden_act", "silu")

        # --- Sharding & backend selection ---
        # GMM_EP — empirically the best backend for Solar on Ironwood among
        # all alternatives in tpu-inference. Backend graveyard (every
        # alternative has been measured against GMM_EP and lost):
        #
        # - FUSED_MOE: 8x slower than GMM_EP even with the bt=128 fix in
        #   unquantized.py:_pick_fused_moe_block_sizes. Pallas EP kernel
        #   per-launch overhead too high for Solar's (D=4096, F=1280) shape.
        # - GMM_TP: F=1280 is not 256-aligned -> TPU MXU padding waste.
        # - DENSE_MAT (GptOssMoE-style dense einsum + take_along_axis with
        #   global indices on a sharded axis): ~40% throughput regression.
        #   Compute waste (16 local experts vs ~1 active per chip => 16x
        #   more FFN compute) + implicit huge AG on cross-chip indices.
        #   Pattern only pays off when num_local_experts ≈ top_k (true for
        #   GPT-OSS, false for Solar with top_k=8, E_local=16).
        # - MEGABLX_GMM (sparse_moe_distributed_fwd, replicated-tokens
        #   branch with ragged_all_to_all combine): also slower than
        #   GMM_EP. The local-permute + mask path in sparse_moe.py keeps
        #   the per-chip compute budget similar to GMM_EP's zero-padded
        #   approach but adds permute / a2a overhead, and Solar's
        #   (top_k=8, E_local=16) means most of each chip's local-permuted
        #   tokens still go through the kernel even after masking.
        #
        # Note: the weight-loader swap-axes block below stays harmless for
        # GMM_EP (gated on ``moe_backend == MEGABLX_GMM``); it's preserved
        # so future MEGABLX_GMM re-tests don't have to re-derive the
        # layout fix.
        #
        # The remaining gap to H200 (TPOT 1.48x, throughput 2.4x at 500
        # concurrent) appears to be structural — outside the scope of the
        # MoE-backend choice we can make here.
        expert_axis_name = ShardingAxisName.EXPERT
        num_expert_parallelism = get_expert_parallelism(expert_axis_name, mesh)
        moe_backend = MoEBackend.GMM_EP

        # --- Router ---
        self.gate = DeepSeekV3Router(
            hidden_size=hidden_size,
            num_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            n_groups=n_group,
            topk_groups=topk_group,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
            rngs=rng,
            moe_backend=moe_backend,
            scoring_func="sigmoid",
            activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
            ed_sharding=P(None, None),
            e_sharding=P(None, ),
            quant_config=quant_config,
        )

        # --- Shared experts (always exactly n_shared_experts in this config) ---
        # Replicate shared-expert weights instead of TP-sharding the
        # intermediate dim. Profiling showed that 88 layers x 2 AR/layer
        # (attn o_proj + shared down-proj) account for ~50% of step time
        # in `async-done`. Replication eliminates the shared-expert AR at
        # the cost of an extra ~5GB HBM/device for the (D, F) weights —
        # cheap relative to the 95GB v5p HBM budget. The down-proj output
        # is now already replicated and just adds into the routed-expert
        # output without a collective.
        self.shared_experts = DeepseekV3MLP(
            dtype=dtype,
            hidden_act=hidden_act,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size * n_shared_experts,
            rngs=rng,
            activation_ffw_td=P(ShardingAxisName.MLP_DATA, None),
            df_sharding=P(None, None),
            fd_sharding=P(None, None),
            quant_config=quant_config,
        )

        # --- Routed experts ---
        # On a 2D ('data', 'model') mesh, expert-parallel sharding shards the
        # E dim across 'model' (= ShardingAxisName.EXPERT). The kernel's
        # internal all-to-all on the EP axis handles routing/combining.
        moe_activation_ffw_td = P(ShardingAxisName.MLP_DATA, None)
        moe_activation_ffw_ted = P(ShardingAxisName.MLP_DATA, None, None)
        moe_edf = P(ShardingAxisName.EXPERT, None, None)
        moe_efd = P(ShardingAxisName.EXPERT, None, None)

        self.experts = SolarOpenSharedFusedMoe(
            dtype=dtype,
            num_local_experts=n_routed_experts,
            apply_expert_weight_before_computation=False,
            expert_axis_name=expert_axis_name,
            num_expert_parallelism=num_expert_parallelism,
            hidden_size=hidden_size,
            intermediate_size_moe=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            mesh=mesh,
            hidden_act=hidden_act,
            rngs=rng,
            quant_config=quant_config,
            activation_ffw_td=moe_activation_ffw_td,
            activation_ffw_ted=moe_activation_ffw_ted,
            edf_sharding=moe_edf,
            efd_sharding=moe_efd,
            moe_backend=moe_backend,
            qwix_quantized_weight_dtype=None,
            prefix=f"{prefix}.experts",
            router=self.gate,
            # shared_experts must NOT live under SharedFusedMoe: that would
            # nest the params under `mlp.experts.shared_experts.*`, but the
            # HF checkpoint stores them at `mlp.shared_experts.*`. We keep
            # ``self.shared_experts`` as a direct child of the block and add
            # its output manually in ``__call__``.
            shared_experts=None,
            scoring_func="sigmoid",
            routed_scaling_factor=routed_scaling_factor,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # ``SolarOpenSharedFusedMoe.__call__`` (inherited from SharedFusedMoe)
        # multiplies the routed output by ``routed_scaling_factor`` before
        # combining with shared experts; since we passed ``shared_experts=None``
        # we add the shared expert output here ourselves.
        routed = self.experts(x)
        return routed + self.shared_experts(x)


# ---------------------------------------------------------------------------
# Decoder layer / model / causal LM.
# ---------------------------------------------------------------------------
class SolarOpenDecoderLayer(JaxModule):

    def __init__(self,
                 config,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 kv_cache_dtype: str,
                 quant_config: VllmQuantConfig,
                 layer_idx: int,
                 vllm_config: VllmConfig,
                 prefix: str = ""):
        rms_norm_eps = config.rms_norm_eps
        hidden_size = config.hidden_size

        self.input_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".input_layernorm",
        )
        self.self_attn = SolarOpenAttention(
            config=config,
            dtype=dtype,
            rng=rng,
            mesh=mesh,
            kv_cache_dtype=kv_cache_dtype,
            quant_config=quant_config,
            prefix=prefix + ".self_attn",
        )
        self.post_attention_layernorm = JaxRmsNorm(
            hidden_size,
            epsilon=rms_norm_eps,
            dtype=dtype,
            param_dtype=dtype,
            scale_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            quant_config=quant_config,
            prefix=prefix + ".post_attention_layernorm",
        )

        first_k_dense_replace = int(getattr(config, "first_k_dense_replace", 0))
        if layer_idx < first_k_dense_replace:
            raise NotImplementedError(
                "SolarOpen dense (non-MoE) layers are not implemented; "
                "config.first_k_dense_replace must be 0.")
        self.mlp = SolarOpenSparseMoeBlock(vllm_config=vllm_config,
                                           rng=rng,
                                           mesh=mesh,
                                           prefix=prefix + ".mlp")

    def __call__(
        self,
        kv_cache: jax.Array,
        x: jax.Array,
        attention_metadata: AttentionMetadata,
    ) -> Tuple[jax.Array, jax.Array]:
        residual = x
        hidden_states = self.input_layernorm(x)
        kv_cache, attn_output = self.self_attn(kv_cache, hidden_states,
                                               attention_metadata)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        return kv_cache, hidden_states


class SolarOpenModel(JaxModule):

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model") -> None:
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        vocab_size = model_config.get_vocab_size()
        dtype = model_config.dtype
        rms_norm_eps = hf_config.rms_norm_eps
        hidden_size = hf_config.hidden_size

        self.is_first_rank = get_pp_group().is_first_rank
        self.is_last_rank = get_pp_group().is_last_rank

        if self.is_first_rank or (hf_config.tie_word_embeddings
                                  and self.is_last_rank):
            self.embed_tokens = JaxEmbed(
                num_embeddings=vocab_size,
                features=hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            hf_config.num_hidden_layers,
            lambda layer_index: SolarOpenDecoderLayer(
                config=hf_config,
                dtype=dtype,
                rng=rng,
                mesh=mesh,
                kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                quant_config=vllm_config.quant_config,
                layer_idx=layer_index,
                vllm_config=vllm_config,
                prefix=f"{prefix}.layers.{layer_index}",
            ))

        if self.is_last_rank:
            self.norm = JaxRmsNorm(
                hidden_size,
                epsilon=rms_norm_eps,
                dtype=dtype,
                param_dtype=dtype,
                scale_init=nnx.with_partitioning(init_fn, (None, )),
                rngs=rng,
                quant_config=vllm_config.quant_config,
                prefix=prefix + ".norm",
            )
        else:
            self.norm = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
    ) -> Tuple[List[jax.Array], jax.Array]:
        if self.is_first_rank:
            assert inputs_embeds is None
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            assert inputs_embeds is not None

        x = inputs_embeds
        new_kv_caches: List[jax.Array] = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, PPMissingLayer):
                new_kv_caches.append(kv_caches[i])
                continue
            kv_cache = kv_caches[i]
            kv_cache, x = layer(kv_cache, x, attention_metadata)
            new_kv_caches.append(kv_cache)

        if self.is_last_rank:
            x = self.norm(x)
        return new_kv_caches, x


class SolarOpenForCausalLM(JaxModule, LoadableWithIterator):

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = SolarOpenModel(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix="model",
        )
        model_config = vllm_config.model_config
        if not model_config.hf_config.tie_word_embeddings:
            if self.model.is_last_rank:
                vocab_size = model_config.get_vocab_size()
                hidden_size = model_config.hf_config.hidden_size
                self.lm_head = JaxEinsum(
                    einsum_str="TD,DV->TV",
                    kernel_shape=(hidden_size, vocab_size),
                    dtype=model_config.dtype,
                    param_dtype=model_config.dtype,
                    rngs=rng,
                    kernel_init=nnx.with_partitioning(
                        init_fn, (None, ShardingAxisName.MLP_TENSOR)),
                    quant_config=vllm_config.quant_config,
                    prefix="lm_head",
                )
            else:
                self.lm_head = PPMissingLayer()

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array]]:
        if not is_first_rank:
            assert intermediate_tensors is not None
            inputs_embeds = intermediate_tensors["hidden_states"]
        kv_caches, x = self.model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
        )
        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x}, )
        return kv_caches, x, []

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, "lm_head") and not isinstance(self.lm_head,
                                                       PPMissingLayer):
            return self.lm_head(hidden_states)
        assert isinstance(self.model.embed_tokens, JaxEmbed)
        return self.model.embed_tokens.decode(hidden_states)
