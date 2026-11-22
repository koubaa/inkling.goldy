"""Reference configurations.

Field names are the transformers config attributes, not the checkpoint aliases.
`sconv_kernel_size` maps to `conv_kernel_size`, `intermediate_size` to
`moe_intermediate_size`, `model_max_length` to `max_position_embeddings`, and
`sliding_window` to `sliding_window_size`. The official `loop_config` object
is flattened by `Inkling2LoopTextConfig.__post_init__` into the `loop_*` fields
below. `layer_types` is left unset so the config builds the stock pattern:
`"hybrid"` when `(layer + 1) % 6 == 0`, otherwise `"hybrid_sliding"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Stock L2 clip on a length-128 head: 2 * sqrt(128).
QK_L2_CLIP = 22.627416997969522
REL_L2_CLIP = 8.0


@dataclass(frozen=True)
class Spec:
    """One random-init reference model and the sequence it is dumped on."""

    family: str
    size: str
    seq_len: int
    weight_seed: int
    token_seed: int
    use_tokenizer: bool
    weight_std: float
    clip_scale_multiplier: float
    hf: dict
    stock_windows: dict | None = None
    notes: str = ""
    bias_scale: float = 4.0

    @property
    def name(self) -> str:
        return f"{self.family}/{self.size}"

    @property
    def sliding_window_size(self) -> int:
        return int(self.hf["sliding_window_size"])

    @property
    def rel_extent(self) -> int:
        return int(self.hf["rel_extent"])

    @property
    def log_scaling_n_floor(self) -> int:
        return int(self.hf["log_scaling_n_floor"])


def _common_text(**overrides) -> dict:
    base = {
        "tie_word_embeddings": False,
        "attention_dropout": 0.0,
        "rms_norm_eps": 1e-6,
        "conv_kernel_size": 4,
        "hidden_act": "silu",
        "scoring_func": "sigmoid",
        "route_scale": 8.0,
        # Stock caps are 4 / 25 / 30. At this weight scale those leave the
        # softcap within 1e-5 of the identity, so a missing softcap would pass
        # a 1e-5 check. These sit in the nonlinear region.
        "mlp_gate_softcap": 0.05,
        "mlp_up_softcap": 0.05,
        "final_logit_softcapping": 0.05,
        # Depth-mix unweighted norms stay at the class default 1e-6. This is
        # different on purpose, so a consumer that uses the wrong eps fails.
        "rms_norm_eps": 1e-5,
        # Tighter than the stock 22.627 / 8 so the clip is on the exercised path
        # at the reference weight scale. The formula is the stock one.
        "qk_l2_clip_threshold": 0.15,
        "rel_l2_clip_threshold": 0.35,
        "log_scaling_alpha": 0.1,
        "n_layers_per_attn_res_block": 6,
        "initializer_range": 0.02,
        "max_position_embeddings": 1048576,
        "pad_token_id": None,
        "bos_token_id": None,
    }
    base.update(overrides)
    return base


SPECS: dict[str, Spec] = {}


def _add(spec: Spec) -> None:
    SPECS[spec.name] = spec


_add(
    Spec(
        family="inkling2",
        size="micro",
        seq_len=48,
        weight_seed=1000,
        token_seed=1001,
        use_tokenizer=False,
        weight_std=0.1,
        clip_scale_multiplier=20.0,
        notes=(
            "Twelve layers is two attention-over-depth blocks and two full-attention "
            "layers under the stock 5 sliding + 1 full pattern. Window 8, relative "
            "extent 16 and log-scale floor 4 are all inside a 48-token sequence. "
            "Q/K and relative L2 thresholds are tighter than the stock 22.627 and 8 "
            "so clipping fires at this weight scale; the clip formula is unchanged."
        ),
        hf=_common_text(
            hidden_size=64,
            num_hidden_layers=12,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=2,
            swa_head_dim=32,
            sliding_window_size=8,
            d_rel=8,
            rel_extent=16,
            log_scaling_n_floor=4,
            vocab_size=256,
            unpadded_vocab_size=250,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            eos_token_id=1,
        ),
    )
)

_add(
    Spec(
        family="inkling2",
        size="tiny",
        seq_len=80,
        weight_seed=2000,
        token_seed=2001,
        use_tokenizer=True,
        weight_std=0.05,
        clip_scale_multiplier=20.0,
        bias_scale=50.0,
        notes=(
            "Widths match the ungated tiny-random/inkling-2 text config "
            "(hidden 128, 12 layers, 8/4 heads of 128, 96 experts choose 8, "
            "expert width 128, vocab 201024 padded / 200058 real). Windows are "
            "shrunk so an 80-token sequence (2.5 times the window) crosses the "
            "sliding window, the relative-position extent and the log-scale floor. "
            "Stock window values are recorded and are not what this dump uses."
        ),
        stock_windows={
            "sliding_window_size": 1024,
            "rel_extent": 1024,
            "log_scaling_n_floor": 128000,
            "qk_l2_clip_threshold": QK_L2_CLIP,
            "rel_l2_clip_threshold": REL_L2_CLIP,
            "source": "tiny-random/inkling-2 text_config and Inkling-2-Small",
        },
        hf=_common_text(
            hidden_size=128,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            swa_num_attention_heads=8,
            swa_num_key_value_heads=4,
            swa_head_dim=128,
            sliding_window_size=32,
            d_rel=16,
            rel_extent=64,
            log_scaling_n_floor=16,
            vocab_size=201024,
            unpadded_vocab_size=200058,
            n_routed_experts=96,
            num_experts_per_tok=8,
            moe_intermediate_size=128,
            eos_token_id=200006,
            # Head dim 128 at this weight scale; thresholds sit inside the
            # measured pre-clip norm distribution so some vectors clip.
            qk_l2_clip_threshold=0.12,
            rel_l2_clip_threshold=0.10,
        ),
    )
)

_add(
    Spec(
        family="inkling2_loop",
        size="micro",
        seq_len=48,
        weight_seed=3000,
        token_seed=3001,
        use_tokenizer=False,
        weight_std=0.1,
        clip_scale_multiplier=20.0,
        notes=(
            "Same widths as inkling2/micro. Layers [6, 12) run three times. "
            "That is one pre-loop block and one looped block, and no post-loop "
            "layers: loop bounds have to be positive multiples of "
            "n_layers_per_attn_res_block (6), so a 12-layer stack cannot also "
            "keep a post-loop block. The 24-layer tiny config covers post-loop."
        ),
        hf=_common_text(
            hidden_size=64,
            num_hidden_layers=12,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=2,
            swa_head_dim=32,
            sliding_window_size=8,
            d_rel=8,
            rel_extent=16,
            log_scaling_n_floor=4,
            vocab_size=256,
            unpadded_vocab_size=250,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            eos_token_id=1,
            loop_layer_index_start=6,
            loop_layer_index_end=12,
            loop_num_passes=3,
            loop_gate_rank=16,
        ),
    )
)

_add(
    Spec(
        family="inkling2_loop",
        size="tiny",
        seq_len=80,
        weight_seed=4000,
        token_seed=4001,
        use_tokenizer=True,
        weight_std=0.05,
        clip_scale_multiplier=20.0,
        bias_scale=50.0,
        notes=(
            "Inkling-2-Mini topology with a reduced hidden size: 24 physical "
            "layers, layers [6, 18) run three times, 24/8 heads, 128 experts "
            "choose 8, vocab 201024/200058. Hidden 128, head dim 32 (stock 128), "
            "expert width 64 (stock 1152) and gate rank 32 (stock 64). Windows "
            "match inkling2/tiny so the 80-token sequence crosses every boundary. "
            "Tokenizer and chat template come from tiny-random/inkling-2, which "
            "shares the Mini vocabulary."
        ),
        stock_windows={
            "hidden_size": 2304,
            "head_dim": 128,
            "swa_head_dim": 128,
            "moe_intermediate_size": 1152,
            "loop_gate_rank": 64,
            "sliding_window_size": 1024,
            "rel_extent": 1024,
            "log_scaling_n_floor": 8192,
            "qk_l2_clip_threshold": QK_L2_CLIP,
            "rel_l2_clip_threshold": REL_L2_CLIP,
            "num_hidden_layers": 24,
            "loop_layer_index_start": 6,
            "loop_layer_index_end": 18,
            "loop_num_passes": 3,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "source": "thinkingmachines/Inkling-2-Mini text_config",
        },
        hf=_common_text(
            hidden_size=128,
            num_hidden_layers=24,
            num_attention_heads=24,
            num_key_value_heads=8,
            head_dim=32,
            swa_num_attention_heads=24,
            swa_num_key_value_heads=8,
            swa_head_dim=32,
            sliding_window_size=32,
            d_rel=16,
            rel_extent=64,
            log_scaling_n_floor=16,
            vocab_size=201024,
            unpadded_vocab_size=200058,
            n_routed_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=64,
            eos_token_id=200006,
            loop_layer_index_start=6,
            loop_layer_index_end=18,
            loop_num_passes=3,
            loop_gate_rank=32,
            qk_l2_clip_threshold=0.055,
            rel_l2_clip_threshold=0.05,
        ),
    )
)


# Fixed chat prompt for the tiny configs. The tokenizer is tiny-random/inkling-2.
TINY_PROMPT = "Explain in one sentence why the sky is usually blue."
TOKENIZER_REPO = "tiny-random/inkling-2"
# Hub revision of tiny-random/inkling-2. A moving main branch would change the
# chat-template token ids and every tiny fixture.
TOKENIZER_REVISION = "0a57c72e4e5b68aafe1d983dbb62980344a5fcd7"
