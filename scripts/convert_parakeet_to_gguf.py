#!/usr/bin/env python3
"""Convert a NeMo Parakeet checkpoint to GGUF (f32 / f16 / q8_0).

The GGUF is fully metadata-driven: all config lives in KV, and tensor names are
kept **verbatim** from the NeMo ``state_dict`` (no renaming) so the C++ port is a
1:1 mapping. The two featurizer buffers (``preprocessor.featurizer.fb`` and
``preprocessor.featurizer.window``) are lifted directly from the checkpoint so the
C++ side never re-derives the mel filterbank with librosa.

Quantization (``--dtype f16|q8_0``) is applied **only** to the large linear
weights that the C++ engine consumes directly via ``ggml_mul_mat`` (the encoder
FFN + attention projections, the subsampling output projection, and the joint
enc/pred projections). ggml dequantizes those on the fly inside the compute
graph. Everything the hand-rolled C++ reads as raw F32 (the mel filterbank /
window, the LSTM prediction net, the joint output projection, batch_norm running
stats, conv kernels, embeddings, all norms and biases, pos_bias) stays F32 -- see
``should_quantize`` and ``docs/quantization.md``.

See ``docs/conversion.md`` for the full schema.
"""
import argparse
import pathlib
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np

try:
    import gguf
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'gguf': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

try:
    from nemo.collections.asr.models import ASRModel
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency 'nemo_toolkit[asr]': {e}", file=sys.stderr)
    print("PARAKEET_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


def _get(cfg, key, default=None):
    """Read ``key`` from an OmegaConf node or plain object, tolerating both."""
    try:
        return cfg[key]
    except Exception:
        return getattr(cfg, key, default)


def detect_arch(m):
    """Map a NeMo model to one of ctc/rnnt/tdt/hybrid_rnnt_ctc/hybrid_tdt_ctc."""
    cfg = m.cfg
    # An aux_ctc *config* block is necessary but not sufficient for a hybrid
    # model: prompt-conditioned RNNT checkpoints (nemotron) carry an unconfigured
    # aux_ctc stub (num_classes=-1, empty vocabulary) but NO ctc decoder and zero
    # ctc_decoder.* weights -- NeMo initializes them RNNT-only. Require an actual
    # ctc_decoder on the model (the same module the engine loads ctc_decoder.*
    # tensors from) before classifying as hybrid; otherwise fall through to the
    # rnnt/tdt detection below.
    has_ctc = getattr(m, "ctc_decoder", None) is not None
    if _get(cfg, "aux_ctc") is not None and has_ctc:
        loss = _get(_get(cfg, "loss", {}) or {}, "loss_name", "")
        durs = _get(_get(cfg, "decoding", {}) or {}, "durations")
        return "hybrid_tdt_ctc" if (loss == "tdt" or durs) else "hybrid_rnnt_ctc"
    if _get(cfg, "joint") is not None:
        durs = _get(_get(cfg, "decoding", {}) or {}, "durations")
        nxo = _get(_get(cfg, "joint", {}) or {}, "num_extra_outputs", 0)
        return "tdt" if (durs or (nxo and nxo > 0)) else "rnnt"
    return "ctc"


def prompt_config(cfg):
    """Return (present, num_prompts, dict_keys, dict_vals, default_lang) for a
    prompt-conditioned model, or (False, 0, [], [], "") otherwise. The prompt
    feature lives under cfg.model_defaults (initialize_prompt_feature +
    prompt_dictionary); the projection weights (prompt_kernel.*) are written
    verbatim by the generic tensor loop, so only the KV metadata is new here."""
    md = _get(cfg, "model_defaults", {}) or {}
    if not bool(_get(md, "initialize_prompt_feature", False)):
        return False, 0, [], [], ""
    pdict = _get(md, "prompt_dictionary", None)
    if not pdict:
        return False, 0, [], [], ""
    num = int(_get(md, "num_prompts", 128))
    keys = [str(k) for k in pdict.keys()]
    vals = [int(pdict[k]) for k in pdict.keys()]
    default_lang = "auto" if "auto" in pdict else keys[0]
    return True, num, keys, vals, default_lang


# ---------------------------------------------------------------------------
# Quantization policy.
#
# The C++ engine only tolerates a non-F32 weight when that weight is fed
# *directly* into ``ggml_mul_mat`` (ggml dequantizes f16/q8_0 src0 on the fly).
# Every other weight is read by hand-rolled C++ as a raw ``float*`` (mel
# filterbank/window, LSTM prediction net, joint output projection, batch_norm
# stats, embeddings), or is reshaped/transposed before the matmul in a way that
# does not survive block-quantized storage (the CTC head is stored [1, d, V] and
# squeezed in-graph; conv pointwise weights are reshaped from [1, in, out]).
# Those MUST stay F32 or the engine produces garbage.
#
# Allowlist of weights that are passed verbatim to ggml_mul_mat (see the audit in
# docs/quantization.md). Names are matched after the verbatim NeMo state_dict
# name; "N" is any layer index.
_QUANTIZABLE_PATTERNS = [
    # Conformer feed-forward modules: linear1 (d->ff) and linear2 (ff->d).
    r"^encoder\.layers\.\d+\.feed_forward[12]\.linear[12]\.weight$",
    # Conformer self-attention projections q/k/v/out/pos.
    r"^encoder\.layers\.\d+\.self_attn\.linear_(q|k|v|out|pos)\.weight$",
    # Subsampling output projection (Linear C*F' -> d_model), fed straight to
    # ggml_mul_mat in subsampling.cpp with no reshape.
    r"^encoder\.pre_encode\.out\.weight$",
    # Joint enc/pred projections (ggml_mul_mat in joint.cpp). NOTE: the joint
    # OUTPUT projection joint.joint_net.2.weight is read as a raw float* and
    # stays F32 -- it is intentionally NOT in this allowlist.
    r"^joint\.enc\.weight$",
    r"^joint\.pred\.weight$",
]
_QUANTIZABLE_RE = [re.compile(p) for p in _QUANTIZABLE_PATTERNS]


def should_quantize(name, shape, dtype):
    """Return the ggml quantization type for ``name`` given the requested dtype.

    ``shape`` is the ggml ``ne`` (reverse of the torch shape), so ``shape[0]`` is
    the contraction / leading dimension -- the axis q8_0 blocks along (block
    size 32). Returns ``None`` (keep F32) unless the tensor is on the linear-
    weight allowlist, is at least 2-D with both dims >= 32, and (for q8_0) has a
    leading dimension divisible by the 32-element block size.
    """
    if dtype == "f32":
        return None
    if not any(rx.match(name) for rx in _QUANTIZABLE_RE):
        return None
    if len(shape) < 2 or shape[0] < 32 or shape[1] < 32:
        return None
    if dtype == "f16":
        return gguf.GGMLQuantizationType.F16
    if dtype == "q8_0":
        if shape[0] % 32 != 0:
            return None  # leading dim not block-aligned -> keep F32
        return gguf.GGMLQuantizationType.Q8_0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local .nemo")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--dtype",
        choices=["f32", "f16", "q8_0"],
        default="f32",
        help="quantization for allowlisted linear weights (everything else f32)",
    )
    args = ap.parse_args()

    is_local = pathlib.Path(args.model).exists()
    use_patches = "indic" in args.model.lower() or pathlib.Path("model_config.yaml").exists()

    # Monkey-patch _setup_tokenizer to build AggregateBPE manually
    if use_patches:
        try:
            from omegaconf import open_dict
            import nemo.collections.asr.parts.mixins.mixins as mixins
            class DummyTokenizer:
                def __init__(self, vocab):
                    self.vocab = vocab
                    self.tokenizer = self
                    self.vocab_size = len(vocab)
                    self.token_id_offset = 0
                    self.offset_token_ids_by_token_id = {}
                    self.tokenizers_dict = {'dummy': self}
                    self.langs_by_token_id = {i: "dummy" for i in range(len(vocab))}
                    self.pad_id = 0
                    self.bos_id = 1
                    self.eos_id = 2
                    self.blank_id = len(vocab)
                def get_vocab(self):
                    return self.vocab
                def ids_to_tokens(self, ids):
                    return [self.vocab[i] for i in ids]
                    
            def patched_setup_tokenizer(self, tokenizer_cfg):
                import os
                import yaml
                import tarfile
                
                cfg = None
                if tarfile.is_tarfile(args.model):
                    with tarfile.open(args.model, 'r') as tar:
                        for member in tar.getmembers():
                            if member.name.endswith('model_config.yaml'):
                                f = tar.extractfile(member)
                                if f:
                                    cfg = yaml.safe_load(f)
                                break
                
                if cfg is None:
                    config_path = 'model_config.yaml'
                    if os.path.isdir(args.model):
                        config_path = os.path.join(args.model, 'model_config.yaml')
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cfg = yaml.safe_load(f)
                
                vocabulary = cfg.get('joint', {}).get('vocabulary', None)
                if vocabulary is None:
                    vocabulary = cfg.get('decoder', {}).get('vocabulary', None)
                    
                if vocabulary is None:
                    langs_cfg = cfg['tokenizer']['langs']
                    vocabulary = []
                    for lang, lang_cfg in langs_cfg.items():
                        vocab_path = lang_cfg['vocab_path']
                        if vocab_path.startswith('nemo:'):
                            vocab_path = vocab_path[5:]
                        with open(vocab_path, 'r', encoding='utf-8') as f:
                            lines = f.read().splitlines()
                            vocabulary.extend(lines)
                
                self.tokenizer = DummyTokenizer(vocabulary)
                self.tokenizer_cfg = tokenizer_cfg
                self.tokenizer_type = "agg"
                
            mixins.ASRModuleMixin._setup_tokenizer = patched_setup_tokenizer
        except Exception as e:
            print(f"Failed to monkey-patch setup_tokenizer: {e}", file=sys.stderr)

        # Monkey-patch RNNTDecoder to ignore multisoftmax
        try:
            import nemo.collections.asr.modules.rnnt as rnnt
            original_decoder_init = rnnt.RNNTDecoder.__init__
            def patched_decoder_init(self, *args, **kwargs):
                if 'multisoftmax' in kwargs:
                    del kwargs['multisoftmax']
                original_decoder_init(self, *args, **kwargs)
            rnnt.RNNTDecoder.__init__ = patched_decoder_init
        except Exception as e:
            pass

        # Monkey-patch ConvASRDecoder
        try:
            import nemo.collections.asr.modules.conv_asr as conv_asr
            original_conv_decoder_init = conv_asr.ConvASRDecoder.__init__
            def patched_conv_decoder_init(self, *args, **kwargs):
                if 'multisoftmax' in kwargs:
                    del kwargs['multisoftmax']
                if 'language_keys' in kwargs:
                    del kwargs['language_keys']
                original_conv_decoder_init(self, *args, **kwargs)
            conv_asr.ConvASRDecoder.__init__ = patched_conv_decoder_init
        except Exception as e:
            pass

        # Monkey-patch ConformerEncoder
        try:
            import nemo.collections.asr.modules.conformer_encoder as conformer_encoder
            original_conformer_init = conformer_encoder.ConformerEncoder.__init__
            def patched_conformer_init(self, *args, **kwargs):
                if 'use_bias' in kwargs:
                    del kwargs['use_bias']
                original_conformer_init(self, *args, **kwargs)
            conformer_encoder.ConformerEncoder.__init__ = patched_conformer_init
        except Exception as e:
            pass

        # Monkey-patch RNNTJoint for multilingual models
        try:
            import nemo.collections.asr.modules.rnnt as rnnt
            original_joint_init = rnnt.RNNTJoint.__init__
            def patched_joint_init(self, *args, **kwargs):
                if 'multilingual' in kwargs:
                    del kwargs['multilingual']
                if 'language_keys' in kwargs:
                    del kwargs['language_keys']
                original_joint_init(self, *args, **kwargs)
            rnnt.RNNTJoint.__init__ = patched_joint_init
            
            import torch
            def patched_joint_net_modules(self, num_classes, pred_n_hidden, enc_n_hidden, joint_n_hidden, activation, dropout):
                pred = torch.nn.Linear(pred_n_hidden, joint_n_hidden)
                enc = torch.nn.Linear(enc_n_hidden, joint_n_hidden)
                if activation == 'relu':
                    act = torch.nn.ReLU(inplace=True)
                elif activation == 'sigmoid':
                    act = torch.nn.Sigmoid()
                elif activation == 'tanh':
                    act = torch.nn.Tanh()
                else:
                    act = torch.nn.ReLU(inplace=True)
                languages = ['as', 'bn', 'brx', 'doi', 'kok', 'gu', 'hi', 'kn', 'ks', 'mai', 'ml', 'mr', 'mni', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']
                multilingual_linear = torch.nn.ModuleDict({
                    lang: torch.nn.Linear(joint_n_hidden, 257) for lang in languages
                })
                layers = [act] + ([torch.nn.Dropout(p=dropout)] if dropout else []) + [multilingual_linear]
                return pred, enc, torch.nn.Sequential(*layers)
            rnnt.RNNTJoint._joint_net_modules = patched_joint_net_modules
        except Exception as e:
            print(f"Failed to monkey-patch RNNTJoint: {e}", file=sys.stderr)

    try:
        if is_local:
            import tarfile
            import tempfile
            import torch
            from omegaconf import OmegaConf
            import os

            class DummyModel:
                def __init__(self, nemo_path):
                    self.tmpdir = tempfile.mkdtemp()
                    with tarfile.open(nemo_path, "r") as tar:
                        tar.extractall(self.tmpdir)
                    
                    config_path = None
                    for root, dirs, files in os.walk(self.tmpdir):
                        if 'model_config.yaml' in files:
                            config_path = os.path.join(root, 'model_config.yaml')
                            break
                    self.cfg = OmegaConf.load(config_path)

                    weights_path = None
                    for root, dirs, files in os.walk(self.tmpdir):
                        if 'model_weights.ckpt' in files:
                            weights_path = os.path.join(root, 'model_weights.ckpt')
                            break
                        elif 'model.safetensors' in files:
                            weights_path = os.path.join(root, 'model.safetensors')
                            break
                    
                    if weights_path.endswith('.safetensors'):
                        from safetensors.torch import load_file
                        self._state_dict = load_file(weights_path)
                    else:
                        self._state_dict = torch.load(weights_path, map_location='cpu')
                
                def state_dict(self):
                    return self._state_dict
                
                def eval(self):
                    pass
                
                @property
                def preprocessor(self):
                    cfg = self.cfg.preprocessor
                    hop_len = int(cfg.window_stride * cfg.sample_rate)
                    class Featurizer:
                        def __init__(self):
                            self.n_fft = cfg.n_fft
                            self.n_mels = cfg.features
                            self.nfilt = cfg.features
                            self.hop_length = hop_len
                            self.win_length = int(cfg.window_size * cfg.sample_rate)
                    class Preprocessor:
                        def __init__(self):
                            self.featurizer = Featurizer()
                    return Preprocessor()
                
                @property
                def tokenizer(self):
                    vocab = []
                    if hasattr(self.cfg, 'joint') and hasattr(self.cfg.joint, 'vocabulary'):
                        vocab = list(self.cfg.joint.vocabulary)
                    elif hasattr(self.cfg, 'decoder') and hasattr(self.cfg.decoder, 'vocabulary'):
                        vocab = list(self.cfg.decoder.vocabulary)
                    else:
                        vocab = ["dummy"] * 8192
                    
                    class Tokenizer:
                        def __init__(self):
                            self.vocab_size = len(vocab)
                        def get_vocab(self):
                            return vocab
                        def ids_to_tokens(self, ids):
                            return [vocab[i] for i in ids]
                    return Tokenizer()

            m = DummyModel(args.model)
        else:
            m = ASRModel.from_pretrained(args.model, map_location="cpu")
    except Exception as e:  # pragma: no cover - network/cache guard
        print(f"PARAKEET_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)
    m.eval()

    arch = detect_arch(m)
    cfg = m.cfg
    enc = cfg.encoder
    feat = m.preprocessor.featurizer  # effective runtime values live here

    w = gguf.GGUFWriter(args.output, "parakeet")
    w.add_string("general.name", args.model)
    w.add_string("parakeet.arch", arch)

    # encoder
    w.add_uint32("parakeet.encoder.feat_in", int(_get(enc, "feat_in")))
    w.add_uint32("parakeet.encoder.d_model", int(_get(enc, "d_model")))
    w.add_uint32("parakeet.encoder.n_layers", int(_get(enc, "n_layers")))
    w.add_uint32("parakeet.encoder.n_heads", int(_get(enc, "n_heads")))
    ffx = int(_get(enc, "ff_expansion_factor", 4))
    w.add_uint32("parakeet.encoder.ff_dim", int(_get(enc, "d_model")) * ffx)
    w.add_uint32("parakeet.encoder.conv_kernel", int(_get(enc, "conv_kernel_size")))
    w.add_string("parakeet.encoder.conv_norm_type",
                 str(_get(enc, "conv_norm_type", "batch_norm")))
    w.add_uint32("parakeet.encoder.subsampling_factor",
                 int(_get(enc, "subsampling_factor")))
    sub_ch = int(_get(enc, "subsampling_conv_channels"))
    if sub_ch == -1:
        sub_ch = int(_get(enc, "d_model"))
    w.add_uint32("parakeet.encoder.subsampling_conv_channels", sub_ch)
    w.add_bool("parakeet.encoder.xscaling", bool(_get(enc, "xscaling", True)))
    w.add_uint32("parakeet.encoder.pos_emb_max_len",
                 int(_get(enc, "pos_emb_max_len", 5000)))

    # encoder bias flag (use_bias=False checkpoints omit the linear biases; the
    # C++ loader reads them with clone_weight_opt and tolerates absence).
    w.add_bool("parakeet.encoder.use_bias", bool(_get(enc, "use_bias", True)))

    # --- Prompt conditioning (multilingual nemotron) ------------------------
    # Orthogonal capability flag (like streaming.present). When present, the C++
    # engine inserts the prompt_kernel (Linear->ReLU->Linear) on the encoder
    # output, selected by a one-hot language vector resolved from target_lang.
    p_present, p_num, p_keys, p_vals, p_default = prompt_config(cfg)
    if p_present:
        w.add_bool("parakeet.prompt.present", True)
        w.add_uint32("parakeet.prompt.num_prompts", p_num)
        w.add_array("parakeet.prompt.dictionary.keys", p_keys)
        w.add_array("parakeet.prompt.dictionary.values", p_vals)
        w.add_string("parakeet.prompt.default_lang", p_default)

    # --- Cache-aware streaming / causal config (Phase 5) ---------------------
    # These KVs describe the chunked-limited attention + causal conv that the
    # streaming FastConformer (e.g. parakeet_realtime_eou_120m-v1) uses. They are
    # emitted ONLY for streaming models (att_context_style != "regular") so that
    # offline checkpoints continue to convert byte-identically; the C++ loader
    # supplies offline-safe defaults (style "regular", causal flags false,
    # streaming block absent) when these keys are missing.
    att_style = str(_get(enc, "att_context_style", "regular"))
    is_streaming = att_style != "regular"
    if is_streaming:
        # att_context_size = [left, right]; streaming models use finite values
        # (e.g. [70, 1]) while offline models use [-1, -1]. Stored as signed
        # int32 so the -1 sentinel survives if a streaming model ever uses it;
        # the loader reads them as int32 and defaults to -1 when absent.
        att_ctx = _get(enc, "att_context_size", [-1, -1]) or [-1, -1]
        # Multi-context models store a LIST of [left,right] presets; the default
        # is the first (NeMo's default att_context_size index). A flat [l,r]
        # (older streaming models like the eou) is used as-is. The first element
        # being a non-scalar (list/tuple/OmegaConf ListConfig) marks the nested
        # form -- detect it by "not a plain number" rather than an exact type so
        # OmegaConf's ListConfig is handled too.
        if att_ctx and not isinstance(att_ctx[0], (int, float)):
            presets = [[int(x) for x in p] for p in att_ctx]
            att_left, att_right = presets[0][0], presets[0][1]
            # Record all presets so a future latency knob can pick another.
            w.add_array("parakeet.encoder.att_context_presets",
                        [int(v) for p in presets for v in p])  # flattened [l,r,l,r,...]
        else:
            att_ctx = [int(x) for x in att_ctx]
            att_left = att_ctx[0] if len(att_ctx) > 0 else -1
            att_right = att_ctx[1] if len(att_ctx) > 1 else -1
        w.add_int32("parakeet.encoder.att_context_left", int(att_left))
        w.add_int32("parakeet.encoder.att_context_right", int(att_right))
        w.add_string("parakeet.encoder.att_context_style", att_style)
        w.add_bool("parakeet.encoder.causal_downsampling",
                   bool(_get(enc, "causal_downsampling", False)))
        # conv_context_size == "causal" (a string) means the depthwise conv uses
        # left-only padding; a list of two ints means symmetric/explicit padding.
        conv_ctx = _get(enc, "conv_context_size", None)
        conv_causal = isinstance(conv_ctx, str) and conv_ctx == "causal"
        w.add_bool("parakeet.encoder.conv_causal", bool(conv_causal))

        # Streaming params read straight off the live encoder's streaming_cfg
        # (populated by setup_streaming_params() in __init__). List fields
        # (chunk_size/shift_size/pre_encode_cache_size) are emitted as int32
        # arrays; scalar fields as int32. Verified field names against
        # CacheAwareStreamingConfig in models/configs/asr_models_config.py.
        m.encoder.setup_streaming_params()
        sc = m.encoder.streaming_cfg

        def _int_list(v):
            return [int(x) for x in (v if isinstance(v, (list, tuple)) else [v])]

        w.add_array("parakeet.streaming.chunk_size", _int_list(sc.chunk_size))
        w.add_array("parakeet.streaming.shift_size", _int_list(sc.shift_size))
        w.add_int32("parakeet.streaming.cache_drop_size", int(sc.cache_drop_size))
        w.add_int32("parakeet.streaming.last_channel_cache_size",
                    int(sc.last_channel_cache_size))
        w.add_int32("parakeet.streaming.valid_out_len", int(sc.valid_out_len))
        w.add_array("parakeet.streaming.pre_encode_cache_size",
                    _int_list(sc.pre_encode_cache_size))
        w.add_int32("parakeet.streaming.drop_extra_pre_encoded",
                    int(sc.drop_extra_pre_encoded))

    # preprocessor (effective values off the featurizer object)
    w.add_uint32("parakeet.preprocessor.sample_rate",
                 int(getattr(feat, "sample_rate", 16000)))
    w.add_uint32("parakeet.preprocessor.n_mels", int(getattr(feat, "nfilt")))
    w.add_uint32("parakeet.preprocessor.n_fft", int(getattr(feat, "n_fft")))
    w.add_uint32("parakeet.preprocessor.win_length", int(getattr(feat, "win_length")))
    w.add_uint32("parakeet.preprocessor.hop_length", int(getattr(feat, "hop_length")))
    pre = getattr(feat, "preemph", None)
    w.add_float32("parakeet.preprocessor.preemph", float(pre) if pre is not None else 0.0)
    w.add_float32("parakeet.preprocessor.mag_power",
                  float(getattr(feat, "mag_power", 2.0)))
    w.add_string("parakeet.preprocessor.normalize",
                 str(getattr(feat, "normalize", "per_feature")))
    lzg = getattr(feat, "log_zero_guard_value", None)
    w.add_float32("parakeet.preprocessor.log_zero_guard",
                  float(lzg) if isinstance(lzg, (int, float)) else 2 ** -24)

    # vocab / tokenizer
    vocab = int(m.tokenizer.vocab_size)
    w.add_uint32("parakeet.vocab_size", vocab)
    w.add_uint32("parakeet.blank_id", vocab)  # blank always == vocab_size
    pieces = [m.tokenizer.ids_to_tokens([i])[0] for i in range(vocab)]
    w.add_array("parakeet.tokenizer.pieces", [str(p) for p in pieces])

    # transducer config
    if arch in ("rnnt", "tdt", "hybrid_rnnt_ctc", "hybrid_tdt_ctc"):
        prednet = _get(cfg.decoder, "prednet", {}) or {}
        w.add_uint32("parakeet.decoder.pred_hidden", int(_get(prednet, "pred_hidden")))
        w.add_uint32("parakeet.decoder.pred_rnn_layers",
                     int(_get(prednet, "pred_rnn_layers", 1)))
        jn = _get(cfg.joint, "jointnet", {}) or {}
        w.add_uint32("parakeet.joint.joint_hidden", int(_get(jn, "joint_hidden")))
        w.add_string("parakeet.joint.activation", str(_get(jn, "activation", "relu")))
        # Greedy max symbols emitted per frame (NeMo decoding.greedy.max_symbols;
        # default 10). Emitted so the C++ decoder honors a model's own value
        # instead of a hardcoded literal.
        greedy = _get(_get(cfg, "decoding", {}) or {}, "greedy", {}) or {}
        max_sym = _get(greedy, "max_symbols", _get(greedy, "max_symbols_per_step", 10))
        w.add_uint32("parakeet.decoding.max_symbols", int(max_sym) if max_sym is not None else 10)
    if arch in ("tdt", "hybrid_tdt_ctc"):
        durs = (_get(_get(cfg, "decoding", {}) or {}, "durations")
                or _get(_get(cfg, "model_defaults", {}) or {}, "tdt_durations"))
        if not durs:
            raise ValueError(
                f"arch={arch} requires TDT durations but none found in "
                "cfg.decoding.durations or cfg.model_defaults.tdt_durations"
            )
        w.add_array("parakeet.tdt.durations", [int(d) for d in durs])

    # tensors: verbatim names. Allowlisted linear weights are quantized per
    # --dtype (ggml dequantizes them on the fly inside ggml_mul_mat); everything
    # else stays f32. Include featurizer buffers explicitly.
    sd = m.state_dict()
    
    # Reconstruct the joint.joint_net.2.weight and joint.joint_net.2.bias from the multilingual weights
    languages = ['as', 'bn', 'brx', 'doi', 'kok', 'gu', 'hi', 'kn', 'ks', 'mai', 'ml', 'mr', 'mni', 'ne', 'or', 'pa', 'sa', 'sat', 'sd', 'ta', 'te', 'ur']
    detected_lang = None
    for lang in languages:
        if f"joint.joint_net.2.{lang}.weight" in sd:
            detected_lang = lang
            break
            
    if detected_lang is not None:
        import torch
        joint_weight = torch.zeros(5633, 640)
        joint_bias = torch.zeros(5633)
        
        lang_weight = sd[f"joint.joint_net.2.{detected_lang}.weight"] # shape [257, 640]
        lang_bias = sd[f"joint.joint_net.2.{detected_lang}.bias"] # shape [257]
        
        lang_idx = languages.index(detected_lang)
        offset = lang_idx * 256
        
        joint_weight[offset:offset + 256, :] = lang_weight[0:256, :]
        joint_weight[5632, :] = lang_weight[256, :]
        
        joint_bias[offset:offset + 256] = lang_bias[0:256]
        joint_bias[5632] = lang_bias[256]
        
        sd["joint.joint_net.2.weight"] = joint_weight
        sd["joint.joint_net.2.bias"] = joint_bias

    written = 0
    quantized = 0
    keep_buffers = {"preprocessor.featurizer.fb", "preprocessor.featurizer.window"}
    for name, t in sd.items():
        if name.startswith("preprocessor.") and name not in keep_buffers:
            continue  # skip preprocessor internals except fb/window
        if not hasattr(t, "detach"):
            continue
            
        # Skip language-specific joint weights (the standard joint.joint_net.2.weight/bias were reconstructed beforehand)
        if name.startswith("joint.joint_net.2.") and name not in ("joint.joint_net.2.weight", "joint.joint_net.2.bias"):
            continue
            
        arr = t.detach().cpu().float().numpy()
        if arr.ndim == 0:
            continue  # skip scalar bookkeeping (e.g. num_batches_tracked)
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        # ggml ne is the reverse of the numpy/torch shape; ne[0] is the leading
        # (contraction) axis q8_0 blocks along.
        ggml_ne = list(arr.shape[::-1])
        qtype = should_quantize(name, ggml_ne, args.dtype)
        if qtype is None:
            w.add_tensor(name, arr)
        else:
            raw = gguf.quantize(arr, qtype)
            # gguf expects raw_shape to be the *byte* shape of the quantized
            # buffer; it derives the element shape from it via raw_dtype.
            w.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=qtype)
            quantized += 1
        written += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(
        f"wrote {args.output}: arch={arch} vocab={vocab} tensors={written} "
        f"dtype={args.dtype} quantized={quantized}"
    )


if __name__ == "__main__":
    main()
