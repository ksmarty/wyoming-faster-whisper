"""Tests for the Qwen3-ASR prompt/front-end helpers.

The model files are multi-GB, so these cover the pure logic only: the chat
template the decoder is prompted with, and the mel front-end. Skipped when
onnxruntime is not installed (the handler imports it at module scope).
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

qwen3_asr_handler = pytest.importorskip(
    "wyoming_faster_whisper.qwen3_asr_handler",
    reason="qwen3-asr extra not installed",
)

_build_prompt_ids = qwen3_asr_handler.Qwen3AsrTranscriber._build_prompt_ids
_causal_mask = qwen3_asr_handler._causal_mask
_log_mel_spectrogram = qwen3_asr_handler._log_mel_spectrogram
_mel_filterbank = qwen3_asr_handler._mel_filterbank
_system_ids = qwen3_asr_handler._system_ids
qwen3_asr_language = qwen3_asr_handler.qwen3_asr_language

_AUDIO_PAD = qwen3_asr_handler._AUDIO_PAD
_BLOCKED = qwen3_asr_handler._MASK_BLOCKED


# --- language normalization ------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "English"),
        ("EN", "English"),
        ("en-US", "English"),  # locale-style
        ("zh-CN", "Chinese"),
        ("yue", "Cantonese"),
        ("xx", None),  # unsupported
        (None, None),
        ("", None),
    ],
)
def test_language_normalization(code, expected) -> None:
    assert qwen3_asr_language(code) == expected


# --- prompt construction ---------------------------------------------------


def test_prompt_has_one_audio_pad_per_encoder_frame() -> None:
    ids = _build_prompt_ids(None, 7, [], [])
    assert ids.count(_AUDIO_PAD) == 7


def test_prompt_matches_chat_template_layout() -> None:
    ids = _build_prompt_ids(None, 1, [], [])
    # <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    assert ids[:9] == [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
    # ...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
    assert ids[10:] == [151670, 151645, 198, 151644, 77091, 198]


def test_context_is_placed_in_the_system_turn() -> None:
    context = [1111, 2222]
    ids = _build_prompt_ids(None, 1, context, [])
    # Context sits after "<|im_start|>system\n" and before "<|im_end|>".
    assert ids[:3] == [151644, 8948, 198]
    assert ids[3:5] == context
    assert ids[5] == 151645


def test_forced_language_suffix_goes_after_the_assistant_turn() -> None:
    language_ids = [3333, qwen3_asr_handler._ASR_TEXT]
    ids = _build_prompt_ids(None, 1, [], language_ids)
    assert ids[-2:] == language_ids
    assert ids[-3] == 198  # newline closing "<|im_start|>assistant\n"


# --- merged decoder: the cached span ---------------------------------------


def test_system_turn_is_the_prompts_prefix() -> None:
    """The cached span must be a literal prefix of the prompt, or reusing its KV
    would silently attend to the wrong positions."""
    context = [1111, 2222]
    ids = _build_prompt_ids(None, 3, context, [])
    system = _system_ids(context)
    assert ids[: len(system)] == system


def test_system_turn_holds_the_context_and_no_audio() -> None:
    system = _system_ids([1111, 2222])
    assert system == [151644, 8948, 198, 1111, 2222, 151645, 198]
    # Nothing about the audio may be inside the cached span.
    assert _AUDIO_PAD not in system


def test_system_turn_length_tracks_the_context() -> None:
    assert len(_system_ids([])) == 5
    assert len(_system_ids([1, 2, 3])) == 8


# --- merged decoder: attention mask ----------------------------------------


def test_mask_is_causal_when_there_is_no_cache() -> None:
    mask = _causal_mask(3, 0)
    assert mask.shape == (1, 1, 3, 3)
    # Row i may attend to columns <= i.
    assert (mask[0, 0] == 0).tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_mask_allows_every_cached_position() -> None:
    mask = _causal_mask(2, 4)
    assert mask.shape == (1, 1, 2, 6)
    # The four cached positions are open to both new tokens...
    assert (mask[0, 0, :, :4] == 0).all()
    # ...and the new tokens stay causal among themselves.
    assert mask[0, 0, 0, 4] == 0
    assert mask[0, 0, 0, 5] == _BLOCKED
    assert mask[0, 0, 1, 4] == 0
    assert mask[0, 0, 1, 5] == 0


def test_single_token_step_attends_to_everything() -> None:
    mask = _causal_mask(1, 7)
    assert mask.shape == (1, 1, 1, 8)
    assert (mask == 0).all()


def test_mask_blocks_with_a_large_negative_and_is_float32() -> None:
    mask = _causal_mask(2, 0)
    assert mask.dtype == np.float32
    assert mask[0, 0, 0, 1] == _BLOCKED
    assert _BLOCKED < -1e30


# --- model file selection --------------------------------------------------


class _FakeSession:
    """Enough of an ORT session to get through the constructor."""

    def __init__(self, path, **kwargs) -> None:
        self.path = str(path)
        self.providers = list(kwargs.get("providers") or ["CPUExecutionProvider"])

    def get_providers(self):
        # The real session reports what it actually ended up using, which is how
        # the handler detects a CUDA provider that failed to load.
        return self.providers

    def get_inputs(self):
        return [SimpleNamespace(name="mel")]

    def get_outputs(self):
        # logits, then present_keys/present_values shaped
        # [num_layers, batch, kv_heads, seq, head_dim] with the batch and
        # sequence dims symbolic, as the real export has them.
        kv = SimpleNamespace(name="present_keys", shape=[4, "batch", 2, "seq", 8])
        return [SimpleNamespace(name="logits", shape=["batch", "q", 99]), kv, kv]


def _fake_model_dir(tmp_path, merged: bool):
    """A model directory with the right filenames and no real weights."""
    path = tmp_path / ("merged" if merged else "split")
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"decoder": {"hidden_size": 8}}))
    (path / "tokenizer.json").write_text("{}")
    (path / "embed_tokens.bin").write_bytes(b"")
    names = ["encoder.int4.onnx"]
    names += (
        ["decoder_merged.int4.onnx"]
        if merged
        else ["decoder_init.int4.onnx", "decoder_step.int4.onnx"]
    )
    for name in names:
        (path / name).write_bytes(b"")
    return path


@pytest.fixture(name="stub_runtime")
def stub_runtime_fixture(monkeypatch):
    monkeypatch.setattr(qwen3_asr_handler.ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(
        qwen3_asr_handler, "Tokenizer", SimpleNamespace(from_file=lambda path: object())
    )
    monkeypatch.setattr(
        qwen3_asr_handler.np,
        "fromfile",
        lambda *args, **kwargs: np.zeros(16, dtype=np.float16),
    )


def test_merged_decoder_is_preferred_when_present(tmp_path, stub_runtime) -> None:
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path, merged=True)), str(tmp_path)
    )
    assert model._merged is not None
    assert model._decoder_init is None
    assert model._decoder_step is None


def test_split_decoder_is_used_when_merged_is_absent(tmp_path, stub_runtime) -> None:
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path, merged=False)), str(tmp_path)
    )
    assert model._merged is None
    assert model._decoder_init is not None
    assert model._decoder_step is not None


def test_empty_kv_is_shaped_from_the_graph(tmp_path, stub_runtime) -> None:
    # Taken from present_keys so the cache does not depend on config.json.
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path, merged=True)), str(tmp_path)
    )
    assert model._empty_kv.shape == (4, 1, 2, 0, 8)
    assert model._empty_kv.dtype == np.float32


def test_prefix_cache_starts_empty(tmp_path, stub_runtime) -> None:
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path, merged=True)), str(tmp_path)
    )
    assert model._prefix_key is None
    assert model._prefix_kv is None


def test_allow_patterns_cover_both_decoder_layouts() -> None:
    patterns = set(qwen3_asr_handler._ALLOW_PATTERNS)
    assert qwen3_asr_handler._MERGED_DECODER in patterns
    assert "decoder_merged.int4.onnx.data" in patterns
    # The split layout must still be downloadable.
    assert {"decoder_init.int4.onnx", "decoder_step.int4.onnx"} <= patterns
    assert "decoder_weights.int4.data" in patterns


# --- mel front-end ---------------------------------------------------------


def test_mel_filterbank_shape_and_normalization() -> None:
    filters = _mel_filterbank()
    assert filters.shape == (128, (400 // 2) + 1)
    assert filters.dtype == np.float32
    assert (filters >= 0).all()
    # Every filter carries some weight (no empty bands).
    assert (filters.sum(axis=1) > 0).all()


def test_log_mel_frame_count_and_range() -> None:
    # 1 second of 16kHz audio -> 100 frames (10ms hop), minus the dropped frame.
    audio = np.zeros(16000, dtype=np.float32)
    mel = _log_mel_spectrogram(audio, _mel_filterbank())
    assert mel.shape == (1, 128, 100)
    assert mel.dtype == np.float32
    # The log scale is clamped to an 8-decade window then rescaled.
    assert mel.max() - mel.min() <= 2.0 + 1e-6
