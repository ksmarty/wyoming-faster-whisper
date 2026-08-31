# Changelog

## Unreleased

- Home Assistant names now fill the prompt budget in four priority tiers: areas/floors that hold an exposed entity, then entity names in the domains people say out loud (`PRIORITY_DOMAINS`: light, switch, fan, media_player, climate, cover, lock, scene, script, todo, vacuum), then the remaining areas/floors, then the remaining entity names
  - Aliases are no longer ranked below every name. An alias is what the speaker says *instead of* the entity's name, so each one now sits directly behind the name it belongs to and shares its tier — previously a home with enough areas and entities to fill the budget lost every alias
  - An entity's area is resolved through its device when it has no area of its own, so the device registry is read too — only when an exposed entity actually needs it
  - In a home that overruns the budget, this stops exposed-by-the-hundred sensors and empty areas from crowding out the lights, scenes and media players a command names

## 3.6.0

- Add `--hass-token` (extra: `hass`) to bias transcription toward the names in Home Assistant: conversation-exposed entity names and aliases, plus area and floor names, read over the websocket API and passed to the model as a prompt (fixes e.g. "What's the temperature of the incubi?" → "What's the temperature of the Ecobee?")
  - Names are refreshed in the background starting at `AudioStart`, so the fetch finishes while the speaker is still talking and adds no latency; a slow or unreachable Home Assistant falls back to the previous names and never fails a transcript
  - Names are added to the prompt in priority order (areas, floors, entity names, aliases) up to `--hass-prompt-max-tokens` (default 200, Whisper's hard cap is 223); `--initial-prompt` is kept at the front
  - Add `--hass-api`, `--hass-refresh-seconds`, `--hass-prompt-max-tokens`, and `--hass-prompt-timeout`

- Add support for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) via `--stt-library qwen3-asr` (extra: `qwen3_asr`), defaulting to [`rhasspy/qwen3-asr-0.6b-onnx-int4-merged`](https://huggingface.co/rhasspy/qwen3-asr-0.6b-onnx-int4-merged)
  - `--initial-prompt` now also biases the Qwen3-ASR backend: it is passed as the model's context prompt, which corrects entity names (e.g. `Vocabulary: Ecobee.` turns "incubator" into "Ecobee")
  - Qwen3-ASR is opt-in only (`auto` never selects it): the model is 785 MB and needs ~1.6 GB of RAM, and it is slower than the per-language defaults

- Qwen3-ASR: support a merged decoder export (`decoder_merged.int4.onnx`) that takes a KV cache and a dynamic sequence length, so the biasing prompt's KV is computed once and reused instead of being re-prefilled every utterance. On a Pi 5 with a 50-name prompt, a 3.2s command goes from 3.42s to 2.20s (1.56x), peak RSS from 2.25 GB to 1.55 GB, and the package from 1407 MB to 785 MB
  - The merged layout is selected automatically when `decoder_merged.int4.onnx` is present; model directories with `decoder_init`/`decoder_step` keep working unchanged
  - The speedup applies to short commands. Long-form audio sees ~1.04x, since the cached prompt is a small share of the work — but the memory saving grows with length (4.16 GB → 2.87 GB on a 30s clip)
  - The default Qwen3-ASR model is now the merged export. On LibriSpeech test-other (n=200) it transcribes byte-identically to the split export with no prompt (5.35% WER for both), and scores 5.33% vs 5.43% with a 50-name prompt — a difference well inside sampling noise. Pass `--model rhasspy/qwen3-asr-0.6b-onnx-int4` for the split export, which stays published

- `--vad-clip` now takes optional speech-to-text libraries, so clipping can be enabled only where it pays off: `--vad-clip qwen3-asr` clips for that backend alone, while a bare `--vad-clip` still applies to every library. Qwen3-ASR costs ~235 ms per second of audio on a Pi 5 (encoder, prefill, *and* per-token decode all scale with input length), so trimming silence off a typical command saves ~20%; faster-whisper pads to 30s internally and gains nothing

- The Docker image now includes the `qwen3-asr` and `hass` extras

## 3.5.0

- Bump torch to avoid regression: https://github.com/pytorch/pytorch/issues/146792
- Use `find_spec` to avoid importing modules for backend check

## 3.4.1

- Use `pysilero-vad>=3.4.0`

## 3.4.0

- Disable VAD by default (use `--vad-clip` to enable)
- Apply `--vad-clip` to all batch backends (not just faster-whisper); clipping happens on the WAV before dispatch. Mainly a latency win for length-proportional backends like sherpa/FunASR on silence-heavy audio; streaming backends are unaffected
- Bump `pysilero-vad` to use GGML version

## 3.3.1

- Ensure zh/yue/ja/ko default to FunASR

## 3.3.0

- Add FunASR speech-to-text backend (`--stt-library funasr`) defaulting to `FunAudioLLM/SenseVoiceSmall` (`@LauraGPT`)
  - Non-autoregressive and notably faster than Whisper; supports English, Chinese, Cantonese, Japanese, and Korean well
  - Install with the `funasr` extra (`pip install '.[funasr]'`)

## 3.2.1

- Fix streaming sherpa cutting off the end of utterances (add tail padding before flushing)
- Default streaming sherpa to the Kroko 2025 zipformer models (mixed-case, punctuated, much better accuracy than the old LibriSpeech model); adds `de`/`es`/`fr` defaults
- Use `--beam-size` for streaming sherpa decoding (beam search when > 1, greedy otherwise)

## 3.2.0

- Fix transformers language
- Add initial prompt to transformers
- Add `--whisper-task` which can be set to "translate" instead of "transcribe" (`@M4TH1EU`)
- Add `--sherpa-streaming` to prefer streaming models (`@pkrahmer`)
- Bump `onnx-asr` to 0.11.0 (supports `istupakov/canary-1b-v2-onnx`)

## 3.1.0

- Refactor to dynamically load models
- Only prefer Parakeet for English (other languages don't detect reliably)
- Add `--vad-filter`, `--vad-threshold`, `--vad-min-speech-ms`, `--vad-min-silence-ms` (thanks @lmoe)
- Add `zeroconf` to Docker image

## 3.0.2

- Set `--data-dir /data` in Docker run script

## 3.0.1

- Fix model auto selection logic

## 3.0.0

- Add support for `sherpa-onnx` and Nvidia's parakeet model
- Add support for [GigaAM](https://github.com/salute-developers/GigaAM) for Russian via [`onnx-asr`](https://github.com/istupakov/onnx-asr)
- Add `--stt-library` to select speech-to-text library (deprecate `--use-transformers`)
- Default `--model` to "auto" (prefer parakeet)
- Add Docker build here
- Default `--language` to "auto"
- Add `--cpu-threads` for faster-whisper (@Zerwin)

## 2.5.0

- Add support for HuggingFace transformers Whisper models (--use-transformers)

## 2.4.0

- Add "auto" for model and beam size (0) to select values based on CPU

## 2.3.0

- Bump faster-whisper package to 1.1.0
- Supports model `turbo` for faster processing

## 2.2.0

- Bump faster-whisper package to 1.0.3

## 2.1.0

- Added `--initial-prompt` (see https://github.com/openai/whisper/discussions/963)

## 2.0.0

- Use faster-whisper PyPI package
- `--model` can now be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

## 1.1.0

- Fix enum use for Python 3.11+
- Add tests and Github actions
- Bump tokenizers to 0.15
- Bump wyoming to 1.5.2

## 1.0.0

- Initial release

