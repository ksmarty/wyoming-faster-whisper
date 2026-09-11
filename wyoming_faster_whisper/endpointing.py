"""Optional server-side voice-command endpoint detection."""

from typing import List, Optional, Protocol

import numpy as np
from numpy.typing import NDArray

SAMPLE_RATE = 16000
SAMPLES_PER_VAD_CHUNK = 512
SECONDS_PER_VAD_CHUNK = SAMPLES_PER_VAD_CHUNK / SAMPLE_RATE


class SpeechProbabilityDetector(Protocol):
    """Stateful detector that scores one 16 kHz mono sample window."""

    def reset(self) -> None:
        pass

    def process_samples(self, samples: List[float]) -> float:
        pass


class VoiceCommandSegmenter:
    """Detect the end of speech without treating brief pauses as an endpoint."""

    def __init__(
        self,
        silence_seconds: float,
        *,
        speech_seconds: float = 0.3,
        command_seconds: float = 1.0,
        before_command_speech_threshold: float = 0.2,
        in_command_speech_threshold: float = 0.5,
    ) -> None:
        if silence_seconds <= 0:
            raise ValueError("silence_seconds must be greater than zero")
        if speech_seconds <= 0:
            raise ValueError("speech_seconds must be greater than zero")
        if command_seconds < speech_seconds:
            raise ValueError("command_seconds must be at least speech_seconds")

        self.silence_seconds = silence_seconds
        self.speech_seconds = speech_seconds
        self.command_seconds = command_seconds
        self.before_command_speech_threshold = before_command_speech_threshold
        self.in_command_speech_threshold = in_command_speech_threshold
        self.in_command = False
        self._speech_seconds_left = 0.0
        self._command_seconds_left = 0.0
        self._silence_seconds_left = 0.0
        self.reset()

    def reset(self) -> None:
        """Reset state for a new audio stream."""
        self.in_command = False
        self._speech_seconds_left = self.speech_seconds
        self._command_seconds_left = self.command_seconds - self.speech_seconds
        self._silence_seconds_left = self.silence_seconds

    def process(self, chunk_seconds: float, speech_probability: float) -> bool:
        """Return False when a started voice command has ended."""
        if not self.in_command:
            if speech_probability > self.before_command_speech_threshold:
                self._speech_seconds_left -= chunk_seconds
                if self._speech_seconds_left <= 0:
                    self.in_command = True
            else:
                self._speech_seconds_left = self.speech_seconds
            return True

        self._command_seconds_left -= chunk_seconds
        if speech_probability > self.in_command_speech_threshold:
            self._silence_seconds_left = self.silence_seconds
        else:
            self._silence_seconds_left -= chunk_seconds

        if self._command_seconds_left <= 0 and self._silence_seconds_left <= 0:
            self.reset()
            return False

        return True


class SileroEndpointDetector:
    """Detect endpoints in normalized 16 kHz, 16-bit, mono PCM."""

    def __init__(
        self,
        silence_seconds: float,
        detector: Optional[SpeechProbabilityDetector] = None,
    ) -> None:
        if detector is None:
            from pysilero_vad import SileroVoiceActivityDetector

            detector = SileroVoiceActivityDetector()

        self._vad = detector
        self._segmenter = VoiceCommandSegmenter(silence_seconds)
        self._pending = np.empty(0, dtype=np.float32)

    def reset(self) -> None:
        """Reset VAD state for a new Wyoming audio stream."""
        self._vad.reset()
        self._segmenter.reset()
        self._pending = np.empty(0, dtype=np.float32)

    def process(self, audio: bytes) -> bool:
        """Return False when enough post-command silence has been observed."""
        samples = _decode_pcm(audio)
        if not samples.size:
            return True

        self._pending = np.concatenate((self._pending, samples))
        offset = 0
        while (offset + SAMPLES_PER_VAD_CHUNK) <= self._pending.size:
            vad_chunk = self._pending[offset : offset + SAMPLES_PER_VAD_CHUNK]
            probability = float(self._vad.process_samples(vad_chunk.tolist()))
            offset += SAMPLES_PER_VAD_CHUNK
            if not self._segmenter.process(SECONDS_PER_VAD_CHUNK, probability):
                self._pending = np.empty(0, dtype=np.float32)
                return False

        if offset:
            self._pending = self._pending[offset:].copy()
        return True


def _decode_pcm(audio: bytes) -> NDArray[np.float32]:
    """Decode normalized 16-bit PCM to float32."""
    if len(audio) % 2:
        raise ValueError("audio chunk contains a partial sample")

    samples = np.frombuffer(audio, dtype="<i2").astype(np.float32)
    samples /= 32768.0
    return samples
