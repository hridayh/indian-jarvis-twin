"""
TTS providers — two options, both fully local:

  mms   (default)  facebook/mms-tts-hin via HuggingFace Transformers
                   ~350 MB, Hindi-native, fast on RTX 2070, no extra install beyond
                   transformers + soundfile (already pulled in by faster-whisper deps).

  coqui            Coqui XTTS v2 — higher quality, multilingual (hi/mr/ta/en),
                   but ~4 GB + torch. Set JARVIS_TTS_PROVIDER=coqui to use.

Usage (factory):
    tts = build_tts(provider="mms")
    wav = tts.synthesize_wav("Sir, stock low hai.", language="hi")
    b64 = tts.wav_bytes_to_b64(wav)
"""
from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger(__name__)


# ── Shared mixin ─────────────────────────────────────────────────────────────

class _TTSBase:
    @staticmethod
    def wav_bytes_to_b64(wav_bytes: bytes) -> str:
        return base64.b64encode(wav_bytes).decode("ascii")


# ── MMS Hindi TTS (lightweight, default) ─────────────────────────────────────

class MMSTTSHindi(_TTSBase):
    """
    Facebook MMS-TTS for Hindi (facebook/mms-tts-hin).
    ~350 MB model, GPU-accelerated on RTX 2070, offline after first download.
    Falls back to CPU if CUDA unavailable.
    """

    MODEL_ID = "facebook/mms-tts-hin"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._sample_rate: int = 16000

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import VitsModel, AutoTokenizer
            import torch
        except ImportError as e:
            raise RuntimeError(
                "transformers / torch not installed. Run: pip install transformers soundfile"
            ) from e

        logger.info("[TTS/MMS] Downloading/loading model '%s' (first run ~350 MB)…", self.MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = VitsModel.from_pretrained(self.MODEL_ID)
        self._sample_rate = self._model.config.sampling_rate

        import torch
        if torch.cuda.is_available():
            logger.info("[TTS/MMS] Calling .cuda() on model — cudnnGetLibConfig warning may appear below…")
            self._model = self._model.cuda()
            logger.info("[TTS/MMS] .cuda() returned — model IS on GPU")
            logger.info("[TTS/MMS] Model loaded on CUDA")
        else:
            logger.info("[TTS/MMS] Model loaded on CPU")

    def synthesize_wav(self, text: str, *, language: str = "hi", **_) -> bytes:
        logger.info("[TTS/MMS] Synthesizing: %r", text[:80])
        self._lazy_load()
        import torch, soundfile as sf

        logger.info("[TTS/MMS] Step A — tokenizing text (%d chars)", len(text))
        inputs = self._tokenizer(text, return_tensors="pt")
        logger.info("[TTS/MMS] Step A — done. input_ids shape: %s", list(inputs["input_ids"].shape))

        on_cuda = torch.cuda.is_available() and next(self._model.parameters()).is_cuda
        logger.info("[TTS/MMS] Step B — model on_cuda=%s", on_cuda)

        if on_cuda:
            logger.info("[TTS/MMS] Step B — moving inputs to CUDA…")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            logger.info("[TTS/MMS] Step B — inputs on CUDA ✅")

        logger.info("[TTS/MMS] Step C — running forward pass (cuDNN disabled to avoid DLL crash)…")
        # cuDNN on this machine crashes with "cudnnGetLibConfig Error 127" — a Windows DLL
        # version mismatch between the cuDNN used at build time vs runtime.
        # Disabling cuDNN makes PyTorch use plain CUDA kernels instead: still GPU-accelerated,
        # avoids the crash, and is only ~10% slower than cuDNN for this small model.
        _prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            with torch.no_grad():
                output = self._model(**inputs).waveform
            logger.info("[TTS/MMS] Step C — forward pass done ✅  waveform shape: %s",
                        list(output.shape))
        except Exception as cuda_err:
            logger.warning("[TTS/MMS] Step C — CUDA failed (%s: %s), falling back to CPU…",
                           type(cuda_err).__name__, cuda_err)
            self._model = self._model.cpu()
            inputs = {k: v.cpu() for k, v in inputs.items()}
            with torch.no_grad():
                output = self._model(**inputs).waveform
            logger.info("[TTS/MMS] Step C — CPU fallback succeeded ✅")
        finally:
            torch.backends.cudnn.enabled = _prev_cudnn

        logger.info("[TTS/MMS] Step D — converting waveform to WAV bytes…")
        wav_np = output.squeeze().cpu().float().numpy()
        buf = io.BytesIO()
        sf.write(buf, wav_np, samplerate=self._sample_rate, format="WAV")
        buf.seek(0)
        wav_bytes = buf.read()
        logger.info("[TTS/MMS] ✅ Synthesized %d bytes of WAV audio", len(wav_bytes))
        return wav_bytes


# ── Coqui XTTS v2 (high quality, heavy) ──────────────────────────────────────

class CoquiTTS(_TTSBase):
    """
    Coqui XTTS v2 — multilingual (hi/mr/ta/en/...).
    Requires: pip install TTS torch   (~4 GB total).
    Set JARVIS_TTS_PROVIDER=coqui to use.
    """

    def __init__(self, *, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"):
        self.model_name = model_name
        self._tts = None

    def _lazy_tts(self):
        if self._tts is not None:
            return self._tts
        try:
            from TTS.api import TTS
        except ImportError as e:
            raise RuntimeError(
                "Coqui TTS not installed. Run: pip install TTS"
            ) from e
        self._tts = TTS(self.model_name)
        return self._tts

    def synthesize_wav(
        self,
        text: str,
        *,
        language: str = "hi",
        speaker_wav: str | None = None,
        **_,
    ) -> bytes:
        import tempfile, os
        tts = self._lazy_tts()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            tmp = f.name
        try:
            kwargs: dict = {"language": language}
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            tts.tts_to_file(text=text, file_path=tmp, **kwargs)
            with open(tmp, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


# ── Factory ───────────────────────────────────────────────────────────────────

def build_tts(provider: str, model_name: str | None = None) -> _TTSBase:
    """
    provider:
        "mms"   → MMSTTSHindi  (default, ~350 MB, offline, RTX 2070 friendly)
        "coqui" → CoquiTTS     (high quality, ~4 GB)
    """
    p = provider.strip().lower()
    if p == "mms":
        return MMSTTSHindi()
    if p == "coqui":
        return CoquiTTS(
            model_name=model_name or "tts_models/multilingual/multi-dataset/xtts_v2"
        )
    raise ValueError(f"Unknown TTS provider '{provider}'. Choose 'mms' or 'coqui'.")
