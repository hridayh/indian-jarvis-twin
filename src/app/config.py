from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_", env_file=".env", extra="ignore")

    # Persistence
    state_store_sqlite_url: str = "sqlite:///./jarvis_state.db"

    # STT
    stt_model_name: str = "medium"  # for faster-whisper: tiny/base/small/medium/large-v3
    stt_device: str = "cpu"  # cpu/cuda

    # Vision
    vision_model_path: str | None = None  # e.g. "./models/yolo.pt"

    # LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    # TTS (Coqui)
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Security / Twilio validation (optional)
    twilio_auth_token: str | None = None

