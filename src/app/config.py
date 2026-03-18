from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JARVIS_", env_file=".env", extra="ignore")

    # Persistence
    state_store_sqlite_url: str = "sqlite:///./jarvis_state.db"

    # STT
    stt_model_name: str = "small"  # for faster-whisper: tiny/base/small/medium/large-v3
    stt_device: str = "cuda"  # cpu/cuda (will fall back to cpu if cuda init fails)

    # Vision
    vision_model_path: str | None = None  # e.g. "./models/yolo.pt"
    vision_sku_mapping_path: str = "./data/vision/sku_mapping.json"
    low_stock_threshold_pct: float = 20.0

    # LLM (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    # TTS (Coqui)
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"

    # Risk model persistence (trained XGBoost models)
    risk_model_dir: str | None = "./models"

    # Which trained label-column drives the live risk outputs
    # (these must match columns in your labels CSV and the saved model filenames)
    risk_stockout_label: str = "stockout_lowstock_w1"
    risk_payment_delay_label: str = "payment_overdue14_w1"

    # Security / Twilio validation (optional)
    twilio_auth_token: str | None = None

