# Indian Jarvis — Digital Twin Voice Assistant (Backend Skeleton)

Python/FastAPI backend scaffold for a vertical-focused “Indian Jarvis” that ingests WhatsApp voice notes + CCTV snapshots, extracts a **Business State** (Sales/Credit/Stock), updates a per-client **Digital Twin**, predicts risks (stockout/payment delay), and produces a Hindi/Hinglish voice-note response.

## Quickstart

```bash
cd projects/indian-jarvis-twin
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.main:app --reload
```

## Endpoints

- `POST /ingest/whatsapp/voice-note`
  - Accepts Twilio WhatsApp webhook form fields (`From`, `MediaUrl0`, `MediaContentType0`, …).
  - Downloads the audio, runs STT → LLM extraction, updates the twin, predicts risk, returns recommendation text and (optionally) TTS bytes.
- `POST /ingest/cctv/snapshot`
  - Accepts an image file upload + `client_id`.
  - Runs vision inventory detection, updates the twin, predicts risk, returns recommendation.

## Notes
- State store defaults to **SQLite**. Neo4j integration is scaffolded as an interface so you can swap in later.
- ML pieces (Whisper/Ultralytics/TTS) are wrapped behind small service classes. You can run with mocks first.

## Training risk models (XGBoost)

This repo stores:
- events in SQLite table `events`
- state snapshots in SQLite table `state_snapshots`

To train **real** models you provide labels as CSVs (see `data/labels/` format in `train.py`).

```bash
python src/train.py --db ./jarvis_state.db --out ./models
```
