# AI Malpractice Detection System

Minimal integration-ready version with:

- `backend`: FastAPI API with only detection routes and detection services
- `frontend`: vanilla JavaScript webcam dashboard for manual testing

## Folder Structure

```text
detection system/
├── backend/
│   ├── routes/
│   │   └── routes.py
│   ├── services/
│   │   └── detection_service.py
│   ├── config.py
│   ├── models.py
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── prd.md
```

## Backend Setup

From `detection system/backend`:

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The backend uses `.env` automatically.

## Frontend Setup

From `detection system/frontend`:

```bash
python -m http.server 5500
```

Open [http://localhost:5500](http://localhost:5500).

## API Endpoints

### REST frame analysis

- `POST /api/v1/analyze-frame`
- Multipart fields:
  - `file` (image)
  - `session_id` (optional)

### WebSocket frame analysis

- `WS /api/v1/ws/analyze?session_id=<optional>`
- Send frame bytes (binary) or JSON with `image_base64`

## Detection Behavior

- Multi-person detection alerts are disabled.
- Alert only if behavior is sustained:
  - head turned (`LEFT`/`RIGHT`) for more than `LOOK_AWAY_WARNING_SECONDS` (default 7s)
  - body turned (`LEFT`/`RIGHT`) for more than `BODY_TURN_WARNING_SECONDS` (default 7s)
- Thresholds are configurable in `.env`.

On first run, the model file is auto-downloaded to `backend/models/pose_landmarker_full.task`.

## GitHub Push Notes

Before pushing:

1. Keep `.env` out of Git (already ignored).
2. Commit `.env.example` for shared defaults.
3. Optionally remove generated `backend/models/pose_landmarker_full.task` if already downloaded (also ignored).
