import base64
import json
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status

from models import DetectionResponse
from services.detection_service import detection_service

router = APIRouter(tags=["detection"])


def _decode_payload(payload: str) -> bytes:
    data = json.loads(payload)
    image_data = data.get("image_base64")
    if not image_data:
        raise ValueError("Missing image_base64 in payload")
    return base64.b64decode(image_data)


@router.post("/analyze-frame", response_model=DetectionResponse)
async def analyze_frame(file: UploadFile = File(...), session_id: str | None = Form(default=None)) -> DetectionResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded frame is empty")

    resolved_session = session_id or f"session-{uuid4().hex[:8]}"
    try:
        result = detection_service.analyze(image_bytes, resolved_session)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Frame analysis failed") from exc
    return DetectionResponse(**result)


@router.websocket("/ws/analyze")
async def analyze_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or f"session-{uuid4().hex[:8]}"
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"] is not None:
                image_bytes = message["bytes"]
            elif "text" in message and message["text"] is not None:
                image_bytes = _decode_payload(message["text"])
            else:
                continue

            result = detection_service.analyze(image_bytes, session_id)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1011, reason="Stream processing failed")
