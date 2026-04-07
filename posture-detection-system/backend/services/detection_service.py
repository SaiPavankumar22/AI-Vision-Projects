from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from config import get_settings


@dataclass
class SessionState:
    look_away_started_at: float | None = None
    body_turn_started_at: float | None = None
    look_away_alert_active: bool = False
    body_turn_alert_active: bool = False
    violation_count: int = 0


class DetectionService:
    def __init__(self) -> None:
        self._pose_landmarker = None
        self._mode = "uninitialized"
        self._sessions: dict[str, SessionState] = {}
        self._model_dir = Path(__file__).resolve().parents[1] / "models"
        self._pose_model_path = self._model_dir / "pose_landmarker_full.task"

    def _get_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState()
        return self._sessions[session_id]

    def _ensure_pose_landmarker(self) -> None:
        if self._mode != "uninitialized":
            return
        self._model_dir.mkdir(parents=True, exist_ok=True)
        if not self._pose_model_path.exists():
            model_url = (
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            )
            urlretrieve(model_url, str(self._pose_model_path))

        base_options = mp.tasks.BaseOptions(model_asset_path=str(self._pose_model_path))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            num_poses=1,
        )
        self._pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._mode = "mediapipe_tasks_pose"

    def _head_from_pose(self, landmarks: list[Any]) -> str:
        settings = get_settings()
        nose = landmarks[0]
        left_ear = landmarks[7]
        right_ear = landmarks[8]
        ears_mid_x = (left_ear.x + right_ear.x) / 2.0
        yaw_delta = nose.x - ears_mid_x
        if yaw_delta < -settings.head_turn_yaw_threshold:
            return "LEFT"
        if yaw_delta > settings.head_turn_yaw_threshold:
            return "RIGHT"
        return "CENTER"

    def _body_from_pose(self, landmarks: list[Any]) -> str:
        settings = get_settings()
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        shoulder_z_delta = left_shoulder.z - right_shoulder.z
        if shoulder_z_delta < -settings.body_turn_z_threshold:
            return "LEFT"
        if shoulder_z_delta > settings.body_turn_z_threshold:
            return "RIGHT"
        return "STRAIGHT"

    def _evaluate_rules(self, state: SessionState, head_direction: str, body_position: str) -> dict[str, Any]:
        settings = get_settings()
        now = monotonic()
        look_away_duration = 0.0
        body_turn_duration = 0.0

        if head_direction in {"LEFT", "RIGHT"}:
            if state.look_away_started_at is None:
                state.look_away_started_at = now
            look_away_duration = max(0.0, now - state.look_away_started_at)
        else:
            state.look_away_started_at = None
            state.look_away_alert_active = False

        if body_position in {"LEFT", "RIGHT"}:
            if state.body_turn_started_at is None:
                state.body_turn_started_at = now
            body_turn_duration = max(0.0, now - state.body_turn_started_at)
        else:
            state.body_turn_started_at = None
            state.body_turn_alert_active = False

        if body_turn_duration > settings.body_turn_warning_seconds:
            if not state.body_turn_alert_active:
                state.violation_count += 1
                state.body_turn_alert_active = True
            return {
                "alert_level": "WARNING",
                "alert_message": "Body turned away for too long",
                "violation_count": state.violation_count,
                "look_away_duration_sec": look_away_duration,
            }

        if look_away_duration > settings.look_away_warning_seconds:
            if not state.look_away_alert_active:
                state.violation_count += 1
                state.look_away_alert_active = True
            return {
                "alert_level": "WARNING",
                "alert_message": "Looking away for too long",
                "violation_count": state.violation_count,
                "look_away_duration_sec": look_away_duration,
            }

        return {
            "alert_level": "OK",
            "alert_message": "Posture and attention are normal",
            "violation_count": state.violation_count,
            "look_away_duration_sec": look_away_duration,
        }

    def analyze(self, frame_bytes: bytes, session_id: str) -> dict[str, Any]:
        start = perf_counter()
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image payload")

        self._ensure_pose_landmarker()
        state = self._get_session(session_id)

        nose_x = None
        left_shoulder_x = None
        right_shoulder_x = None
        faces_count = 1  # multi-person detection intentionally disabled
        head_direction = "UNKNOWN"
        body_position = "UNKNOWN"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._pose_landmarker.detect(mp_image)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            nose_x = float(landmarks[0].x)
            left_shoulder_x = float(landmarks[11].x)
            right_shoulder_x = float(landmarks[12].x)
            head_direction = self._head_from_pose(landmarks)
            body_position = self._body_from_pose(landmarks)
        else:
            faces_count = 0

        rules = self._evaluate_rules(state, head_direction=head_direction, body_position=body_position)
        processing_ms = (perf_counter() - start) * 1000.0
        return {
            "session_id": session_id,
            "head_direction": head_direction,
            "body_position": body_position,
            "faces_count": faces_count,
            "alert_level": rules["alert_level"],
            "alert_message": rules["alert_message"],
            "violation_count": rules["violation_count"],
            "look_away_duration_sec": round(rules["look_away_duration_sec"], 2),
            "metrics": {
                "nose_x": nose_x,
                "left_shoulder_x": left_shoulder_x,
                "right_shoulder_x": right_shoulder_x,
                "processing_ms": round(processing_ms, 2),
            },
        }


detection_service = DetectionService()
