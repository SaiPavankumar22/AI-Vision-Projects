from typing import Literal

from pydantic import BaseModel, Field

HeadDirection = Literal["LEFT", "RIGHT", "CENTER", "UNKNOWN"]
BodyDirection = Literal["LEFT", "RIGHT", "STRAIGHT", "UNKNOWN"]
AlertLevel = Literal["OK", "WARNING", "CRITICAL"]


class DetectionMetrics(BaseModel):
    nose_x: float | None = None
    left_shoulder_x: float | None = None
    right_shoulder_x: float | None = None
    processing_ms: float | None = None


class DetectionResponse(BaseModel):
    session_id: str
    head_direction: HeadDirection
    body_position: BodyDirection
    faces_count: int = 0
    alert_level: AlertLevel
    alert_message: str
    violation_count: int = 0
    look_away_duration_sec: float = 0.0
    metrics: DetectionMetrics = Field(default_factory=DetectionMetrics)
