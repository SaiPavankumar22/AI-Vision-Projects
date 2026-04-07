from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Malpractice Detection API"
    environment: str = Field(default="development")
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500"

    look_away_warning_seconds: float = 7.0
    body_turn_warning_seconds: float = 7.0
    head_turn_yaw_threshold: float = 0.06
    body_turn_z_threshold: float = 0.14

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
