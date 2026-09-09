"""
Libra Backend - Core Configuration Settings
Uses pydantic-settings to validate environment variables safely.
"""

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    # Core API Settings
    app_name: str = "Libra AI Laboratory & Assistant API"
    app_version: str = "0.1.0"
    libra_env: str = Field(default="development", alias="LIBRA_ENV")
    libra_learning_mode: bool = Field(default=True, alias="LIBRA_LEARNING_MODE")
    libra_host: str = Field(default="127.0.0.1", alias="LIBRA_HOST")
    libra_port: int = Field(default=8000, alias="LIBRA_PORT")
    libra_log_level: str = Field(default="INFO", alias="LIBRA_LOG_LEVEL")

    # Security & Networking
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    # Model Storage
    libra_cache_dir: str = Field(default="./models", alias="LIBRA_CACHE_DIR")
    libra_data_dir: str = Field(default="./data", alias="LIBRA_DATA_DIR")
    max_disk_usage_percent: float = Field(default=90.0, alias="MAX_DISK_USAGE_PERCENT")

    # Provider Keys (Optional)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")

    # Local Engine URLs
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    vllm_base_url: str = Field(default="http://localhost:8000", alias="VLLM_BASE_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
