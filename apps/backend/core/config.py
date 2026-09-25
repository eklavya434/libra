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
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
        ],
        alias="CORS_ORIGINS",
        description="Comma/JSON list of allowed browser origins. Empty in production when frontend and backend share one origin.",
    )
    libra_trusted_proxies: int = Field(default=1, alias="LIBRA_TRUSTED_PROXIES")

    # Guest identity & multi-tenant isolation
    libra_session_ttl_days: int = Field(default=30, ge=1, le=365, alias="LIBRA_SESSION_TTL_DAYS")

    # Public-safety gates (mission §16/§18): dangerous features default OFF on
    # the shared deployment. Code execution also requires an explicit operator flag.
    libra_public_code_exec_enabled: bool = Field(default=False, alias="LIBRA_PUBLIC_CODE_EXEC")
    # Execution backend when code execution is enabled:
    #   ""          (default) -> refused: code infrastructure not available
    #   "jail"      -> isolated subprocess worker with wall-clock timeouts and
    #                  OS resource limits where supported (recommended).
    #   "inprocess" -> trusted single-operator mode (executes inside the API
    #                  process; NOT safe for multi-tenant/public deployments).
    libra_code_sandbox: str = Field(default="", alias="LIBRA_CODE_SANDBOX")

    # Rate limiting (per-client-IP token bucket)
    libra_rate_limit_per_minute: int = Field(default=120, alias="LIBRA_RATE_LIMIT_PER_MINUTE")
    libra_rate_limit_burst: int = Field(default=30, alias="LIBRA_RATE_LIMIT_BURST")

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
    kimi_api_key: str = Field(default="", alias="KIMI_API_KEY")
    moonshot_api_key: str = Field(default="", alias="MOONSHOT_API_KEY")
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_deepseek_api_key: str = Field(default="", alias="NVIDIA_DEEPSEEK_API_KEY")
    nvidia_kimi_api_key: str = Field(default="", alias="NVIDIA_KIMI_API_KEY")

    # Local Engine URLs
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    vllm_base_url: str = Field(default="http://localhost:8000", alias="VLLM_BASE_URL")

    # Supabase (Postgres / Storage / optional JWT auth)
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_anon_key: str = Field(default="", alias="SUPABASE_ANON_KEY")
    supabase_service_role_key: str = Field(default="", alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_storage_bucket: str = Field(default="libra-files", alias="SUPABASE_STORAGE_BUCKET")
    supabase_jwt_secret: str = Field(default="", alias="SUPABASE_JWT_SECRET")
    supabase_jwt_audience: str = Field(default="authenticated", alias="SUPABASE_JWT_AUDIENCE")
    # Opt-in: validate `Authorization: Bearer <jwt>` issued by Supabase Auth.
    # Guests remain the default and are always supported.
    supabase_auth_enabled: bool = Field(default=False, alias="LIBRA_SUPABASE_AUTH_ENABLED")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
