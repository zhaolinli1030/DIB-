import os
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List

class Settings(BaseSettings):
    # API设置
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "DataInsightBot"
    
    # CORS设置
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",  # Vite默认端口
        "http://127.0.0.1:5173"
    ]
    
    # LLM设置
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
    
    # 文件上传设置
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_FILE_SIZE: str = os.getenv("MAX_FILE_SIZE", "100MB")
    
    # 会话设置
    SESSION_TIMEOUT: int = 60 * 60  # 1小时
    
    # 数据库设置（MVP阶段使用文件系统）
    USE_DATABASE: bool = False
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./datainsight.db")
    
    # 调试设置
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes", "on")

    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # 允许额外的环境变量，但忽略它们
    )

settings = Settings()