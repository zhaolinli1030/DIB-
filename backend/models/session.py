from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class SessionModel(BaseModel):
    """会话数据模型"""
    
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None  # 存储DataFrame元数据，而不是实际数据
    data_analysis: Optional[Dict[str, Any]] = None
    query_history: List[str] = Field(default_factory=list)
    current_analysis: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True