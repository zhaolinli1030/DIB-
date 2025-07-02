from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class AnalysisModel(BaseModel):
    """分析结果数据模型"""
    
    analysis_id: str
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    query: str
    intent: Dict[str, Any]
    result: Dict[str, Any]
    visualization: Dict[str, Any]
    interpretation: Dict[str, Any]
    
    class Config:
        arbitrary_types_allowed = True

class InsightModel(BaseModel):
    """洞察点数据模型"""
    
    title: str
    description: str

class NextStepModel(BaseModel):
    """建议后续分析数据模型"""
    
    title: str
    description: str

class InterpretationModel(BaseModel):
    """解释数据模型"""
    
    summary: str
    insights: List[InsightModel]
    next_steps: List[NextStepModel]