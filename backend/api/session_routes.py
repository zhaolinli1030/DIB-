from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional

from ..orchestrator.orchestrator import Orchestrator, get_orchestrator 

router = APIRouter()


class SessionFeedback(BaseModel):
    """会话反馈模型 (Session Feedback Model)"""
    feedback_text: str
    rating: Optional[int] = None

@router.post("/create")
async def create_session(orchestrator: Orchestrator = Depends(get_orchestrator)): # <--- 修改: 使用依赖注入
    """创建新的分析会话 (Create a new analysis session)"""
    session_id = await orchestrator.create_session()
    return {"session_id": session_id}

@router.get("/{session_id}")
async def get_session_info(session_id: str, orchestrator: Orchestrator = Depends(get_orchestrator)): # <--- 修改: 使用依赖注入
    """获取会话信息 (Get session information)"""
    session_data = orchestrator.get_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 只返回必要的会话信息，不包括完整的数据
    # Only return necessary session information, excluding the full data
    return {
        "session_id": session_id,
        "created_at": session_data.get("created_at"),
        "last_active": session_data.get("last_active"),
        "has_data": session_data.get("data") is not None,
        "query_count": len(session_data.get("query_history", []))
    }

@router.post("/{session_id}/feedback")
async def submit_feedback(session_id: str, feedback: SessionFeedback, orchestrator: Orchestrator = Depends(get_orchestrator)): # <--- 修改: 使用依赖注入
    """提交会话反馈 (Submit session feedback)"""
    session_data = orchestrator.get_session_data(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 在实际应用中，这里应该将反馈存储到数据库
    # In a real application, feedback should be stored in a database
    # 对于MVP，我们只返回确认信息
    # For MVP, we just return a confirmation message
    return {
        "message": "Feedback received successfully",
        "session_id": session_id
    }

@router.get("/debug/all-sessions")
async def get_all_sessions(orchestrator: Orchestrator = Depends(get_orchestrator)):
    """获取所有活跃会话（调试用）"""
    try:
        sessions_info = []
        for session_id, session_data in orchestrator.sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "created_at": session_data.get("created_at"),
                "last_active": session_data.get("last_active"),
                "has_data": session_data.get("data") is not None,
                "has_analysis": session_data.get("data_analysis") is not None,
                "data_shape": session_data["data"].shape if session_data.get("data") is not None else None,
                "query_count": len(session_data.get("query_history", [])),
                "persistent": session_data.get("persistent", True)
            })
        
        return {
            "total_sessions": len(sessions_info),
            "sessions": sessions_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sessions: {str(e)}")